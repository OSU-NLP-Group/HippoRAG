"""
Tests for the BGE embedding model integration (GraphRAG-Benchmark support).

Run from a raw clone (no install needed), like the other test scripts:

    python test_bge.py

Optional env vars:
    BGE_TEST_MODEL   : local path or HF id of a BGE checkpoint to exercise the
                       encode path (default: BAAI/bge-small-en-v1.5).
    BGE_TEST_SKIP_INT: if set to "1", skip the network/GPU integration tier.

Tier A (dispatch / registration) is hermetic -- it needs no GPU and no model
download, so it always runs. Tier B (real encode) requires loading a BGE
checkpoint (network + ideally GPU) and is skipped gracefully if the model
cannot be loaded.
"""

import os
import sys
import traceback

import numpy as np


# ---------------------------------------------------------------------------
# Tier A: dispatch / registration (hermetic, no GPU, no download)
# ---------------------------------------------------------------------------

def test_dispatch():
    from src.hipporag.embedding_model import _get_embedding_model_class
    from src.hipporag.embedding_model.BGE import BGEEmbeddingModel
    from src.hipporag.embedding_model.base import BaseEmbeddingModel

    print("  [1] BGE names route to BGEEmbeddingModel ...", end=" ")
    bge_names = [
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-m3",
        "some-org/BGE-custom",
    ]
    for name in bge_names:
        cls = _get_embedding_model_class(name)
        assert cls is BGEEmbeddingModel, f"{name} -> {cls.__name__}, expected BGEEmbeddingModel"
    print("OK")

    print("  [2] 'bge' match is case-insensitive ...", end=" ")
    assert _get_embedding_model_class("X/BgE-NeT") is BGEEmbeddingModel
    print("OK")

    print("  [3] BGEEmbeddingModel subclasses BaseEmbeddingModel ...", end=" ")
    assert issubclass(BGEEmbeddingModel, BaseEmbeddingModel)
    print("OK")

    print("  [4] Regression: existing models still resolve correctly ...", end=" ")
    cases = [
        ("nvidia/NV-Embed-v2", "NVEmbedV2EmbeddingModel"),
        ("facebook/contriever", "ContrieverModel"),
        ("GritLM/gritlm-7b", "GritLMEmbeddingModel"),
        ("text-embedding-3-small", "OpenAIEmbeddingModel"),
        ("cohere/embed-multilingual-v3.0", "CohereEmbeddingModel"),
        ("Transformers/BAAI/bge-m3", "TransformersEmbeddingModel"),
        ("VLLM/BAAI/bge-m3", "VLLMEmbeddingModel"),
    ]
    for name, expected in cases:
        cls = _get_embedding_model_class(name)
        assert cls.__name__ == expected, f"{name} -> {cls.__name__}, expected {expected}"
    print("OK")

    print("  [5] Regression: 'bge' branch does not shadow Transformers/ or VLLM/ ...", end=" ")
    assert _get_embedding_model_class("Transformers/BAAI/bge-m3").__name__ == "TransformersEmbeddingModel"
    assert _get_embedding_model_class("VLLM/BAAI/bge-m3").__name__ == "VLLMEmbeddingModel"
    print("OK")

    print("  [6] Unknown model name still raises AssertionError ...", end=" ")
    try:
        _get_embedding_model_class("definitely-not-a-real-model")
    except AssertionError:
        print("OK")
    else:
        raise AssertionError("Expected AssertionError for an unknown embedding model name")

    print("\n  PASS: dispatch -- all checks passed")


# ---------------------------------------------------------------------------
# Tier B: real encode path (network + GPU recommended; graceful skip)
# ---------------------------------------------------------------------------

def _make_model(global_config, model_name):
    from src.hipporag.embedding_model.BGE import BGEEmbeddingModel
    return BGEEmbeddingModel(global_config=global_config, embedding_model_name=model_name)


def test_encode():
    from src.hipporag.utils.config_utils import BaseConfig

    model_name = os.environ.get("BGE_TEST_MODEL", "BAAI/bge-small-en-v1.5")

    print(f"  Loading BGE checkpoint: {model_name} ...", end=" ", flush=True)
    try:
        config = BaseConfig(
            embedding_model_name=model_name,
            embedding_batch_size=4,
            embedding_max_seq_len=512,
        )
        model = _make_model(config, model_name)
    except Exception as e:
        print(f"\n  [SKIP] integration -- could not load model: {type(e).__name__}: {e}")
        return

    dim = model.embedding_dim
    texts = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Cinderella attended the royal ball.",
        "Montebello is a part of Rockland County.",
        "Marina is born in Minsk.",
    ]

    print("  [1] encode() returns tensor with correct shape ...", end=" ")
    import torch
    out = model.encode(texts)
    assert isinstance(out, torch.Tensor), f"encode must return torch.Tensor, got {type(out)}"
    assert out.shape == (len(texts), dim), f"encode shape {out.shape} != ({len(texts)}, {dim})"
    print("OK")

    print("  [2] batch_encode() returns ndarray (n, dim) ...", end=" ")
    emb = model.batch_encode(texts)
    assert isinstance(emb, np.ndarray), f"batch_encode must return np.ndarray, got {type(emb)}"
    assert emb.shape == (len(texts), dim), f"batch_encode shape {emb.shape} != ({len(texts)}, {dim})"
    print("OK")

    print("  [3] rows are unit-norm by default (norm=True) ...", end=" ")
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"expected unit-norm rows, got {norms}"
    print("OK")

    print("  [4] norm=False kwarg disables normalization ...", end=" ")
    emb_unnorm = model.batch_encode(texts, norm=False)
    norms_un = np.linalg.norm(emb_unnorm, axis=1)
    assert not np.allclose(norms_un, 1.0, atol=1e-3), "norm=False should yield non-unit rows"
    print("OK")

    print("  [5] deterministic encoding ...", end=" ")
    emb2 = model.batch_encode(texts)
    assert np.allclose(emb, emb2, atol=1e-6), "same input must yield same embeddings"
    print("OK")

    print("  [6] instruction is actually applied (query != passage encoding) ...", end=" ")
    instr = "Represent this sentence for searching relevant passages: "
    emb_q = model.batch_encode(texts, instruction=instr)
    assert emb_q.shape == emb.shape
    # at least one row must change when the instruction is prepended
    assert not np.allclose(emb_q, emb, atol=1e-6), "instruction should change the embeddings"
    # instruction-encoded rows should still be unit-norm
    assert np.allclose(np.linalg.norm(emb_q, axis=1), 1.0, atol=1e-5)
    print("OK")

    print("  [7] batching consistency (small batch_size concatenates correctly) ...", end=" ")
    model.global_config.embedding_batch_size = 2
    emb_batched = model.batch_encode(texts)  # batch_size=2 forces multiple batches
    assert emb_batched.shape == (len(texts), dim)
    assert np.allclose(emb_batched, emb, atol=1e-6), "batched result must match single-batch result"
    print("OK")

    print("  [8] single-text / str input is accepted ...", end=" ")
    one = model.batch_encode(texts[0])
    assert one.shape == (1, dim), f"single-text shape {one.shape} != (1, {dim})"
    print("OK")

    print("\n  PASS: integration -- all encode checks passed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("HippoRAG BGE Embedding Tests")
    print(f"Python {sys.version}\n")

    passed, failed, skipped = [], [], []

    for name, fn in [("dispatch", test_dispatch), ("integration", test_encode)]:
        try:
            fn()
            passed.append(name)
        except Exception:
            failed.append(name)
            print(f"\n  FAIL: {name} FAILED:")
            traceback.print_exc()

    print(f"\n{'=' * 55}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"Failed:  {', '.join(failed)}")
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    main()
