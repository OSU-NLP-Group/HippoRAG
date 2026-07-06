"""
Tests for vector store backends (Parquet, Qdrant, ChromaDB, Milvus).

No OpenAI key or GPU required -- a deterministic MockEmbeddingModel is used.

Usage
-----
# All backends (skip Qdrant/Chroma if packages not installed)
python tests_vector_stores.py

# Install optional backends first:
pip install qdrant-client   # for Qdrant tests
pip install chromadb        # for Chroma tests
pip install "pymilvus[milvus_lite]"  # for Milvus Lite tests
"""

import os
import sys
import shutil
import tempfile
import traceback
import importlib
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Mock embedding model -- returns stable random vectors, no GPU needed
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 64


class MockEmbeddingModel:
    """Returns deterministic fixed-size embeddings based on text hash."""

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        results = []
        for text in texts:
            rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
            vec = rng.random(EMBEDDING_DIM).astype(np.float32)
            vec /= np.linalg.norm(vec)
            results.append(vec)
        return np.array(results)


EMBEDDING_MODEL = MockEmbeddingModel()

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

DOCS = [
    "Oliver Badman is a politician.",
    "George Rankin is a politician.",
    "Thomas Marwick is a politician.",
    "Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    "Erik Hort's birthplace is Montebello.",
    "Marina is born in Minsk.",
    "Montebello is a part of Rockland County.",
]

EXTRA_DOCS = [
    "Tom Hort's birthplace is Montebello.",
    "Sam Hort's birthplace is Montebello.",
]

# ---------------------------------------------------------------------------
# Common test logic -- backend-agnostic
# ---------------------------------------------------------------------------

def _run_store_tests(store, label: str):
    """Run the full insert / query / delete test suite on any store backend."""
    print(f"\n  [1] Insert {len(DOCS)} documents ...", end=" ")
    store.insert_strings(DOCS)
    ids = store.get_all_ids()
    assert len(ids) == len(DOCS), f"Expected {len(DOCS)} IDs, got {len(ids)}"
    print("OK")

    print("  [2] No-op re-insert (idempotency) ...", end=" ")
    store.insert_strings(DOCS)
    assert len(store.get_all_ids()) == len(DOCS), "Re-insert must not create duplicates"
    print("OK")

    print("  [3] get_all_texts ...", end=" ")
    texts = store.get_all_texts()
    assert set(DOCS) == texts, "get_all_texts mismatch"
    print("OK")

    print("  [4] text_to_hash_id lookup ...", end=" ")
    first_doc = DOCS[0]
    hid = store.text_to_hash_id[first_doc]
    assert isinstance(hid, str) and len(hid) > 0
    print("OK")

    print("  [5] get_row / get_rows ...", end=" ")
    row = store.get_row(hid)
    assert row["content"] == first_doc
    rows = store.get_rows([hid])
    assert rows[hid]["content"] == first_doc
    print("OK")

    print("  [6] get_embeddings order & shape ...", end=" ")
    all_ids = store.get_all_ids()
    embs = store.get_embeddings(all_ids)
    assert embs.shape == (len(DOCS), EMBEDDING_DIM), (
        f"Shape mismatch: {embs.shape} vs expected ({len(DOCS)}, {EMBEDDING_DIM})"
    )
    print("OK")

    print("  [7] get_embedding single ...", end=" ")
    single = store.get_embedding(hid)
    assert single.shape == (EMBEDDING_DIM,)
    print("OK")

    print("  [8] get_missing_string_hash_ids ...", end=" ")
    new_texts = DOCS[:2] + EXTRA_DOCS
    missing = store.get_missing_string_hash_ids(new_texts)
    assert len(missing) == len(EXTRA_DOCS), (
        f"Expected {len(EXTRA_DOCS)} missing, got {len(missing)}"
    )
    print("OK")

    print(f"  [9] Incremental insert ({len(EXTRA_DOCS)} new docs) ...", end=" ")
    store.insert_strings(EXTRA_DOCS)
    assert len(store.get_all_ids()) == len(DOCS) + len(EXTRA_DOCS)
    print("OK")

    print("  [10] Delete extra docs ...", end=" ")
    ids_to_delete = [store.text_to_hash_id[t] for t in EXTRA_DOCS]
    store.delete(ids_to_delete)
    assert len(store.get_all_ids()) == len(DOCS)
    for t in EXTRA_DOCS:
        assert t not in store.text_to_hash_id
    print("OK")


def _test_persistence(make_store, label: str):
    """Verify that data survives a store teardown and reload."""
    print(f"\n  [11] Persistence (save -> reload) ...", end=" ")
    store1 = make_store()
    store1.insert_strings(DOCS)
    ids_before = set(store1.get_all_ids())

    # Close before reopening -- required for backends that hold file locks (e.g. Qdrant local)
    store1.close()

    store2 = make_store()  # fresh instance, same path -- should reload
    ids_after = set(store2.get_all_ids())
    assert ids_before == ids_after, (
        f"IDs changed after reload:\n  before={ids_before}\n  after={ids_after}"
    )
    print("OK")


# ---------------------------------------------------------------------------
# Per-backend test runners
# ---------------------------------------------------------------------------

def test_parquet(tmp_dir: str):
    from src.hipporag.embedding_store import EmbeddingStore

    label = "Parquet"
    print(f"\n{'='*55}")
    print(f"  Backend: {label}")
    print(f"{'='*55}")

    db_path = os.path.join(tmp_dir, "parquet", "chunk_embeddings")
    store = EmbeddingStore(EMBEDDING_MODEL, db_path, batch_size=16, namespace="chunk")
    _run_store_tests(store, label)

    def make_store():
        return EmbeddingStore(EMBEDDING_MODEL, db_path, batch_size=16, namespace="chunk")

    _test_persistence(make_store, label)
    print(f"\n  PASS: {label} -- all tests passed")


def test_qdrant(tmp_dir: str):
    if importlib.util.find_spec("qdrant_client") is None:
        print("\n  [SKIP] Qdrant -- qdrant-client not installed  (pip install qdrant-client)")
        return

    from src.hipporag.vector_stores.qdrant_store import QdrantEmbeddingStore

    label = "Qdrant (local)"
    print(f"\n{'='*55}")
    print(f"  Backend: {label}")
    print(f"{'='*55}")

    db_path = os.path.join(tmp_dir, "qdrant", "chunk_embeddings")

    class _FakeConfig:
        qdrant_url = None  # use local file mode

    cfg = _FakeConfig()
    store = QdrantEmbeddingStore(EMBEDDING_MODEL, db_path, batch_size=16, namespace="chunk", global_config=cfg)
    _run_store_tests(store, label)
    store.close()  # release file lock before persistence test opens the same path

    def make_store():
        return QdrantEmbeddingStore(EMBEDDING_MODEL, db_path, batch_size=16, namespace="chunk", global_config=cfg)

    _test_persistence(make_store, label)
    print(f"\n  PASS: {label} -- all tests passed")


def test_chroma(tmp_dir: str):
    if importlib.util.find_spec("chromadb") is None:
        print("\n  [SKIP] ChromaDB -- chromadb not installed  (pip install chromadb)")
        return

    from src.hipporag.vector_stores.chroma_store import ChromaEmbeddingStore

    label = "ChromaDB (local)"
    print(f"\n{'='*55}")
    print(f"  Backend: {label}")
    print(f"{'='*55}")

    db_path = os.path.join(tmp_dir, "chroma", "chunk_embeddings")

    class _FakeConfig:
        chroma_host = None  # use local persistent mode
        chroma_port = 8000

    cfg = _FakeConfig()
    store = ChromaEmbeddingStore(EMBEDDING_MODEL, db_path, batch_size=16, namespace="chunk", global_config=cfg)
    _run_store_tests(store, label)

    def make_store():
        return ChromaEmbeddingStore(EMBEDDING_MODEL, db_path, batch_size=16, namespace="chunk", global_config=cfg)

    _test_persistence(make_store, label)
    print(f"\n  PASS: {label} -- all tests passed")


def test_milvus(tmp_dir: str):
    if importlib.util.find_spec("pymilvus") is None:
        print('\n  [SKIP] Milvus -- pymilvus not installed  (pip install "pymilvus[milvus_lite]")')
        return

    from src.hipporag.vector_stores.milvus_store import MilvusEmbeddingStore

    label = "Milvus Lite (local)"
    print(f"\n{'='*55}")
    print(f"  Backend: {label}")
    print(f"{'='*55}")

    db_path = os.path.join(tmp_dir, "milvus", "chunk_embeddings")

    class _FakeConfig:
        milvus_uri = None  # use local Milvus Lite mode
        milvus_token = None
        milvus_db_name = None
        milvus_consistency_level = "Session"

    cfg = _FakeConfig()
    try:
        store = MilvusEmbeddingStore(EMBEDDING_MODEL, db_path, batch_size=16, namespace="chunk", global_config=cfg)
    except ImportError as exc:
        print(f'\n  [SKIP] Milvus -- {exc}  (pip install "pymilvus[milvus_lite]")')
        return

    _run_store_tests(store, label)
    store.close()

    def make_store():
        return MilvusEmbeddingStore(EMBEDDING_MODEL, db_path, batch_size=16, namespace="chunk", global_config=cfg)

    _test_persistence(make_store, label)
    print(f"\n  PASS: {label} -- all tests passed")


def test_factory(tmp_dir: str):
    """Verify get_embedding_store() returns the right class for each type."""
    from src.hipporag.embedding_store import get_embedding_store, EmbeddingStore, BaseEmbeddingStore

    print(f"\n{'='*55}")
    print("  Factory: get_embedding_store()")
    print(f"{'='*55}")

    class _Config:
        vector_store_type = "parquet"
        qdrant_url = None
        chroma_host = None
        chroma_port = 8000
        milvus_uri = None
        milvus_token = None
        milvus_db_name = None
        milvus_consistency_level = "Session"

    cfg = _Config()
    store = get_embedding_store(EMBEDDING_MODEL, os.path.join(tmp_dir, "factory"), 16, "chunk", cfg)
    assert isinstance(store, EmbeddingStore), f"Expected EmbeddingStore, got {type(store)}"
    assert isinstance(store, BaseEmbeddingStore), "Must inherit BaseEmbeddingStore"
    print("  [1] parquet -> EmbeddingStore  OK")

    if importlib.util.find_spec("qdrant_client"):
        from src.hipporag.vector_stores.qdrant_store import QdrantEmbeddingStore
        cfg.vector_store_type = "qdrant"
        store = get_embedding_store(EMBEDDING_MODEL, os.path.join(tmp_dir, "factory_qdrant"), 16, "chunk", cfg)
        assert isinstance(store, QdrantEmbeddingStore)
        print("  [2] qdrant  -> QdrantEmbeddingStore  OK")

    if importlib.util.find_spec("chromadb"):
        from src.hipporag.vector_stores.chroma_store import ChromaEmbeddingStore
        cfg.vector_store_type = "chroma"
        store = get_embedding_store(EMBEDDING_MODEL, os.path.join(tmp_dir, "factory_chroma"), 16, "chunk", cfg)
        assert isinstance(store, ChromaEmbeddingStore)
        print("  [3] chroma  -> ChromaEmbeddingStore  OK")

    if importlib.util.find_spec("pymilvus"):
        from src.hipporag.vector_stores.milvus_store import MilvusEmbeddingStore
        cfg.vector_store_type = "milvus"
        try:
            store = get_embedding_store(EMBEDDING_MODEL, os.path.join(tmp_dir, "factory_milvus"), 16, "chunk", cfg)
        except ImportError as exc:
            print(f'  [4] milvus  -> SKIP ({exc})')
        else:
            assert isinstance(store, MilvusEmbeddingStore)
            store.close()
            print("  [4] milvus  -> MilvusEmbeddingStore  OK")

    print(f"\n  PASS: Factory -- all checks passed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("HippoRAG Vector Store Tests")
    print(f"Python {sys.version}")

    tmp_dir = tempfile.mkdtemp(prefix="hipporag_vstore_test_")
    print(f"\nTemp directory: {tmp_dir}")

    passed, failed = [], []

    for name, fn in [
        ("Parquet", test_parquet),
        ("Qdrant", test_qdrant),
        ("ChromaDB", test_chroma),
        ("Milvus", test_milvus),
        ("Factory", test_factory),
    ]:
        try:
            fn(tmp_dir)
            passed.append(name)
        except Exception:
            failed.append(name)
            print(f"\n  FAIL: {name} FAILED:")
            traceback.print_exc()

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*55}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"Failed:  {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    main()
