from ctx_to_lora.data.processing import (
    select_repo_contexts_bm25,
    split_too_long_ctx,
)


def test_precomputed_semantic_chunks_are_not_resplit_or_rebalanced():
    chunks = [[1, 2, 3], [10, 11]]
    result = split_too_long_ctx(
        {"ctx_ids": chunks},
        model_name_or_path="unused-for-prechunked-input",
        num_chunk_probs={1: 1.0},
        max_chunk_len=4,
        min_chunk_len=-1,
        max_num_split=None,
        is_train=True,
    )
    assert result == {"ctx_ids": chunks, "n_ctx_chunks": 2}


def test_bm25_processing_selects_before_tokenization_and_keeps_metadata_aligned():
    sample = {
        "repo_contexts": [
            "database rollback transaction",
            "http retry timeout handler",
            "render interface",
        ],
        "chunk_ids": ["db", "net", "ui"],
        "chunk_paths": [["db/session.py"], ["net/client.py"], ["ui/view.py"]],
        "retrieval_query": "Where is the network retry timeout handled?",
    }
    result = select_repo_contexts_bm25(sample, top_k=2)
    assert result["chunk_ids"][0] == "net"
    assert len(result["repo_contexts"]) == len(result["chunk_paths"]) == 2
