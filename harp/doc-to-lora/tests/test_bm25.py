from ctx_to_lora.retrieval.bm25 import BM25Document, BM25Index, code_tokens


def test_code_tokenizer_preserves_and_splits_identifiers_and_paths():
    tokens = code_tokens("src/httpClient.py parse_response")
    for expected in ("src/httpclient.py", "src", "http", "client", "py", "parse_response", "parse", "response"):
        assert expected in tokens


def test_bm25_prefers_matching_code_and_is_deterministic():
    documents = [
        BM25Document("b", "database transaction rollback", ("db/session.py",)),
        BM25Document("a", "HTTP retry timeout handling", ("net/http_client.py",)),
        BM25Document("c", "render user interface", ("ui/view.py",)),
    ]
    index = BM25Index(documents)
    assert index.top_k_ids("Where is http_client retry timeout handled?", 2)[0] == "a"
    assert index.top_k_ids("term-that-does-not-exist", 3) == ["a", "b", "c"]
