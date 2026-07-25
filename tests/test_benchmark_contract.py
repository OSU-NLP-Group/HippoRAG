"""
Hermetic contract test for GraphRAG-Benchmark compatibility.

Verifies the public API surface that the external GraphRAG-Benchmark script
(Examples/run_hipporag2.py) depends on. Makes NO network, GPU, or API calls --
everything is checked structurally (signatures, dataclass fields, packaged data
files, dispatch tables).

Collected by pytest (functions are named ``test_*``). Uses the INSTALLED
``hipporag`` package, because it must reflect exactly what an external consumer
sees after `pip install hipporag` / `pip install -e .`:

    pytest tests/test_benchmark_contract.py

A failure here means a change in HippoRAG would silently break the benchmark.
"""

import inspect
import os


def _package_dir():
    import hipporag

    assert hipporag.__file__ is not None
    return os.path.dirname(hipporag.__file__)


def _dspy_prompt_path():
    return os.path.join(
        _package_dir(), "prompts", "dspy_prompts", "filter_llama3.3-70B-Instruct.json"
    )


def test_import_surface():
    # Mirrors `import hipporag` + hipporag.__file__ in run_hipporag2.py.
    import hipporag

    assert hipporag.__file__ is not None, "hipporag.__file__ is None"
    assert os.path.isdir(_package_dir()), f"package dir not found: {_package_dir()}"


def test_packaged_dspy_prompt():
    # Mirrors how run_hipporag2.py computes DSPY_FILTER_PATH.
    path = _dspy_prompt_path()
    assert os.path.isfile(path), f"missing packaged prompt: {path}"


def test_base_config_kwargs(tmp_path):
    # The exact call shape used by Examples/run_hipporag2.py::process_corpus.
    from hipporag.utils.config_utils import BaseConfig

    config = BaseConfig(
        save_dir=str(tmp_path),
        llm_base_url="https://api.openai.com/v1",
        llm_name="gpt-4o-mini",
        embedding_model_name="text-embedding-3-small",
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        rerank_dspy_file_path=_dspy_prompt_path(),
        retrieval_top_k=5,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=10,
        openie_mode="online",
    )

    # The benchmark also sets llm_mode / embedding_dim after construction.
    config.llm_mode = "openai"
    config.llm_mode = "ollama"
    config.embedding_dim = 1536

    # Spot-check the values actually consumed downstream.
    assert config.llm_name == "gpt-4o-mini"
    assert config.embedding_model_name == "text-embedding-3-small"
    assert config.graph_type == "facts_and_sim_passage_node_unidirectional"
    assert config.retrieval_top_k == 5
    assert config.qa_top_k == 5
    assert config.linking_top_k == 5
    assert config.max_qa_steps == 3
    assert config.corpus_len == 10
    assert config.llm_mode == "ollama"
    assert config.embedding_dim == 1536


def test_string_to_bool():
    from hipporag.utils.misc_utils import string_to_bool

    assert string_to_bool("yes") is True
    assert string_to_bool("true") is True
    assert string_to_bool("1") is True
    assert string_to_bool("no") is False
    assert string_to_bool("false") is False
    assert string_to_bool("0") is False


def test_embedding_dispatch_for_benchmark():
    from hipporag.embedding_model import _get_embedding_model_class

    # benchmark passes the local contriever path; dispatch matches on substring.
    assert _get_embedding_model_class("facebook/contriever").__name__ == "ContrieverModel"
    assert _get_embedding_model_class("/home/xzs/data/model/contriever").__name__ == "ContrieverModel"
    # benchmark's openai_emb=True branch uses this name.
    assert _get_embedding_model_class("text-embedding-3-small").__name__ == "OpenAIEmbeddingModel"


def test_query_solution_to_dict():
    from hipporag.utils.misc_utils import QuerySolution

    sol = QuerySolution(
        question="What is George Rankin's occupation?",
        docs=["George Rankin is a politician."],
        answer="Politician",
    )
    d = sol.to_dict()
    assert isinstance(d, dict)
    assert "docs" in d, "to_dict() must contain 'docs'"
    assert "answer" in d, "to_dict() must contain 'answer'"
    assert d["answer"] == "Politician"
    assert isinstance(d["docs"], list)


def test_rag_qa_signature():
    from hipporag.HippoRAG import HippoRAG

    params = inspect.signature(HippoRAG.rag_qa).parameters
    assert "queries" in params, "rag_qa must accept 'queries'"
    assert "gold_docs" in params, "rag_qa must accept 'gold_docs'"
    assert "gold_answers" in params, "rag_qa must accept 'gold_answers'"
    # The benchmark relies on gold_answers=None as the default (no-eval path).
    assert params["gold_answers"].default is None, "gold_answers must default to None"


def test_index_artifact_paths_are_model_independent(tmp_path):
    """Two-stage benchmark workflow: index with model A, query with model B sharing
    the same save_dir. Index artifact paths must NOT depend on llm_name /
    embedding_model_name -- only on save_dir. This is what lets an index built with
    one model be queried with another (GraphRAG-Benchmark run_hipporag2.py --phase).
    """
    import numpy as np

    import hipporag
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    dspy_path = os.path.join(
        os.path.dirname(hipporag.__file__),
        "prompts", "dspy_prompts", "filter_llama3.3-70B-Instruct.json",
    )

    class _MockLLM:
        # DSPyFilter reads `extraction_llm.infer` at init; OpenIE stores the llm.
        def infer(self, messages, **kwargs):
            return "", {}, False

    class _MockEmb:
        embedding_dim = 8

        def batch_encode(self, texts, **kwargs):
            return np.random.rand(len(texts), self.embedding_dim).astype("float32")

    def _build(llm_name, save_dir):
        return HippoRAG(
            global_config=BaseConfig(
                save_dir=save_dir,
                llm_name=llm_name,
                embedding_model_name="text-embedding-3-small",
                openie_mode="online",
                rerank_dspy_file_path=dspy_path,
            ),
            extraction_llm=_MockLLM(),
            qa_llm=_MockLLM(),
            embedding_model=_MockEmb(),
        )

    dir_a = str(tmp_path / "strong")
    dir_b = str(tmp_path / "cheap")
    rag_a = _build("strong-llm", dir_a)
    rag_b = _build("cheap-llm", dir_b)

    # Paths are flat under save_dir and contain NO model name.
    assert rag_a.openie_results_path == os.path.join(dir_a, "openie_results_ner.json")
    assert rag_b.openie_results_path == os.path.join(dir_b, "openie_results_ner.json")
    assert rag_a._graph_pickle_filename == os.path.join(dir_a, "graph.pickle")
    assert rag_b._graph_pickle_filename == os.path.join(dir_b, "graph.pickle")
    for model_name in ("strong-llm", "cheap-llm", "text-embedding-3-small"):
        assert model_name not in rag_a.openie_results_path
        assert model_name not in rag_a._graph_pickle_filename


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
