"""
Hermetic contract test for GraphRAG-Benchmark compatibility.

This test verifies the public API surface that the external GraphRAG-Benchmark
script (Examples/run_hipporag2.py) depends on. It makes NO network, GPU, or API
calls -- everything is checked structurally (signatures, dataclass fields,
packaged data files, dispatch tables).

It uses the INSTALLED `hipporag` package (like tests/test_bedrock_mantle.py),
because it must reflect exactly what an external consumer sees after
`pip install hipporag` / `pip install -e .`. Run it as:

    python tests/test_benchmark_contract.py

A failure here means a change in HippoRAG would silently break the benchmark.
"""

import os
import sys
import inspect
import traceback


def check_import_surface():
    print("  [1] `import hipporag` resolves and exposes __file__ ...", end=" ")
    import hipporag
    assert hipporag.__file__ is not None, "hipporag.__file__ is None"
    package_dir = os.path.dirname(hipporag.__file__)
    assert os.path.isdir(package_dir), f"package dir not found: {package_dir}"
    print("OK")
    return package_dir


def check_packaged_dspy_prompt(package_dir):
    print("  [2] packaged dspy_prompts JSON exists at hipporag.__file__/... ...", end=" ")
    # Mirrors how run_hipporag2.py computes DSPY_FILTER_PATH.
    dspy_path = os.path.join(
        package_dir, "prompts", "dspy_prompts", "filter_llama3.3-70B-Instruct.json"
    )
    assert os.path.isfile(dspy_path), f"missing packaged prompt: {dspy_path}"
    print("OK")
    return dspy_path


def check_base_config_kwargs(dspy_path, tmp_dir):
    print("  [3] BaseConfig accepts every kwarg used by run_hipporag2.py ...", end=" ")
    from hipporag.utils.config_utils import BaseConfig

    # The exact call shape used by Examples/run_hipporag2.py::process_corpus.
    config = BaseConfig(
        save_dir=tmp_dir,
        llm_base_url="https://api.openai.com/v1",
        llm_name="gpt-4o-mini",
        embedding_model_name="text-embedding-3-small",
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        rerank_dspy_file_path=dspy_path,
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
    print("OK")


def check_string_to_bool():
    print("  [4] misc_utils.string_to_bool is importable and sane ...", end=" ")
    from hipporag.utils.misc_utils import string_to_bool
    assert string_to_bool("yes") is True
    assert string_to_bool("true") is True
    assert string_to_bool("1") is True
    assert string_to_bool("no") is False
    assert string_to_bool("false") is False
    assert string_to_bool("0") is False
    print("OK")


def check_embedding_dispatch_for_benchmark():
    print("  [5] embedding dispatch resolves benchmark model names ...", end=" ")
    from hipporag.embedding_model import _get_embedding_model_class
    # benchmark passes the local contriever path; dispatch matches on substring.
    assert _get_embedding_model_class("facebook/contriever").__name__ == "ContrieverModel"
    assert _get_embedding_model_class("/home/xzs/data/model/contriever").__name__ == "ContrieverModel"
    # benchmark's openai_emb=True branch uses this name.
    assert _get_embedding_model_class("text-embedding-3-small").__name__ == "OpenAIEmbeddingModel"
    print("OK")


def check_query_solution_to_dict():
    print("  [6] QuerySolution.to_dict() exposes 'docs' and 'answer' ...", end=" ")
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
    print("OK")


def check_rag_qa_signature():
    print("  [7] HippoRAG.rag_qa signature matches the benchmark call ...", end=" ")
    from hipporag.HippoRAG import HippoRAG
    params = inspect.signature(HippoRAG.rag_qa).parameters
    assert "queries" in params, "rag_qa must accept 'queries'"
    assert "gold_docs" in params, "rag_qa must accept 'gold_docs'"
    assert "gold_answers" in params, "rag_qa must accept 'gold_answers'"
    # The benchmark relies on gold_answers=None as the default (no-eval path).
    assert params["gold_answers"].default is None, (
        "gold_answers must default to None"
    )
    print("OK")


def main():
    import tempfile

    print("HippoRAG GraphRAG-Benchmark contract test")
    print(f"Python {sys.version}\n")

    checks = [
        ("import surface", check_import_surface),
        ("string_to_bool", check_string_to_bool),
        ("embedding dispatch", check_embedding_dispatch_for_benchmark),
        ("QuerySolution.to_dict", check_query_solution_to_dict),
        ("rag_qa signature", check_rag_qa_signature),
    ]

    passed, failed = [], []

    # [1]
    try:
        package_dir = check_import_surface()
        passed.append("import surface")
    except Exception:
        failed.append("import surface")
        print("FAILED")
        traceback.print_exc()
        # Without the package nothing else can run.
        _summarize(passed, failed)
        sys.exit(1)

    # [2]
    try:
        dspy_path = check_packaged_dspy_prompt(package_dir)
        passed.append("packaged dspy prompt")
    except Exception:
        failed.append("packaged dspy prompt")
        print("FAILED")
        traceback.print_exc()
        dspy_path = None

    # [3]
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            if dspy_path is None:
                raise AssertionError("cannot test BaseConfig: dspy prompt missing")
            check_base_config_kwargs(dspy_path, tmp_dir)
        passed.append("BaseConfig kwargs")
    except Exception:
        failed.append("BaseConfig kwargs")
        print("FAILED")
        traceback.print_exc()

    # [4]-[7]
    for name, fn in checks[1:]:
        try:
            fn()
            passed.append(name)
        except Exception:
            failed.append(name)
            print("FAILED")
            traceback.print_exc()

    _summarize(passed, failed)
    sys.exit(1 if failed else 0)


def _summarize(passed, failed):
    print(f"\n{'=' * 55}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"Failed:  {', '.join(failed)}")


if __name__ == "__main__":
    main()
