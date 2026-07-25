# AGENTS.md

Repo-specific guidance for OpenCode sessions working on HippoRAG 2.

## Communication language

- Reply to the user **in the language they use** (Russian prompt → Russian reply, English prompt → English reply). Match their language consistently across the whole session.
- This applies only to conversational replies. **Code, code comments, commit messages, docstrings, and documentation files must always be in English**, regardless of the user's language.

## Tooling & verification

- **Test runner: pytest** (configured via `pytest.ini`; installed through the `dev` extra: `pip install -e .[dev]`). **No lint / typecheck / formatter / CI is configured** — do not invent commands; there is no `ruff`, `mypy`, or `.github/` workflow.
- Python 3.10 via conda is the supported runtime (see `README.md`).
- Hermetic tests live in `tests/test_*.py` and are **collected by pytest** (functions named `test_*`, plus `unittest.TestCase` classes). Provider integration scripts under `tests/integration/run_*.py` are plain scripts — pytest does **not** collect them (filename `run_*`); run them directly, e.g. `python tests/integration/run_openai.py`.
- Run tests with:
  ```sh
  pytest                                  # all hermetic tests (the default; -m "not integration")
  pytest -m integration                   # also GPU/network-marked tests (BGE encode tier)
  python tests/integration/run_openai.py  # provider integration (plain script)
  ```

## Tests — what each one needs

Hermetic tests are collected by `pytest`; provider scripts are run manually. All tests under `tests/` use the installed `hipporag` package, so `pip install -e .` is required.

| Script | pytest? | Requires |
|---|---|---|
| `tests/test_benchmark_contract.py` | yes | **Nothing external** — verifies the public API contract that GraphRAG-Benchmark's `Examples/run_hipporag2.py` depends on. Run after every upstream sync. |
| `tests/test_vector_stores.py` | yes | **Nothing external** — `MockEmbeddingModel`; Parquet always, Qdrant/Chroma/Milvus auto-skip if their client lib is missing. |
| `tests/test_bedrock_mantle.py` | yes (`unittest.TestCase`) | **Nothing external** — Bedrock Mantle dispatch + auth (mocks). |
| `tests/test_bge.py` | Tier A yes; Tier B (`test_encode`) only via `pytest -m integration` | Tier A hermetic; Tier B loads a BGE checkpoint (network/GPU), gracefully skipped if unavailable. Override: `BGE_TEST_MODEL=...`; force-skip: `BGE_TEST_SKIP_INT=1`. |
| `tests/integration/run_openai.py` | no (plain script) | `OPENAI_API_KEY`. Cheap but bills OpenAI. |
| `tests/integration/run_local.py` | no (plain script) | A vLLM server at `http://localhost:6578/v1`. |
| `tests/integration/run_azure.py` | no (plain script) | Azure OpenAI endpoints/credentials. |
| `tests/integration/run_transformers.py` | no (plain script) | GPU; runs a local Transformers model offline. |

Optional vector-store backends are skipped automatically if their client lib is missing: `pip install qdrant-client`, `pip install chromadb`, and/or `pip install "pymilvus[milvus_lite]"` to enable those tests.

## Import paths (easy to get wrong)

The package uses a **`src/` layout** (`setup.py` sets `package_dir={"": "src"}`).

- After `pip install hipporag` (or `pip install -e .`): `from hipporag import HippoRAG`. **Everything under `tests/` uses this style** and requires the installed package.
- Running from a raw clone **without** install — only `main.py` does this now: use the `src.` prefix, e.g. `from src.hipporag import HippoRAG` or `from src.hipporag.HippoRAG import HippoRAG`. Match the style already used in the file you're editing.

## Heavy / Linux-only dependencies

`requirements.txt` and `setup.py` use **lower-bounded `>=` constraints** (e.g. `torch>=2.5`, `vllm>=0.10`, `transformers>=4.45`, `openai>=2.0`) so the benchmark can pick up newer compatible releases. **`vllm` is Linux-only and pulls CUDA**, so `pip install -e .` will fail on macOS/Windows and on GPU-less machines unless you install in two steps (core deps first, skip vllm). Assume a Linux + CUDA + conda environment. The `milvus` vector-store backend is an optional extra: `pip install -e .[milvus]`. The `dev` extra adds the pytest runner: `pip install -e .[dev]`.

## Entry points

- `main.py` — reproduction CLI. Loads `reproduce/dataset/{dataset}_corpus.json` and `{dataset}.json`. Note the path quirk: if `--save_dir` is the default `outputs`, it appends the dataset name (`outputs/{dataset}`); otherwise it joins with `_`.
- `examples/demo_openai.py`, `demo_local.py`, `demo_azure.py`, `demo_bedrock.py`, `demo_bedrock_mantle.py` — minimal single-file examples (share `examples/_shared.py`).
- Library entrypoint: `src/hipporag/HippoRAG.py` → class `HippoRAG`. `src/hipporag/__init__.py` only re-exports `HippoRAG`.

## Configuration: one dataclass

`src/hipporag/utils/config_utils.py::BaseConfig` is the **single source of truth** — every module reads from it. `HippoRAG.__init__` accepts a `global_config` plus convenience kwargs (`save_dir`, `llm_model_name`, `llm_base_url`, `embedding_model_name`, `embedding_base_url`, `azure_endpoint`, `azure_embedding_endpoint`) that **override** fields on the passed config in place. Boolean CLI flags are parsed with `string_to_bool` (accepts `yes/no/true/false/t/f/y/n/1/0`). The `llm_mode` and `embedding_dim` fields are **informational only** (kept for GraphRAG-Benchmark compatibility): HippoRAG never reads them for dispatch/sizing — the LLM is chosen by `llm_name`/`llm_base_url` and the embedding dimension is always derived from the model itself.

## Model-name dispatch (non-obvious)

Class selection is driven by **string matching on the model name**, not explicit config:

- LLM (`src/hipporag/llm/__init__.py`): prefix `bedrock-mantle/` → `BedrockMantleLLM` (Responses API, needs `AWS_BEARER_TOKEN_BEDROCK`); prefix `bedrock/` → `BedrockLLM`; prefix `Transformers/` → `TransformersLLM`; otherwise → `CacheOpenAI` (covers OpenAI, Azure via `azure_endpoint`, and any OpenAI-compatible server via `llm_base_url`, including a local vLLM server).
- Embedding (`src/hipporag/embedding_model/__init__.py`): substring match on `GritLM`, `NV-Embed-v2`, `contriever`, `text-embedding`, `cohere`, `bge` (case-insensitive); prefix `Transformers/` or `VLLM/`. **Anything else raises `ValueError`** — add a branch if you add a model.
- OpenIE mode (`BaseConfig.openie_mode`): `online` (default), `offline` (vLLM batch), or `Transformers-offline`.

When adding a model, update the relevant `__init__.py` getter; there is no registry.

## Two LLMs: indexing vs inference

`HippoRAG.__init__` accepts `extraction_llm` (used by `OpenIE` for offline indexing) and `qa_llm` (used by `DSPyFilter` for retrieval-augmented QA), in addition to the unified `llm` parameter. When `extraction_llm`/`qa_llm` are `None`, they default to `self.llm` (same model for both stages).

**Two-stage workflow** (for using different models across indexing vs query):
```sh
python -m hipporag.HippoRAG --phase index --model_name <indexing_llm>
python -m hipporag.HippoRAG --phase query --model_name <query_llm>
```

**In a single run** (same model, Python API):
```python
hippo = HippoRAG(llm_model_name="gpt-4o-mini", ...)  # llm used for both stages
```

**Two models in one run** (Python API, offline index):
```python
hippo = HippoRAG(
    extraction_llm=TransformersLLM("meta-llama/Llama-3.3-70B-Instruct"),
    qa_llm=CacheOpenAI("gpt-4o-mini"),
    ...
)
```

### `information_extraction_model_name` — NOT a model name

`information_extraction_model_name` is a **class selector** (`Literal["openie_openai_gpt"]`), not a model name. It selects the OpenIE implementation at index 291 of `HippoRAG.__init__`:

```python
openie_model_name = self.global_config.information_extraction_model_name.split('_')[-1]
openie_class = openie_map.get(openie_model_name, None)
```

`TransformersOffline` (`src/hipporag/information_extraction/openie_transformers_offline.py:43`) reads `global_config.llm_name`, not `information_extraction_model_name`, to determine which model to load. Setting `information_extraction_model_name="transformers_offline"` will raise `NotImplementedError` (see `openie_map.get('offline', None)` → no entry in `openie_map`).

## Two-stage offline indexing workflow

`openie_mode='offline'` is **intentionally a dead end** for a single run: `HippoRAG.pre_openie` dumps OpenIE results to disk and then `assert False`s with the message "run online indexing for future retrieval." The intended flow (documented in README §"Run with vLLM offline batch"):

1. `python main.py ... --openie_mode offline --skip_graph` → writes `openie_results_ner.json`, then stops.
2. Re-run the same command in `online` mode (or against a running vLLM server); it loads the cached OpenIE file via `force_openie_from_scratch=False` and builds the graph.

Do not "fix" the `assert False` in `pre_openie` — it is by design.

## Vector store backends

`get_embedding_store()` (`src/hipporag/embedding_store.py`) dispatches on `BaseConfig.vector_store_type`:

- `parquet` (default) — local Parquet file, no extra deps. `EmbeddingStore`.
- `qdrant` — `src/hipporag/vector_stores/qdrant_store.py`, lazy-imports `qdrant_client`. Local file mode when `qdrant_url=None`, else remote.
- `chroma` — `src/hipporag/vector_stores/chroma_store.py`, lazy-imports `chromadb`. Local persistent when `chroma_host=None`.
- `milvus` — `src/hipporag/vector_stores/milvus_store.py`, lazy-imports `pymilvus`. Requires the `milvus` extra: `pip install -e .[milvus]`.

Every backend subclasses `BaseEmbeddingStore` and **must** keep `text_to_hash_id` in sync, because `HippoRAG.delete()` reads it directly.

## Outputs layout & caching

`outputs/` is gitignored. The working directory **equals** `save_dir` (flat layout): the caller is responsible for picking a distinct `save_dir` per experiment, matching the convention used by the other frameworks in GraphRAG-Benchmark. The index location no longer depends on the LLM/embedding model names, so an index built with one model can be queried with another. Artifacts land in:

```
{save_dir}/
  ├── chunk_embeddings/vdb_chunk.parquet
  ├── entity_embeddings/vdb_entity.parquet
  ├── fact_embeddings/vdb_fact.parquet
  ├── graph.pickle
  ├── openie_results_ner.json
  └── llm_cache/{llm_name}_cache.sqlite   # LLM response cache, still keyed by model name
```

Reuse is controlled by two `BaseConfig` flags:

- `force_index_from_scratch` — ignore existing `graph.pickle` / stores and rebuild.
- `force_openie_from_scratch` — ignore existing `openie_results_ner.json` and re-extract.

When re-running an experiment you typically must **delete** the cached files first (see README §"Debugging Note"). `embedding_model` is set to `None` while `openie_mode='offline'`, so embedding-dependent code paths are not usable in pure offline mode.

## Datasets

`reproduce/dataset/` holds corpora (`{name}_corpus.json`) and queries (`{name}.json`) for `sample`, `musique`, `hotpotqa`, `2wikimultihopqa`. `sample.json` / `sample_corpus.json` are the small debugging set. Corpus and query JSON schemas are documented in README §"Custom Datasets" — follow the `{title, text, idx}` and `{id, question, answer, answerable, paragraphs}` shapes exactly.

## Workflow conventions (from CONTRIBUTING.md)

- File an issue before opening a PR for non-trivial changes.
- Fork-and-branch model; PRs target `main`.
- Before submitting, run whichever test scripts your change touches (see table above). There is no automated CI to catch regressions.
