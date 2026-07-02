# AGENTS.md

Repo-specific guidance for OpenCode sessions working on HippoRAG 2.

## Communication language

- Reply to the user **in the language they use** (Russian prompt → Russian reply, English prompt → English reply). Match their language consistently across the whole session.
- This applies only to conversational replies. **Code, code comments, commit messages, docstrings, and documentation files must always be in English**, regardless of the user's language.

## Tooling & verification

- **No lint / typecheck / formatter / CI is configured.** Do not invent commands; there is no `pytest`, `ruff`, `mypy`, `pyproject.toml`, or `.github/` workflow.
- Python 3.10 via conda is the supported runtime (see `README.md`).
- Tests are **plain scripts, not pytest cases**. Run them directly, e.g. `python tests_openai.py`. They `print` results and `assert`; there is no test runner config.

## Tests — what each one needs

Most tests make real network or GPU calls. Pick by what your environment has:

| Script | Requires |
|---|---|
| `tests_vector_stores.py` | **Nothing external** — uses a built-in `MockEmbeddingModel`. This is the only hermetic sanity check; run it first. |
| `tests_openai.py` | `OPENAI_API_KEY` env var. Cheap but bills OpenAI. |
| `tests_local.py` | A vLLM server at `http://localhost:6578/v1` **and** Azure endpoints hardcoded in the script. |
| `tests_azure.py` | Azure OpenAI endpoints/credentials. |
| `test_transformers.py` | GPU; runs a local Transformers model offline (note: singular `test_`, not `tests_`). |
| `test_bge.py` | Tier A (dispatch) is hermetic; Tier B (encode) loads a BGE checkpoint (network/GPU), gracefully skipped if unavailable. Model override: `BGE_TEST_MODEL=BAAI/bge-small-en-v1.5`. |

Optional vector-store backends are skipped automatically if their client lib is missing: `pip install qdrant-client` and/or `pip install chromadb` to enable those tests.

## Import paths (easy to get wrong)

The package uses a **`src/` layout** (`setup.py` sets `package_dir={"": "src"}`).

- After `pip install hipporag` (or `pip install -e .`): `from hipporag import HippoRAG`.
- Running from a raw clone **without** install (what `main.py`, all `tests_*.py`, `test_transformers.py`, and `test_bge.py` do): use the `src.` prefix, e.g. `from src.hipporag import HippoRAG` or `from src.hipporag.HippoRAG import HippoRAG`. Match the style already used in the file you're editing.

## Heavy / Linux-only dependencies

`requirements.txt` and `setup.py` pin exact versions of `torch==2.5.1`, `vllm==0.6.6.post1`, `transformers==4.45.2`, `gritlm==1.0.2`. **`vllm` is Linux-only and pulls CUDA**, so `pip install -e .` will fail on macOS/Windows and on GPU-less machines unless you install in two steps (core deps first, skip vllm). Assume a Linux + CUDA + conda environment.

## Entry points

- `main.py` — reproduction CLI. Loads `reproduce/dataset/{dataset}_corpus.json` and `{dataset}.json`. Note the path quirk: if `--save_dir` is the default `outputs`, it appends the dataset name (`outputs/{dataset}`); otherwise it joins with `_`.
- `main_dpr.py`, `main_azure.py` — DPR-only and Azure variants of `main.py`.
- `demo.py`, `demo_openai.py`, `demo_local.py`, `demo_azure.py`, `demo_bedrock.py` — minimal single-file examples.
- Library entrypoint: `src/hipporag/HippoRAG.py` → class `HippoRAG`. `src/hipporag/__init__.py` only re-exports `HippoRAG`.

## Configuration: one dataclass

`src/hipporag/utils/config_utils.py::BaseConfig` is the **single source of truth** — every module reads from it. `HippoRAG.__init__` accepts a `global_config` plus convenience kwargs (`save_dir`, `llm_model_name`, `llm_base_url`, `embedding_model_name`, `embedding_base_url`, `azure_endpoint`, `azure_embedding_endpoint`) that **override** fields on the passed config in place. Boolean CLI flags are parsed with `string_to_bool` (accepts `yes/no/true/false/t/f/y/n/1/0`). The `llm_mode` and `embedding_dim` fields are **informational only** (kept for GraphRAG-Benchmark compatibility): HippoRAG never reads them for dispatch/sizing — the LLM is chosen by `llm_name`/`llm_base_url` and the embedding dimension is always derived from the model itself.

## Model-name dispatch (non-obvious)

Class selection is driven by **string matching on the model name**, not explicit config:

- LLM (`src/hipporag/llm/__init__.py`): prefix `bedrock` → `BedrockLLM`; prefix `Transformers/` → `TransformersLLM`; otherwise → `CacheOpenAI` (covers OpenAI, Azure via `azure_endpoint`, and any OpenAI-compatible server via `llm_base_url`, including a local vLLM server).
- Embedding (`src/hipporag/embedding_model/__init__.py`): substring match on `GritLM`, `NV-Embed-v2`, `contriever`, `text-embedding`, `cohere`, `bge` (case-insensitive); prefix `Transformers/` or `VLLM/`. **Anything else raises `AssertionError`** — add a branch if you add a model.
- OpenIE mode (`BaseConfig.openie_mode`): `online` (default), `offline` (vLLM batch), or `Transformers-offline`.

When adding a model, update the relevant `__init__.py` getter; there is no registry.

## Two-stage offline indexing workflow

`openie_mode='offline'` is **intentionally a dead end** for a single run: `HippoRAG.pre_openie` dumps OpenIE results to disk and then `assert False`s with the message "run online indexing for future retrieval." The intended flow (documented in README §"Run with vLLM offline batch"):

1. `python main.py ... --openie_mode offline --skip_graph` → writes `openie_results_ner_*.json`, then stops.
2. Re-run the same command in `online` mode (or against a running vLLM server); it loads the cached OpenIE file via `force_openie_from_scratch=False` and builds the graph.

Do not "fix" the `assert False` in `pre_openie` — it is by design.

## Vector store backends

`get_embedding_store()` (`src/hipporag/embedding_store.py`) dispatches on `BaseConfig.vector_store_type`:

- `parquet` (default) — local Parquet file, no extra deps. `EmbeddingStore`.
- `qdrant` — `src/hipporag/vector_stores/qdrant_store.py`, lazy-imports `qdrant_client`. Local file mode when `qdrant_url=None`, else remote.
- `chroma` — `src/hipporag/vector_stores/chroma_store.py`, lazy-imports `chromadb`. Local persistent when `chroma_host=None`.

Every backend subclasses `BaseEmbeddingStore` and **must** keep `text_to_hash_id` in sync, because `HippoRAG.delete()` reads it directly.

## Outputs layout & caching

`outputs/` is gitignored. For a given run, artifacts land in:

```
{save_dir}/{llm_name}_{embedding_name}/   # '/' in model names → '_'
  ├── chunk_embeddings/vdb_chunk.parquet
  ├── entity_embeddings/vdb_entity.parquet
  ├── fact_embeddings/vdb_fact.parquet
  └── graph.pickle
{save_dir}/openie_results_ner_{llm_name}.json   # NB: one level up from the per-model dir
```

Reuse is controlled by two `BaseConfig` flags:

- `force_index_from_scratch` — ignore existing `graph.pickle` / stores and rebuild.
- `force_openie_from_scratch` — ignore existing `openie_results_ner_*.json` and re-extract.

When re-running an experiment you typically must **delete** the cached files first (see README §"Debugging Note"). `embedding_model` is set to `None` while `openie_mode='offline'`, so embedding-dependent code paths are not usable in pure offline mode.

## Datasets

`reproduce/dataset/` holds corpora (`{name}_corpus.json`) and queries (`{name}.json`) for `sample`, `musique`, `hotpotqa`, `2wikimultihopqa`. `sample.json` / `sample_corpus.json` are the small debugging set. Corpus and query JSON schemas are documented in README §"Custom Datasets" — follow the `{title, text, idx}` and `{id, question, answer, answerable, paragraphs}` shapes exactly.

## Workflow conventions (from CONTRIBUTING.md)

- File an issue before opening a PR for non-trivial changes.
- Fork-and-branch model; PRs target `main`.
- Before submitting, run whichever test scripts your change touches (see table above). There is no automated CI to catch regressions.
