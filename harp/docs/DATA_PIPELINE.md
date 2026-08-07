# Data pipeline

The training corpus is repository-grounded QA tied to immutable
`(repo_id, commit_sha)` snapshots. Each row stores a question, answer, source
family, task category, stable logical identifier, tokenized prompt/response
boundaries, and evidence spans where available.

## Source families

The retained generators cover four inputs:

1. deterministic repository facts derived from ASTs and static source surfaces;
2. deterministic issue, patch, test, and cross-file relations;
3. grounded LLM-generated questions from prepared repository context packs;
4. LLM rewrites of deterministic rows that preserve the original provenance.

The final training contract groups these as `ast` and `llm`. The `ast` label is
historical: it includes static, behavioral, patch, and cross-file templates in
addition to literal AST extraction.

## Frozen snapshot corpus

The production corpus is selected without access to evaluation questions:

- snapshots are serialized in deterministic repository-path order;
- canonical chunks contain at most 65,536 model tokens;
- eligible training snapshots contain one through nine chunks;
- LLM-family rows are retained first and deterministic rows fill the remaining
  budget by stable-ID sampling;
- snapshot caps, repository-disjoint validation, ordering, and hashes are
  frozen in a `READY.json` manifest;
- assistant answer positions are identified by `response_start:response_end`.

The repository text itself is materialized by the recovered canonical
chunking pipeline under `pipelines/chunking/`. It first builds a frozen 130K
source representation from exact Git commits and then deterministically
repacks it to the 64K representation consumed by training. See
[`pipelines/chunking/README.md`](../pipelines/chunking/README.md) for entry
points, formats, and commands.

The programs that create the frozen training artifact are:

```text
pipelines/data/select_doc_to_lora_snapshot_memory_64k_10m.py
pipelines/data/extract_doc_to_lora_snapshot_memory_candidates_exact.py
pipelines/data/build_doc_to_lora_snapshot_memory_64k_10m_shard.py
pipelines/data/finalize_doc_to_lora_snapshot_memory_64k_10m.py
```

Each program exposes `--help`. Large selection and materialization steps should
run as CPU Slurm jobs against AD-local storage, not on the login node.

## Code2LoRA embedding index

Code2LoRA also requires one validated 2,048-dimensional repository embedding
per `(repo_id, commit_sha)`. `repotune_issuefix.build_repo_embeddings` builds
the source embedding table. `pipelines/data/build_repoqa_baseline_data.py`
projects the frozen corpus contract into the exact embedding index consumed by
training.

The generated corpus, repository clones, token caches, and embedding tables are
external artifacts and are excluded by `.gitignore`.
