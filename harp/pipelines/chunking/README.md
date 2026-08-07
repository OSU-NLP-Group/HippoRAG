# Repository chunking

This directory contains the chunk-construction implementation used by HARP.
The public canonical builder requires storage locations to be supplied
explicitly rather than embedding site-specific paths.

## The 64K pipeline used by HARP

The experimental 64K store was made in two deterministic stages:

1. `build_doc_to_lora_repo_chunks.py` checks out each immutable repository
   snapshot and creates a canonical 130K source store. Files are ordered by
   repository-relative path, whole files are greedily packed, and oversized
   files are split at Python top-level AST boundaries when possible, then at
   blank-line, line, or character boundaries as fallbacks.
2. `repack_doc_to_lora_repo_chunks_multires.py --targets 64k` reconstructs the
   file parts from that frozen source store and repacks them with a 60,000-token
   soft target, a 64,000-token payload limit, and a 65,536-token final context
   cap. Chunk IDs are SHA-256 hashes of the rendered payload.

The builder writes one repository directory containing `chunks.parquet`,
`snapshots.parquet`, and `audit.json`. `chunks.parquet` stores deduplicated
payloads; `snapshots.parquet` records their ordered assignment to every
`(repo_id, commit_sha)` snapshot.

## Programs

- `build_doc_to_lora_repo_chunks.py`: build the canonical source chunks from
  pinned Git snapshots listed in a static Parquet manifest.
- `freeze_doc_to_lora_repo_chunks.sh`: validate completed source repositories
  and freeze their ordered `repositories.jsonl` index.
- `repack_doc_to_lora_repo_chunks_multires.py`: derive 8K, 32K, or 64K stores
  without walking Git again.
- `repack_doc_to_lora_repo_chunks_snapshot_shard.py`: split very large
  repositories by contiguous snapshot ranges.
- `merge_doc_to_lora_repo_snapshot_shards.py`: merge those snapshot-range
  outputs atomically.
- `freeze_doc_to_lora_multires_intersection.py`: freeze and hash the valid
  repository/snapshot intersection across resolutions.
- `audit_d2l_repo_chunks.py`: profile snapshot sizes and implied chunk/rank
  distributions before a build.
- `canonicalize_doc_to_lora_repoqa_chunk_ids.py`: remove duplicate
  content-addressed IDs from already materialized RepoQA indexes.

## Reproducing 64K chunks

Run the expensive build and repack as CPU Slurm arrays on AD-local storage.
Each array task should invoke the relevant Python entry point directly. A
single-task smoke test looks like:

```bash
python pipelines/chunking/build_doc_to_lora_repo_chunks.py \
  --input-static /path/to/static_snapshots.parquet \
  --repo-root /path/to/repository/clones \
  --output-root /path/to/chunks_130k \
  --model-name google/gemma-4-E2B-it \
  --local-files-only \
  --soft-tokens 120000 \
  --hard-tokens 130000 \
  --model-max-tokens 131072 \
  --repo-num-shards 1 \
  --repo-shard-index 0 \
  --resume
```

After freezing the completed 130K repositories, repack one frozen repository
record per task:

```bash
python pipelines/chunking/repack_doc_to_lora_repo_chunks_multires.py \
  --repositories-jsonl /path/to/chunks_130k/freezes/FREEZE/repositories.jsonl \
  --repo-index 0 \
  --output-root /path/to/chunks_multires \
  --model-name google/gemma-4-E2B-it \
  --local-files-only \
  --targets 64k
```

For production Slurm submission, request CPUs explicitly and request no GPU;
these programs are tokenizer/Git/Parquet workloads. Keep repository clones and
generated Parquets on storage local to the selected AD.

## Provenance

The public programs are identified by these hashes. The canonical builder is
the portable release variant: its three site-specific default paths were
removed and must be supplied with command-line arguments.

```text
029de49a393ba9a67e872d0dcee640b1ba7e8a05791fe9d2a2a9a4074e794bde  build_doc_to_lora_repo_chunks.py
78676bc5ccb9e232ec089eb22c2cb389d4b0aa536c1e1c67c642639cdb2d3cea  repack_doc_to_lora_repo_chunks_multires.py
0e044fcafd92b8a242d4d47990bffee39ce36c36db0281c1ac6f2ba9eae58014  repack_doc_to_lora_repo_chunks_snapshot_shard.py
b4420ce72bf48e290c75b4479d1c57315760647a37a7d6e18cbe960edf56b0bd  merge_doc_to_lora_repo_snapshot_shards.py
```
