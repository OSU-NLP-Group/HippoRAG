#!/usr/bin/env python3
"""Materialize one exact 64K/K<=9 snapshot-memory corpus shard."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from build_doc_to_lora_snapshot_memory_shard import (
    QA_SCHEMA,
    canonical_context_hash,
    digest_hex,
    digest_int,
    diversified,
)


GROUP_SCHEMA = pa.schema(
    [
        ("context_group_id", pa.string()),
        ("stage", pa.string()),
        ("repo_id", pa.string()),
        ("commit_sha", pa.string()),
        ("repo_dir", pa.string()),
        ("selected_chunk_ids", pa.list_(pa.string())),
        ("selected_context_hash", pa.string()),
        ("qa_file", pa.string()),
        ("qa_row_group", pa.int32()),
        ("qa_start", pa.int32()),
        ("qa_count", pa.int32()),
        ("qa_pack_starts", pa.list_(pa.int32())),
        ("qa_pack_counts", pa.list_(pa.int32())),
        ("qa_pack_token_counts", pa.list_(pa.int32())),
        ("ast_qas", pa.int32()),
        ("llm_qas", pa.int32()),
        ("context_payload_tokens", pa.int64()),
        ("answer_side_tokens", pa.int64()),
        ("estimated_cost_tokens", pa.int64()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--final-data-root", type=Path, required=True)
    parser.add_argument("--snapshot-shard", type=int, required=True)
    parser.add_argument("--snapshot-shards", type=int, default=16)
    parser.add_argument("--qa-pack-token-budget", type=int, default=16384)
    parser.add_argument("--seed", default="snapshot-memory-64k-k9-10m-v1")
    return parser.parse_args()


def load_repo_metadata(repo_dir: str) -> dict[str, Any]:
    chunks = pq.read_table(
        Path(repo_dir) / "chunks.parquet",
        columns=["chunk_id", "payload_sha256", "payload_token_count"],
        memory_map=True,
    ).to_pylist()
    payload_hashes = {
        str(row["chunk_id"]): str(row["payload_sha256"]) for row in chunks
    }
    payload_tokens = {
        str(row["chunk_id"]): int(row["payload_token_count"]) for row in chunks
    }
    snapshots: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in pq.read_table(
        Path(repo_dir) / "snapshots.parquet",
        columns=["commit_sha", "chunk_index", "chunk_id"],
        memory_map=True,
    ).to_pylist():
        snapshots[str(row["commit_sha"])].append(
            (int(row["chunk_index"]), str(row["chunk_id"]))
        )
    return {
        "payload_hashes": payload_hashes,
        "payload_tokens": payload_tokens,
        "snapshots": {
            commit: list(dict.fromkeys(chunk for _index, chunk in sorted(values)))
            for commit, values in snapshots.items()
        },
    }


def frozen_packs(
    rows: list[dict[str, Any]], token_budget: int
) -> list[list[dict[str, Any]]]:
    packs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    tokens = 0
    for row in rows:
        row_tokens = int(row["total_qa_tokens"])
        if row_tokens > token_budget:
            raise ValueError(
                f"QA {row['source_qa_id']} has {row_tokens} tokens, "
                f"above the {token_budget} pack budget"
            )
        if current and tokens + row_tokens > token_budget:
            packs.append(current)
            current = []
            tokens = 0
        current.append(row)
        tokens += row_tokens
    if current:
        packs.append(current)
    if not packs:
        raise ValueError("A selected snapshot produced no QA packs")
    return packs


def main() -> int:
    args = parse_args()
    if not 0 <= args.snapshot_shard < args.snapshot_shards:
        raise ValueError("Invalid snapshot shard")
    selected = {}
    for row in pq.read_table(args.selection, memory_map=True).to_pylist():
        key = (str(row["repo_id"]), str(row["commit_sha"]))
        shard = (
            digest_int(args.seed, "snapshot-shard", *key) % args.snapshot_shards
        )
        if shard == args.snapshot_shard:
            selected[key] = row
    if not selected:
        raise ValueError(f"Selection shard {args.snapshot_shard} is empty")

    candidates: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {
        key: {"ast": [], "llm": []} for key in selected
    }
    paths = sorted(
        glob.glob(
            str(
                args.candidate_root
                / "source_bucket_*"
                / f"shard_{args.snapshot_shard:02d}.parquet"
            )
        )
    )
    if len(paths) != 64:
        raise ValueError(f"Expected 64 candidate inputs, found {len(paths)}")
    seen_source_ids = set()
    for path in paths:
        for batch in pq.ParquetFile(path, memory_map=True).iter_batches(
            batch_size=16_384
        ):
            for row in batch.to_pylist():
                key = (str(row["repo_id"]), str(row["commit_sha"]))
                if key not in candidates:
                    raise ValueError(f"Unselected snapshot candidate {key}")
                source_id = str(row["source_qa_id"])
                if source_id in seen_source_ids:
                    raise ValueError(f"Duplicate candidate source ID {source_id}")
                seen_source_ids.add(source_id)
                candidates[key][str(row["qa_family"])].append(row)

    repo_cache: dict[str, dict[str, Any]] = {}
    qa_writers: dict[tuple[str, str], pq.ParquetWriter] = {}
    qa_row_groups: Counter[tuple[str, str]] = Counter()
    group_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    stats = Counter()
    try:
        for key, selection in sorted(
            selected.items(),
            key=lambda item: (
                str(item[1]["split"]),
                str(item[1]["stage"]),
                int(item[1]["num_chunks"]),
                str(item[1]["selection_hash"]),
            ),
        ):
            ast = diversified(candidates[key]["ast"], f"{args.seed}-ast-{key}")
            llm = diversified(candidates[key]["llm"], f"{args.seed}-llm-{key}")
            target_ast = int(selection["target_ast_qas"])
            target_llm = int(selection["target_llm_qas"])
            if len(ast) < target_ast or len(llm) < target_llm:
                raise ValueError(
                    f"{key} has AST/LLM {len(ast)}/{len(llm)}, "
                    f"needs {target_ast}/{target_llm}"
                )
            chosen = diversified(
                ast[:target_ast] + llm[:target_llm],
                f"{args.seed}-combined-{key}",
            )
            packs = frozen_packs(chosen, args.qa_pack_token_budget)

            split = str(selection["split"])
            stage = str(selection["stage"])
            k = int(selection["num_chunks"])
            repo_dir = str(selection["repo_dir"])
            metadata = repo_cache.get(repo_dir)
            if metadata is None:
                metadata = load_repo_metadata(repo_dir)
                repo_cache[repo_dir] = metadata
            chunks = metadata["snapshots"].get(key[1])
            if not chunks or len(chunks) != k or not 1 <= len(chunks) <= 9:
                raise ValueError(
                    f"Complete snapshot mismatch for {key}: selected K={k}, "
                    f"actual={len(chunks) if chunks else 0}"
                )
            context_hash = canonical_context_hash(
                key[0], key[1], chunks, chunks, metadata["payload_hashes"]
            )

            output_key = (stage, split)
            relative_qa = (
                Path(stage)
                / split
                / f"snapshot_shard_{args.snapshot_shard:02d}.qa.parquet"
            )
            writer = qa_writers.get(output_key)
            if writer is None:
                local_qa = args.output_root / relative_qa
                local_qa.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(
                    local_qa, QA_SCHEMA, compression="zstd", write_statistics=True
                )
                qa_writers[output_key] = writer

            qa_rows = []
            pack_starts = []
            pack_counts = []
            pack_tokens = []
            for pack_index, pack in enumerate(packs):
                pack_starts.append(len(qa_rows))
                pack_counts.append(len(pack))
                pack_tokens.append(sum(int(row["total_qa_tokens"]) for row in pack))
                for row in pack:
                    qa_rows.append(
                        {
                            "logical_example_id": str(row["logical_example_id"]),
                            "source_qa_id": str(row["source_qa_id"]),
                            "repo_id": key[0],
                            "commit_sha": key[1],
                            "split": split,
                            "stage": stage,
                            "coverage_mode": "snapshot_memory_full_64k",
                            "evidence_chunk_ids": [],
                            "mapping_failure": str(row["mapping_failure"] or ""),
                            "context_resolution": "64k",
                            "input_ids": [int(value) for value in row["input_ids"]],
                            "response_start": int(row["response_start"]),
                            "response_end": int(row["response_end"]),
                            "answer_token_count": int(row["answer_token_count"]),
                            "total_qa_tokens": int(row["total_qa_tokens"]),
                            "source_family": str(row["source_family"]),
                            "qa_family": str(row["qa_family"]),
                            "task_category": str(row["task_category"]),
                            "duplicate_multiplicity": int(row["duplicate_multiplicity"]),
                            "source_stage": str(row["source_stage"]),
                            "source_coverage_mode": str(row["source_coverage_mode"]),
                            "snapshot_pack_index": pack_index,
                        }
                    )
            writer.write_table(pa.Table.from_pylist(qa_rows, schema=QA_SCHEMA))
            row_group = qa_row_groups[output_key]
            qa_row_groups[output_key] += 1
            context_tokens = sum(metadata["payload_tokens"][chunk] for chunk in chunks)
            answer_tokens = sum(pack_tokens)
            group_rows[output_key].append(
                {
                    "context_group_id": digest_hex(
                        "snapshot-memory-64k-group-v1", key[0], key[1]
                    ),
                    "stage": stage,
                    "repo_id": key[0],
                    "commit_sha": key[1],
                    "repo_dir": repo_dir,
                    "selected_chunk_ids": chunks,
                    "selected_context_hash": context_hash,
                    "qa_file": str((args.final_data_root / relative_qa).resolve()),
                    "qa_row_group": row_group,
                    "qa_start": 0,
                    "qa_count": len(qa_rows),
                    "qa_pack_starts": pack_starts,
                    "qa_pack_counts": pack_counts,
                    "qa_pack_token_counts": pack_tokens,
                    "ast_qas": target_ast,
                    "llm_qas": target_llm,
                    "context_payload_tokens": context_tokens,
                    "answer_side_tokens": answer_tokens,
                    "estimated_cost_tokens": context_tokens + answer_tokens,
                }
            )
            stats["snapshots"] += 1
            stats["logical_qas"] += len(qa_rows)
            stats["ast_qas"] += target_ast
            stats["llm_qas"] += target_llm
            stats["physical_qa_packs"] += len(packs)
            stats[f"k_{k}_snapshots"] += 1
            stats[f"{stage}/{split}/snapshots"] += 1
            stats[f"{stage}/{split}/logical_qas"] += len(qa_rows)
    finally:
        for writer in qa_writers.values():
            writer.close()

    for (stage, split), rows in group_rows.items():
        path = (
            args.output_root
            / stage
            / split
            / f"snapshot_shard_{args.snapshot_shard:02d}.groups.parquet"
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(
            pa.Table.from_pylist(rows, schema=GROUP_SCHEMA),
            temporary,
            compression="zstd",
            write_statistics=True,
            row_group_size=4_096,
        )
        temporary.replace(path)
    summary = {
        "format": "doc_to_lora_snapshot_memory_64k_10m_build_shard_v1",
        "snapshot_shard": args.snapshot_shard,
        "snapshot_shards": args.snapshot_shards,
        "selection": str(args.selection.resolve()),
        "candidate_files": paths,
        "counters": dict(sorted(stats.items())),
        "output_root": str(args.output_root.resolve()),
        "final_data_root": str(args.final_data_root.resolve()),
    }
    path = args.output_root / f"snapshot_shard_{args.snapshot_shard:02d}.summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
