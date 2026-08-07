#!/usr/bin/env python3
"""Extract every available QA for the exact selected repository snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from extract_doc_to_lora_snapshot_memory_candidates import (
    CANDIDATE_SCHEMA,
    FAMILY_KIND,
    PARTITIONS,
    SOURCE_COLUMNS,
)


def digest_int(*parts: object) -> int:
    value = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--bucket", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot-shards", type=int, default=16)
    parser.add_argument("--seed", default="snapshot-memory-64k-k9-10m-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.bucket < 64:
        raise ValueError("bucket must be in [0, 63]")
    selected = {}
    for row in pq.read_table(args.selection, memory_map=True).to_pylist():
        key = (str(row["repo_id"]), str(row["commit_sha"]))
        selected[key] = {
            "split": str(row["split"]),
            "shard": digest_int(args.seed, "snapshot-shard", *key)
            % args.snapshot_shards,
        }

    output_rows: list[list[dict]] = [[] for _ in range(args.snapshot_shards)]
    scanned = matched = 0
    counts: Counter[str] = Counter()
    inputs = []
    for partition in PARTITIONS:
        path = args.data_root / partition / f"bucket_{args.bucket:03d}.qa.parquet"
        if not path.is_file():
            continue
        inputs.append(str(path.resolve()))
        for batch in pq.ParquetFile(path, memory_map=True).iter_batches(
            columns=SOURCE_COLUMNS, batch_size=16_384
        ):
            scanned += batch.num_rows
            for row in batch.to_pylist():
                key = (str(row["repo_id"]), str(row["commit_sha"]))
                metadata = selected.get(key)
                if metadata is None or str(row["split"]) != metadata["split"]:
                    continue
                source_family = str(row["source_family"])
                qa_family = FAMILY_KIND.get(source_family)
                if qa_family is None:
                    raise ValueError(f"Unclassified source family {source_family!r}")
                priority = digest_int(
                    args.seed, "candidate", qa_family, row["source_qa_id"]
                )
                candidate = {
                    "logical_example_id": str(row["logical_example_id"]),
                    "source_qa_id": str(row["source_qa_id"]),
                    "repo_id": key[0],
                    "commit_sha": key[1],
                    "split": str(row["split"]),
                    "source_stage": str(row["stage"]),
                    "source_coverage_mode": str(row["coverage_mode"]),
                    "evidence_chunk_ids": [
                        str(value) for value in row["evidence_chunk_ids"] or []
                    ],
                    "mapping_failure": str(row["mapping_failure"] or ""),
                    "source_context_resolution": str(row["context_resolution"]),
                    "input_ids": [int(value) for value in row["input_ids"]],
                    "response_start": int(row["response_start"]),
                    "response_end": int(row["response_end"]),
                    "answer_token_count": int(row["answer_token_count"]),
                    "total_qa_tokens": int(row["total_qa_tokens"]),
                    "source_family": source_family,
                    "qa_family": qa_family,
                    "task_category": str(row["task_category"]),
                    "duplicate_multiplicity": int(row["duplicate_multiplicity"]),
                    "candidate_priority": priority,
                    "snapshot_shard": int(metadata["shard"]),
                }
                output_rows[int(metadata["shard"])].append(candidate)
                matched += 1
                counts[qa_family] += 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for shard, rows in enumerate(output_rows):
        path = args.output_root / f"shard_{shard:02d}.parquet"
        temporary = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(
            pa.Table.from_pylist(rows, schema=CANDIDATE_SCHEMA),
            temporary,
            compression="zstd",
            write_statistics=True,
            row_group_size=16_384,
        )
        temporary.replace(path)
        outputs.append(str(path.resolve()))
    summary = {
        "format": "doc_to_lora_snapshot_memory_candidates_exact_v1",
        "bucket": args.bucket,
        "selection": str(args.selection.resolve()),
        "snapshot_shards": args.snapshot_shards,
        "source_rows_scanned": scanned,
        "candidate_rows": matched,
        "candidate_family_counts": dict(sorted(counts.items())),
        "input_files": inputs,
        "output_files": outputs,
    }
    success = args.output_root / "_SUCCESS.json"
    temporary = success.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(success)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
