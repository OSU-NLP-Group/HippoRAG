#!/usr/bin/env python3
"""Atomically merge snapshot-range repacking outputs for one repository."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_doc_to_lora_repo_chunks as base  # noqa: E402
import repack_doc_to_lora_repo_chunks_multires as repack  # noqa: E402
from repack_doc_to_lora_repo_chunks_snapshot_shard import atomic_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repositories-jsonl", required=True, type=Path)
    parser.add_argument("--repo-index", required=True, type=int)
    parser.add_argument("--snapshot-num-shards", required=True, type=int)
    parser.add_argument("--partial-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--target", required=True, choices=sorted(repack.TARGETS))
    args = parser.parse_args()

    record = repack.load_repository_record(args.repositories_jsonl, args.repo_index)
    source_repo_dir = Path(record["chunks_parquet"]).parent.name
    source_root = args.partial_root / source_repo_dir
    destination = args.output_root / args.target / "repositories" / source_repo_dir
    if repack.output_complete(destination):
        print(json.dumps({"status": "already_complete", "repo_id": record["repo_id"]}))
        return

    audits: list[dict[str, Any]] = []
    for shard_index in range(args.snapshot_num_shards):
        shard = source_root / f"shard_{shard_index:03d}_of_{args.snapshot_num_shards:03d}"
        audit_path = shard / "audit.json"
        if not repack.output_complete(shard):
            raise RuntimeError(f"Incomplete snapshot shard: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit["repo_id"] != record["repo_id"]:
            raise ValueError(f"Repository mismatch in {audit_path}")
        if audit.get("target") != args.target:
            raise ValueError(
                f"Target mismatch in {audit_path}: {audit.get('target')!r} != {args.target!r}"
            )
        audits.append(audit)

    destination.mkdir(parents=True, exist_ok=True)
    chunks_final = destination / "chunks.parquet"
    snapshots_final = destination / "snapshots.parquet"
    chunks_tmp = destination / f"chunks.parquet.tmp.merge.{os.getpid()}"
    snapshots_tmp = destination / f"snapshots.parquet.tmp.merge.{os.getpid()}"
    chunks_tmp.unlink(missing_ok=True)
    snapshots_tmp.unlink(missing_ok=True)

    chunk_writer = base.BufferedParquetWriter(chunks_tmp, base.CHUNK_SCHEMA)
    seen: set[str] = set()
    unique_chunks = 0
    duplicate_chunks = 0
    try:
        for audit in audits:
            parquet = pq.ParquetFile(audit["chunks_parquet"], memory_map=True)
            for row_group in range(parquet.num_row_groups):
                table = parquet.read_row_group(row_group)
                for row in table.to_pylist():
                    chunk_id = str(row["chunk_id"])
                    if chunk_id in seen:
                        duplicate_chunks += 1
                        continue
                    seen.add(chunk_id)
                    chunk_writer.write(row)
                    unique_chunks += 1
    except Exception:
        chunk_writer.close()
        chunks_tmp.unlink(missing_ok=True)
        raise
    else:
        chunk_writer.close()

    snapshot_writer = pq.ParquetWriter(snapshots_tmp, base.SNAPSHOT_SCHEMA, compression="zstd")
    snapshot_rows = 0
    try:
        # Shards own contiguous snapshot ranges, so shard order preserves the
        # canonical snapshot_id/chunk_index ordering from the source manifest.
        for audit in audits:
            parquet = pq.ParquetFile(audit["snapshots_parquet"], memory_map=True)
            for row_group in range(parquet.num_row_groups):
                table = parquet.read_row_group(row_group)
                snapshot_writer.write_table(table)
                snapshot_rows += table.num_rows
    except Exception:
        snapshot_writer.close()
        chunks_tmp.unlink(missing_ok=True)
        snapshots_tmp.unlink(missing_ok=True)
        raise
    else:
        snapshot_writer.close()

    chunks_tmp.replace(chunks_final)
    snapshots_tmp.replace(snapshots_final)
    counters: dict[str, int] = {}
    for audit in audits:
        for key, value in audit["counters"].items():
            counters[key] = counters.get(key, 0) + int(value)
    counters["unique_chunks"] = unique_chunks
    counters["cross_shard_duplicate_chunks"] = duplicate_chunks
    counters["snapshot_chunk_references"] = snapshot_rows
    result = {
        "status": "complete",
        "repo_id": record["repo_id"],
        "target": args.target,
        "source_chunks_parquet": record["chunks_parquet"],
        "source_snapshots_parquet": record["snapshots_parquet"],
        "chunks_parquet": str(chunks_final),
        "snapshots_parquet": str(snapshots_final),
        "snapshot_num_shards": args.snapshot_num_shards,
        "config": repack.TARGETS[args.target],
        "counters": counters,
    }
    atomic_json(destination / "audit.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
