#!/usr/bin/env python3
"""Freeze four-round worst-case 64K Stage-1 and Stage-2 DDP fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def choose(rows: list[dict], k: int, count: int = 32) -> list[dict]:
    candidates = sorted(
        (row for row in rows if len(row["selected_chunk_ids"]) == k),
        key=lambda row: (
            int(row["context_payload_tokens"]),
            int(row["answer_side_tokens"]),
            int(row["qa_count"]),
            str(row["context_group_id"]),
        ),
        reverse=True,
    )
    selected = []
    repos = set()
    for row in candidates:
        repo = str(row["repo_id"])
        if repo in repos:
            continue
        selected.append(row)
        repos.add(repo)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} distinct-repository K={k} fixtures")
    return selected


def main() -> int:
    args = parse_args()
    ready = json.loads(args.ready.read_text())
    if ready.get("format") != "doc_to_lora_repoqa_snapshot_memory_v1":
        raise ValueError("Expected snapshot-memory READY")
    cases = {}
    for name, stage, k in (
        ("stage1_k2_worst", "stage1", 2),
        ("stage2a_k9_worst", "stage2a", 9),
    ):
        rows = []
        for path in ready["partitions"][f"{stage}/train"]["group_manifests"]:
            rows.extend(pq.read_table(path, memory_map=True).to_pylist())
        selected = choose(rows, k)
        root = args.output_root / name
        root.mkdir(parents=True, exist_ok=True)
        groups = root / "groups.parquet"
        temporary = groups.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(selected), temporary, compression="zstd")
        temporary.replace(groups)
        entry = {
            "logical_qas": sum(int(row["qa_count"]) for row in selected),
            "physical_qa_rows": sum(int(row["qa_count"]) for row in selected),
            "context_groups": len(selected),
            "physical_qa_packs": sum(
                len(row["qa_pack_counts"]) for row in selected
            ),
            "rank_assignment": "cost_bucketed_ddp_rounds_v1",
            "world_size": 8,
            "padding_group_ordinal": 0,
            "group_manifests": [str(groups.resolve())],
            "group_manifest_sha256": {str(groups.resolve()): sha256(groups)},
        }
        fixture = {
            key: value
            for key, value in ready.items()
            if key not in {"partitions", "validation_panels"}
        }
        fixture.update(
            {
                "selected_snapshots": len(selected),
                "selected_logical_qas": entry["logical_qas"],
                "partitions": {
                    f"{stage}/train": entry,
                    f"{stage}/val": entry,
                },
                "validation_panels": {
                    stage: {"fast": entry, "checkpoint": entry}
                },
                "fixture_source_ready": str(args.ready.resolve()),
                "fixture_source_ready_sha256": sha256(args.ready),
            }
        )
        ready_path = root / "READY.json"
        temporary = ready_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
        temporary.replace(ready_path)
        cases[name] = {
            "stage": stage,
            "k": k,
            "groups": len(selected),
            "steps": len(selected) // 8,
            "minimum_context_payload_tokens": min(
                int(row["context_payload_tokens"]) for row in selected
            ),
            "maximum_context_payload_tokens": max(
                int(row["context_payload_tokens"]) for row in selected
            ),
            "maximum_qas": max(int(row["qa_count"]) for row in selected),
            "maximum_answer_side_tokens": max(
                int(row["answer_side_tokens"]) for row in selected
            ),
            "ready": str(ready_path.resolve()),
            "ready_sha256": sha256(ready_path),
        }
    manifest = {
        "format": "doc_to_lora_snapshot_memory_64k_gpu_fixtures_v1",
        "source_ready": str(args.ready.resolve()),
        "source_ready_sha256": sha256(args.ready),
        "cases": cases,
    }
    path = args.output_root / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
