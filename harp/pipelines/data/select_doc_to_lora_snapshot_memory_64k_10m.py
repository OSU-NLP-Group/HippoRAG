#!/usr/bin/env python3
"""Freeze the exact 64K/K<=9 snapshot-memory selection for a 10M-QA run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def digest_hex(*parts: object) -> str:
    return hashlib.sha256(
        "\0".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--snapshot-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--train-logical-qas", type=int, default=10_000_000)
    parser.add_argument("--val-snapshots", type=int, default=400)
    parser.add_argument("--val-qas-per-family", type=int, default=16)
    parser.add_argument("--resolution", default="64k")
    parser.add_argument("--maximum-chunks", type=int, default=9)
    parser.add_argument("--stage1-maximum-chunks", type=int, default=2)
    parser.add_argument("--seed", default="snapshot-memory-64k-k9-10m-v1")
    return parser.parse_args()


def choose_validation(rows: list[dict], target: int, seed: str) -> list[dict]:
    eligible = [
        row
        for row in rows
        if min(int(row["ast_qa_count"]), int(row["llm_qa_count"])) >= 16
    ]
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        by_repo[str(row["repo_id"])].append(row)
    chosen: dict[tuple[str, str], dict] = {}
    for repo_id, candidates in sorted(by_repo.items()):
        row = min(
            candidates,
            key=lambda value: (
                int(value["num_chunks"]),
                digest_hex(seed, repo_id, value["commit_sha"]),
            ),
        )
        chosen[(str(row["repo_id"]), str(row["commit_sha"]))] = row
    remaining = sorted(
        (
            row
            for row in eligible
            if (str(row["repo_id"]), str(row["commit_sha"])) not in chosen
        ),
        key=lambda value: (
            int(value["num_chunks"]),
            digest_hex(seed, value["repo_id"], value["commit_sha"]),
        ),
    )
    for row in remaining[: target - len(chosen)]:
        chosen[(str(row["repo_id"]), str(row["commit_sha"]))] = row
    if len(chosen) != target:
        raise ValueError(f"Selected {len(chosen)} validation snapshots, need {target}")
    return list(chosen.values())


def allocate_train_ast(rows: list[dict], total_qas: int, seed: str) -> int:
    llm_total = sum(int(row["llm_qa_count"]) for row in rows)
    ast_budget = total_qas - llm_total
    if ast_budget < 0:
        raise ValueError("All eligible LLM QAs exceed the total QA budget")
    if ast_budget > sum(int(row["ast_qa_count"]) for row in rows):
        raise ValueError("The eligible AST+LLM pool is smaller than the QA budget")

    lower = max(int(row["llm_qa_count"]) for row in rows)
    upper = max(
        int(row["llm_qa_count"]) + int(row["ast_qa_count"]) for row in rows
    )
    while lower < upper:
        cap = (lower + upper) // 2
        available = sum(
            min(
                int(row["ast_qa_count"]),
                max(0, cap - int(row["llm_qa_count"])),
            )
            for row in rows
        )
        if available >= ast_budget:
            upper = cap
        else:
            lower = cap + 1
    cap = lower
    for row in rows:
        row["target_llm_qas"] = int(row["llm_qa_count"])
        row["target_ast_qas"] = min(
            int(row["ast_qa_count"]),
            max(0, cap - int(row["llm_qa_count"])),
        )
    excess = sum(int(row["target_ast_qas"]) for row in rows) - ast_budget
    removable = sorted(
        (row for row in rows if int(row["target_ast_qas"]) > 0),
        key=lambda row: digest_hex(seed, "exact-ast-trim", row["repo_id"], row["commit_sha"]),
        reverse=True,
    )
    index = 0
    while excess:
        row = removable[index % len(removable)]
        if int(row["target_ast_qas"]) > 0:
            row["target_ast_qas"] = int(row["target_ast_qas"]) - 1
            excess -= 1
        index += 1
    return cap


def main() -> int:
    args = parse_args()
    availability = pq.read_table(args.availability, memory_map=True).to_pylist()
    index = pq.read_table(args.snapshot_index, memory_map=True)
    index = index.filter(pc.equal(index["resolution"], args.resolution))
    freeze = {
        (str(row["repo_id"]), str(row["commit_sha"])): {
            "repo_dir": str(row["repo_dir"]),
            "num_chunks": int(row["num_chunks"]),
        }
        for row in index.to_pylist()
    }

    eligible: dict[str, list[dict]] = {"train": [], "val": []}
    for source in availability:
        split = str(source["split"])
        metadata = freeze.get((str(source["repo_id"]), str(source["commit_sha"])))
        if split not in eligible or metadata is None:
            continue
        if not 1 <= metadata["num_chunks"] <= args.maximum_chunks:
            continue
        row = dict(source)
        row.update(metadata)
        eligible[split].append(row)

    train = eligible["train"]
    if not train:
        raise ValueError("No eligible training snapshots")
    cap = allocate_train_ast(train, args.train_logical_qas, args.seed)
    val = choose_validation(eligible["val"], args.val_snapshots, f"{args.seed}-val")
    for row in val:
        pairs = min(
            args.val_qas_per_family,
            int(row["ast_qa_count"]),
            int(row["llm_qa_count"]),
        )
        row["target_ast_qas"] = pairs
        row["target_llm_qas"] = pairs

    output_rows = []
    for rows in (train, val):
        for row in rows:
            k = int(row["num_chunks"])
            target_ast = int(row["target_ast_qas"])
            target_llm = int(row["target_llm_qas"])
            output_rows.append(
                {
                    "repo_id": str(row["repo_id"]),
                    "commit_sha": str(row["commit_sha"]),
                    "split": str(row["split"]),
                    "stage": (
                        "stage1" if k <= args.stage1_maximum_chunks else "stage2a"
                    ),
                    "repo_dir": str(row["repo_dir"]),
                    "context_resolution": args.resolution,
                    "num_chunks": k,
                    "target_ast_qas": target_ast,
                    "target_llm_qas": target_llm,
                    "target_logical_qas": target_ast + target_llm,
                    "ast_qa_count": int(row["ast_qa_count"]),
                    "llm_qa_count": int(row["llm_qa_count"]),
                    "selection_hash": digest_hex(
                        args.seed, row["repo_id"], row["commit_sha"]
                    ),
                }
            )
    output_rows.sort(
        key=lambda row: (
            row["split"],
            row["stage"],
            row["num_chunks"],
            row["selection_hash"],
        )
    )
    schema = pa.schema(
        [
            ("repo_id", pa.string()),
            ("commit_sha", pa.string()),
            ("split", pa.string()),
            ("stage", pa.string()),
            ("repo_dir", pa.string()),
            ("context_resolution", pa.string()),
            ("num_chunks", pa.int32()),
            ("target_ast_qas", pa.int32()),
            ("target_llm_qas", pa.int32()),
            ("target_logical_qas", pa.int32()),
            ("ast_qa_count", pa.int64()),
            ("llm_qa_count", pa.int64()),
            ("selection_hash", pa.string()),
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(output_rows, schema=schema), temporary, compression="zstd")
    temporary.replace(args.output)

    def summarize(rows: list[dict]) -> dict:
        return {
            "snapshots": len(rows),
            "repositories": len({str(row["repo_id"]) for row in rows}),
            "logical_qas": sum(
                int(row["target_ast_qas"]) + int(row["target_llm_qas"])
                for row in rows
            ),
            "ast_qas": sum(int(row["target_ast_qas"]) for row in rows),
            "llm_qas": sum(int(row["target_llm_qas"]) for row in rows),
            "stage_histogram": dict(sorted(Counter(
                "stage1"
                if int(row["num_chunks"]) <= args.stage1_maximum_chunks
                else "stage2a"
                for row in rows
            ).items())),
            "k_histogram": dict(sorted(Counter(int(row["num_chunks"]) for row in rows).items())),
        }

    train_summary = summarize(train)
    if train_summary["logical_qas"] != args.train_logical_qas:
        raise ValueError("Exact train QA budget was not met")
    summary = {
        "format": "doc_to_lora_snapshot_memory_64k_10m_selection_v1",
        "seed": args.seed,
        "resolution": args.resolution,
        "maximum_chunks": args.maximum_chunks,
        "stage1_maximum_chunks": args.stage1_maximum_chunks,
        "per_snapshot_qa_cap": cap,
        "policy": {
            "train": "all_eligible_llm_then_ast_fill_to_exact_budget",
            "validation": "fixed_equal_family_panel",
        },
        "availability": str(args.availability.resolve()),
        "availability_sha256": sha256(args.availability),
        "snapshot_index": str(args.snapshot_index.resolve()),
        "snapshot_index_sha256": sha256(args.snapshot_index),
        "train": train_summary,
        "val": summarize(val),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = args.summary.with_suffix(args.summary.suffix + ".tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_summary.replace(args.summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
