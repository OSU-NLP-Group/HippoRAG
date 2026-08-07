#!/usr/bin/env python3
"""Freeze a shared SFT subset and Code2LoRA embedding index."""

from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from repotune_issuefix.repoqa_baselines import (
    QA_COLUMNS,
    group_rows,
    length_bucket,
    load_ready,
    read_group_qas,
    sha256_file,
    stable_u64,
)


def quotas_from_counts(counts: Counter, total: int) -> dict[tuple[str, str, str], int]:
    population = sum(counts.values())
    raw = {key: total * value / population for key, value in counts.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


META_COLUMNS = (
    "logical_example_id", "repo_id", "commit_sha", "stage", "qa_family",
    "task_category",
)


def iter_train_rows(ready, columns=QA_COLUMNS):
    for stage in ("stage1", "stage2a"):
        for group in group_rows(ready, stage, "train"):
            yield from read_group_qas(group, columns=columns)


def build_subset(
    ready,
    output: Path,
    target: int,
    seed: int,
    prefix_train: Path | None = None,
) -> dict:
    strata = Counter()
    repo_min: dict[str, tuple[int, str, tuple[str, str, str]]] = {}
    seen = 0
    for row in iter_train_rows(ready, META_COLUMNS):
        key = (str(row["stage"]), str(row["qa_family"]), str(row["task_category"]))
        strata[key] += 1
        row_id = str(row["logical_example_id"])
        repo = str(row["repo_id"])
        priority = stable_u64(seed, row_id)
        candidate = (priority, row_id, key)
        if repo not in repo_min or candidate < repo_min[repo]:
            repo_min[repo] = candidate
        seen += 1
    if seen != 10_000_000:
        raise ValueError(f"Expected 10M train rows, found {seen}")
    quotas = quotas_from_counts(strata, target)
    guaranteed = {value[1] for value in repo_min.values()}
    guaranteed_by_stratum = Counter(value[2] for value in repo_min.values())
    remaining = {key: quotas[key] - guaranteed_by_stratum[key] for key in quotas}
    if min(remaining.values()) < 0:
        raise ValueError("A proportional stratum quota cannot cover repository guarantees")

    heaps: dict[tuple[str, str, str], list[tuple[int, str]]] = defaultdict(list)
    for row in iter_train_rows(ready, META_COLUMNS):
        row_id = str(row["logical_example_id"])
        if row_id in guaranteed:
            continue
        key = (str(row["stage"]), str(row["qa_family"]), str(row["task_category"]))
        limit = remaining[key]
        if limit <= 0:
            continue
        priority = stable_u64(seed, row_id)
        item = (-priority, row_id)
        heap = heaps[key]
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    selected = set(guaranteed)
    for heap in heaps.values():
        selected.update(row_id for _priority, row_id in heap)
    if len(selected) != target:
        raise ValueError(f"Expected {target} selected IDs, found {len(selected)}")

    prefix_rows = []
    prefix_ids: set[str] = set()
    if prefix_train is not None:
        prefix_table = pq.read_table(prefix_train, memory_map=True)
        prefix_rows = prefix_table.to_pylist()
        prefix_ids = {str(row["logical_example_id"]) for row in prefix_rows}
        if len(prefix_ids) != len(prefix_rows):
            raise ValueError("Prefix SFT data contains duplicate logical_example_id values")
        if not prefix_ids.issubset(selected):
            raise ValueError(
                f"The frozen prefix is not nested in the {target}-row selection: "
                f"{len(prefix_ids - selected)} IDs are missing"
            )

    continuation = []
    selected_repos = set()
    selected_strata = Counter()
    for value in prefix_rows:
        selected_repos.add(str(value["repo_id"]))
        selected_strata[(
            str(value["stage"]),
            str(value["qa_family"]),
            str(value["task_category"]),
        )] += 1
    for row in iter_train_rows(ready):
        row_id = str(row["logical_example_id"])
        if row_id not in selected or row_id in prefix_ids:
            continue
        value = dict(row)
        value["selection_priority"] = stable_u64(seed, row_id)
        value["length_bucket"] = length_bucket(len(value["input_ids"]))
        continuation.append(value)
        selected_repos.add(str(value["repo_id"]))
        selected_strata[(str(value["stage"]), str(value["qa_family"]), str(value["task_category"]))] += 1
    if len(prefix_rows) + len(continuation) != target or len(selected_repos) != int(ready["train_repositories"]):
        raise ValueError("Selected SFT materialization failed count/repository coverage")
    continuation.sort(key=lambda row: (row["length_bucket"], row["selection_priority"], row["logical_example_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    if prefix_train is not None:
        # The 1M prefix contains only low hash priorities and was inferred as
        # int64. A 5M selection reaches the uint64 midpoint, so some valid
        # stable_u64 values exceed INT64_MAX. Widen that metadata field while
        # preserving every prefix value and row position.
        priority_index = prefix_table.schema.get_field_index("selection_priority")
        if priority_index < 0:
            raise ValueError("Prefix SFT schema lacks selection_priority")
        output_schema = prefix_table.schema.set(
            priority_index,
            pa.field("selection_priority", pa.uint64()),
        )
        prefix_table = prefix_table.cast(output_schema)
        temporary = output.with_suffix(output.suffix + ".tmp")
        writer = pq.ParquetWriter(temporary, output_schema, compression="zstd")
        try:
            writer.write_table(prefix_table, row_group_size=2048)
            for start in range(0, len(continuation), 8192):
                table = pa.Table.from_pylist(
                    continuation[start : start + 8192],
                    schema=output_schema,
                )
                writer.write_table(table, row_group_size=2048)
        finally:
            writer.close()
        temporary.replace(output)
    else:
        pq.write_table(
            pa.Table.from_pylist(continuation),
            output,
            compression="zstd",
            row_group_size=2048,
        )
    return {
        "rows": len(prefix_rows) + len(continuation),
        "path": output.name,
        "repositories": len(selected_repos),
        "strata": len(selected_strata),
        "sha256": sha256_file(output),
        "prefix": (
            {
                "path": str(prefix_train.resolve()),
                "rows": len(prefix_rows),
                "sha256": sha256_file(prefix_train),
                "ordering": "exact row-value and row-order prefix",
            }
            if prefix_train is not None
            else None
        ),
    }


def build_validation(ready, output: Path) -> dict:
    rows = []
    for stage in ("stage1", "stage2a"):
        for group in group_rows(ready, stage, "val"):
            rows.extend(read_group_qas(group))
    rows.sort(key=lambda row: (length_bucket(len(row["input_ids"])), stable_u64(991, row["logical_example_id"])))
    if len(rows) != 12_800:
        raise ValueError(f"Expected 12,800 validation QAs, found {len(rows)}")
    pq.write_table(pa.Table.from_pylist(rows), output, compression="zstd", row_group_size=2048)
    return {"rows": len(rows), "sha256": sha256_file(output)}


def build_embedding_index(ready, source: Path, output: Path) -> dict:
    train_needed = {
        (str(group["repo_id"]), str(group["commit_sha"]))
        for stage in ("stage1", "stage2a")
        for group in group_rows(ready, stage, "train")
    }
    val_needed = {
        (str(group["repo_id"]), str(group["commit_sha"]))
        for stage in ("stage1", "stage2a")
        for group in group_rows(ready, stage, "val")
    }
    needed = train_needed | val_needed
    table = pq.read_table(source, columns=["repo_id", "commit_sha", "repo_state_embedding"], memory_map=True)
    found = {}
    duplicates = 0
    identical_duplicates = 0
    conflicting_duplicates = 0
    conflict_l2_sum = 0.0
    conflict_l2_max = 0.0
    conflict_examples = []
    for row in table.to_pylist():
        key = (str(row["repo_id"]), str(row["commit_sha"]))
        if key not in needed:
            continue
        embedding = np.asarray(row["repo_state_embedding"], dtype=np.float32)
        if embedding.shape != (2048,) or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid repository embedding for {key}: {embedding.shape}")
        if key in found:
            duplicates += 1
            prior = found[key]
            if np.array_equal(prior, embedding):
                identical_duplicates += 1
            else:
                # Historical Code2LoRA data preparation canonically retained
                # the first physical source row for a repeated snapshot key.
                # Preserve that behavior, but quantify every conflict rather
                # than allowing the resolution to be silent.
                conflicting_duplicates += 1
                l2 = float(np.linalg.norm(prior - embedding))
                conflict_l2_sum += l2
                conflict_l2_max = max(conflict_l2_max, l2)
                if len(conflict_examples) < 20:
                    conflict_examples.append({
                        "repo_id": key[0],
                        "commit_sha": key[1],
                        "l2_distance_from_first": l2,
                    })
        else:
            found[key] = embedding
    missing = needed - set(found)
    if missing:
        raise ValueError(f"Missing {len(missing)} Code2LoRA embeddings")
    rows = [
        {
            "repo_id": key[0],
            "commit_sha": key[1],
            "repo_state_embedding": found[key].tolist(),
        }
        for key in sorted(found)
    ]
    pq.write_table(pa.Table.from_pylist(rows), output, compression="zstd", row_group_size=1024)
    return {
        "rows": len(rows),
        "required_train_snapshots": len(train_needed),
        "required_validation_snapshots": len(val_needed),
        "source_sha256": sha256_file(source),
        "duplicate_resolution": "first_physical_source_row_per_snapshot_key",
        "duplicates_total": duplicates,
        "duplicates_identical": identical_duplicates,
        "duplicates_conflicting": conflicting_duplicates,
        "conflict_mean_l2_distance": (
            conflict_l2_sum / conflicting_duplicates if conflicting_duplicates else 0.0
        ),
        "conflict_max_l2_distance": conflict_l2_max,
        "conflict_examples": conflict_examples,
        "sha256": sha256_file(output),
    }


def reuse_subset(path: Path, expected_rows: int, expected_repositories: int | None = None) -> dict:
    parquet = pq.ParquetFile(path, memory_map=True)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"Cannot reuse {path}: expected {expected_rows} rows, "
            f"found {parquet.metadata.num_rows}"
        )
    summary = {
        "rows": expected_rows,
        "path": path.name,
        "sha256": sha256_file(path),
        "reused": True,
    }
    if expected_repositories is not None:
        repositories = set(
            str(value.as_py())
            for batch in parquet.iter_batches(columns=["repo_id"])
            for value in batch.column(0)
        )
        if len(repositories) != expected_repositories:
            raise ValueError(
                f"Cannot reuse {path}: expected {expected_repositories} repositories, "
                f"found {len(repositories)}"
            )
        summary["repositories"] = len(repositories)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready", required=True)
    parser.add_argument("--embedding-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--prefix-train",
        type=Path,
        default=None,
        help="Frozen smaller SFT parquet that must remain the exact row prefix",
    )
    args = parser.parse_args()
    ready = load_ready(args.ready)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / f"sft_train_{args.target}.parquet"
    val_path = output / "sft_val_12800.parquet"
    embedding_path = output / "code2lora_snapshot_embeddings.parquet"
    if train_path.exists() or val_path.exists():
        if not (train_path.exists() and val_path.exists()):
            raise ValueError("Only one of the two reusable SFT parquet files exists")
        train_summary = reuse_subset(
            train_path, args.target, int(ready["train_repositories"])
        )
        val_summary = reuse_subset(val_path, 12_800)
    else:
        train_summary = build_subset(
            ready,
            train_path,
            args.target,
            args.seed,
            prefix_train=args.prefix_train,
        )
        val_summary = build_validation(ready, val_path)
    summary = {
        "format": "repoqa_baseline_data_v1",
        "source_ready": str(Path(args.ready).resolve()),
        "source_ready_sha256": sha256_file(args.ready),
        "seed": args.seed,
        "target": args.target,
        "train": train_summary,
        "validation": val_summary,
        "code2lora_embeddings": build_embedding_index(
            ready, Path(args.embedding_source), embedding_path
        ),
    }
    temporary = output / "READY.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "READY.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
