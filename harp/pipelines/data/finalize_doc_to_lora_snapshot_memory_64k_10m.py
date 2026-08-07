#!/usr/bin/env python3
"""Audit, cost-schedule, and publish the exact 64K/K<=9 10M-QA corpus."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from build_doc_to_lora_snapshot_memory_shard import canonical_context_hash


FORMAT = "doc_to_lora_repoqa_snapshot_memory_v1"
RANK_ASSIGNMENT = "cost_bucketed_ddp_rounds_v1"
PARTITIONS = tuple(
    f"{stage}/{split}"
    for stage in ("stage1", "stage2a")
    for split in ("train", "val")
)
FAMILY_KIND = {
    "deterministic_original": "ast",
    "deterministic_cross_file_aug": "ast",
    "llm_generated": "llm",
    "llm_rewrite": "llm",
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_hex(*parts: object) -> str:
    return hashlib.sha256(
        "\0".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--base-ready", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--tokenizer", default="google/gemma-4-E2B-it")
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--qa-pack-token-budget", type=int, default=16384)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--fast-panel-snapshots", type=int, default=32)
    parser.add_argument("--checkpoint-panel-snapshots", type=int, default=100)
    parser.add_argument("--seed", default="snapshot-memory-64k-k9-10m-v1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_repo_metadata(repo_dir: str) -> dict[str, Any]:
    chunks = pq.read_table(
        Path(repo_dir) / "chunks.parquet",
        columns=["chunk_id", "payload_sha256"],
        memory_map=True,
    ).to_pylist()
    payload_hashes = {
        str(row["chunk_id"]): str(row["payload_sha256"]) for row in chunks
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
        "snapshots": {
            commit: list(dict.fromkeys(chunk for _index, chunk in sorted(values)))
            for commit, values in snapshots.items()
        },
    }


def schedule_groups(
    groups: list[dict[str, Any]], world_size: int, seed: str
) -> tuple[list[dict[str, Any]], int, dict[str, float]]:
    ordered = sorted(
        groups,
        key=lambda row: (
            int(row["estimated_cost_tokens"]),
            int(row["context_payload_tokens"]),
            int(row["answer_side_tokens"]),
            str(row["context_group_id"]),
        ),
    )
    rounds = []
    for start in range(0, len(ordered), world_size):
        members = ordered[start : start + world_size]
        members.sort(
            key=lambda row: digest_hex(seed, "rank", row["context_group_id"])
        )
        rounds.append(members)
    rounds.sort(
        key=lambda members: digest_hex(
            seed, "round", *(row["context_group_id"] for row in members)
        )
    )
    flattened = []
    padding_ordinal = 0
    overhead_numerator = 0
    overhead_denominator = 0
    for members in rounds:
        if len(members) < world_size:
            padding_ordinal = len(flattened)
        costs = [int(row["estimated_cost_tokens"]) for row in members]
        if costs:
            overhead_numerator += max(costs) * len(costs)
            overhead_denominator += sum(costs)
        flattened.extend(members)
    return (
        flattened,
        padding_ordinal,
        {
            "rounds": len(rounds),
            "estimated_straggler_overhead": (
                overhead_numerator / max(1, overhead_denominator)
            ),
        },
    )


def write_schedule(path: Path, groups: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        pa.Table.from_pylist(groups),
        temporary,
        compression="zstd",
        write_statistics=True,
        row_group_size=4_096,
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    ready_path = args.data_root / "READY.json"
    if ready_path.exists() and not args.overwrite:
        raise FileExistsError(ready_path)
    selection_rows = pq.read_table(args.selection, memory_map=True).to_pylist()
    selection = {
        (str(row["repo_id"]), str(row["commit_sha"])): row
        for row in selection_rows
    }
    if len(selection) != len(selection_rows):
        raise ValueError("Duplicate selected repository snapshot")

    source_ids = set()
    logical_ids = set()
    seen_snapshots = set()
    repo_split: dict[str, str] = {}
    repo_cache: dict[str, dict[str, Any]] = {}
    partition_records = {}
    total_family_counts = Counter()
    total_train_qas = 0
    all_val_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for partition in PARTITIONS:
        stage, split = partition.split("/")
        qa_files = sorted(
            Path(path)
            for path in glob.glob(str(args.data_root / stage / split / "*.qa.parquet"))
        )
        group_files = sorted(
            Path(path)
            for path in glob.glob(
                str(args.data_root / stage / split / "snapshot_shard_*.groups.parquet")
            )
        )
        if not qa_files or not group_files:
            raise ValueError(f"Missing QA/group files for {partition}")

        groups = [
            group
            for path in group_files
            for group in pq.read_table(path, memory_map=True).to_pylist()
        ]
        family_counts = Counter()
        source_family_counts = Counter()
        k_histogram = Counter()
        logical_qas = physical_packs = supervised_tokens = answer_side_tokens = 0
        referenced_row_groups = set()
        for group in groups:
            key = (str(group["repo_id"]), str(group["commit_sha"]))
            selected = selection.get(key)
            if (
                selected is None
                or str(selected["split"]) != split
                or str(selected["stage"]) != stage
            ):
                raise ValueError(f"Unselected or wrong-partition group {key}")
            if key in seen_snapshots:
                raise ValueError(f"Snapshot occurs in multiple groups: {key}")
            seen_snapshots.add(key)
            prior_split = repo_split.get(key[0])
            if prior_split and prior_split != split:
                raise ValueError(f"Repository split leakage for {key[0]}")
            repo_split[key[0]] = split

            repo_dir = str(group["repo_dir"])
            if repo_dir != str(selected["repo_dir"]):
                raise ValueError(f"Freeze directory mismatch for {key}")
            metadata = repo_cache.get(repo_dir)
            if metadata is None:
                metadata = load_repo_metadata(repo_dir)
                repo_cache[repo_dir] = metadata
            all_chunks = metadata["snapshots"].get(key[1])
            group_chunks = [str(value) for value in group["selected_chunk_ids"]]
            if (
                not all_chunks
                or group_chunks != all_chunks
                or len(all_chunks) != int(selected["num_chunks"])
                or not 1 <= len(all_chunks) <= 9
            ):
                raise ValueError(f"Group does not use a complete 64K K<=9 snapshot {key}")
            actual_hash = canonical_context_hash(
                key[0], key[1], all_chunks, all_chunks, metadata["payload_hashes"]
            )
            if actual_hash != str(group["selected_context_hash"]):
                raise ValueError(f"Context hash mismatch for {key}")

            qa_file = Path(str(group["qa_file"]))
            row_group = int(group["qa_row_group"])
            reference = (str(qa_file.resolve()), row_group)
            if reference in referenced_row_groups:
                raise ValueError(f"QA row group referenced twice: {reference}")
            referenced_row_groups.add(reference)
            parquet = pq.ParquetFile(qa_file, memory_map=True)
            if row_group >= parquet.metadata.num_row_groups:
                raise ValueError(f"Invalid QA row group for {key}")
            qa_table = parquet.read_row_group(row_group)
            if int(group["qa_start"]) != 0 or int(group["qa_count"]) != qa_table.num_rows:
                raise ValueError(f"QA group range is not the full row group for {key}")
            rows = qa_table.to_pylist()
            starts = [int(value) for value in group["qa_pack_starts"]]
            counts = [int(value) for value in group["qa_pack_counts"]]
            token_counts = [int(value) for value in group["qa_pack_token_counts"]]
            expected_starts = []
            offset = 0
            for count in counts:
                expected_starts.append(offset)
                offset += count
            if (
                not starts
                or starts != expected_starts
                or offset != len(rows)
                or len(starts) != len(token_counts)
            ):
                raise ValueError(f"Malformed pack metadata for {key}")
            for pack_index, (start, count, frozen_tokens) in enumerate(
                zip(starts, counts, token_counts)
            ):
                pack = rows[start : start + count]
                actual_tokens = sum(int(row["total_qa_tokens"]) for row in pack)
                if actual_tokens != frozen_tokens or actual_tokens > args.qa_pack_token_budget:
                    raise ValueError(f"Pack token budget mismatch for {key}")
                if any(int(row["snapshot_pack_index"]) != pack_index for row in pack):
                    raise ValueError(f"Snapshot pack index mismatch for {key}")

            local_families = Counter()
            for row in rows:
                source_id = str(row["source_qa_id"])
                logical_id = str(row["logical_example_id"])
                if source_id in source_ids or logical_id in logical_ids:
                    raise ValueError(f"Duplicate QA ID in corpus: {source_id}")
                source_ids.add(source_id)
                logical_ids.add(logical_id)
                if (
                    str(row["repo_id"]) != key[0]
                    or str(row["commit_sha"]) != key[1]
                    or str(row["split"]) != split
                    or str(row["stage"]) != stage
                ):
                    raise ValueError(f"QA partition/snapshot mismatch for {source_id}")
                start = int(row["response_start"])
                end = int(row["response_end"])
                ids = row["input_ids"]
                if not 0 < start < end <= len(ids) or int(row["answer_token_count"]) != end - start:
                    raise ValueError(f"Invalid answer span for {source_id}")
                source_family = str(row["source_family"])
                qa_family = str(row["qa_family"])
                if FAMILY_KIND.get(source_family) != qa_family:
                    raise ValueError(f"QA family mismatch for {source_id}")
                if int(row["duplicate_multiplicity"]) != 1:
                    raise ValueError("The 10M corpus requires unique multiplicity")
                local_families[qa_family] += 1
                family_counts[qa_family] += 1
                source_family_counts[source_family] += 1
                supervised_tokens += end - start
                answer_side_tokens += len(ids)
            if (
                local_families["ast"] != int(selected["target_ast_qas"])
                or local_families["llm"] != int(selected["target_llm_qas"])
                or int(group["ast_qas"]) != local_families["ast"]
                or int(group["llm_qas"]) != local_families["llm"]
            ):
                raise ValueError(f"Frozen family target mismatch for {key}")
            logical_qas += len(rows)
            physical_packs += len(counts)
            k_histogram[len(all_chunks)] += 1
            if split == "val":
                all_val_groups[stage].append(group)

        expected_row_groups = {
            (str(path.resolve()), row_group)
            for path in qa_files
            for row_group in range(pq.ParquetFile(path).metadata.num_row_groups)
        }
        if referenced_row_groups != expected_row_groups:
            raise ValueError(f"QA row-group coverage mismatch for {partition}")

        scheduled, padding_ordinal, schedule_stats = schedule_groups(
            groups, args.world_size, f"{args.seed}-{partition}"
        )
        schedule_path = args.data_root / "schedules" / stage / split / "groups.parquet"
        write_schedule(schedule_path, scheduled)
        record = {
            "logical_qas": logical_qas,
            "physical_qa_rows": logical_qas,
            "context_groups": len(groups),
            "optimizer_steps": math.ceil(len(groups) / args.world_size),
            "physical_qa_packs": physical_packs,
            "supervised_tokens": supervised_tokens,
            "answer_side_tokens": answer_side_tokens,
            "qa_family_counts": dict(sorted(family_counts.items())),
            "source_family_counts": dict(sorted(source_family_counts.items())),
            "context_group_chunk_histogram": {
                str(key): value for key, value in sorted(k_histogram.items())
            },
            "rank_assignment": RANK_ASSIGNMENT,
            "world_size": args.world_size,
            "padding_group_ordinal": padding_ordinal,
            "schedule_stats": schedule_stats,
            "group_manifests": [str(schedule_path.resolve())],
            "group_manifest_sha256": {
                str(schedule_path.resolve()): sha256(schedule_path)
            },
            "qa_files": [str(path.resolve()) for path in qa_files],
            "qa_file_sha256": {
                str(path.resolve()): sha256(path) for path in qa_files
            },
        }
        partition_records[partition] = record
        total_family_counts.update(family_counts)
        if split == "train":
            total_train_qas += logical_qas

    if seen_snapshots != set(selection):
        raise ValueError(
            f"Selected snapshot coverage mismatch: seen={len(seen_snapshots)}, "
            f"expected={len(selection)}"
        )
    if total_train_qas != 10_000_000:
        raise ValueError(f"Training corpus has {total_train_qas} QAs, expected 10M")

    validation_panels: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for stage, groups in all_val_groups.items():
        ordered, _, _ = schedule_groups(groups, args.world_size, f"{args.seed}-{stage}-panel")
        for name, limit in (
            ("fast", args.fast_panel_snapshots),
            ("checkpoint", args.checkpoint_panel_snapshots),
        ):
            panel = ordered[: min(limit, len(ordered))]
            panel_path = args.data_root / "validation_panels" / stage / name / "groups.parquet"
            write_schedule(panel_path, panel)
            validation_panels[stage][name] = {
                "logical_qas": sum(int(row["qa_count"]) for row in panel),
                "physical_qa_rows": sum(int(row["qa_count"]) for row in panel),
                "context_groups": len(panel),
                "physical_qa_packs": sum(len(row["qa_pack_counts"]) for row in panel),
                "rank_assignment": RANK_ASSIGNMENT,
                "world_size": args.world_size,
                "padding_group_ordinal": max(0, len(panel) - 1),
                "group_manifests": [str(panel_path.resolve())],
                "group_manifest_sha256": {
                    str(panel_path.resolve()): sha256(panel_path)
                },
            }

    train_repos = sorted(repo for repo, split in repo_split.items() if split == "train")
    val_repos = sorted(repo for repo, split in repo_split.items() if split == "val")
    split_files = {}
    for split, repos in (("train", train_repos), ("val", val_repos)):
        path = args.data_root / f"{split}_repositories.txt"
        temporary = path.with_suffix(".txt.tmp")
        temporary.write_text("".join(f"{repo}\n" for repo in repos))
        temporary.replace(path)
        split_files[split] = {"path": str(path.resolve()), "sha256": sha256(path)}

    base_ready = json.loads(args.base_ready.read_text())
    train_family = Counter()
    for partition, record in partition_records.items():
        if partition.endswith("/train"):
            train_family.update(record["qa_family_counts"])
    ready = {
        "format": FORMAT,
        "loss": "answer_token_ce",
        "objective": "snapshot_memory_weighted_answer_ce_plus_l1_and_wrong_repo_contrastive",
        "qa_loss_weights": {"ast": 1.0, "llm": 1.8},
        "train_logical_qas": total_train_qas,
        "train_qa_family_counts": dict(sorted(train_family.items())),
        "train_effective_loss_mass": {
            "ast": float(train_family["ast"]),
            "llm": 1.8 * float(train_family["llm"]),
        },
        "selected_snapshots": len(selection),
        "selected_logical_qas": sum(
            record["logical_qas"] for record in partition_records.values()
        ),
        "qa_family_counts": dict(sorted(total_family_counts.items())),
        "snapshot_policy": {
            "resolution": "64k",
            "complete_snapshot": True,
            "minimum_chunks": 1,
            "maximum_chunks": 9,
            "stage1_maximum_chunks": 2,
            "stage2_minimum_chunks": 3,
            "question_independent_context": True,
            "qa_pack_token_budget": args.qa_pack_token_budget,
            "cost_bucketed_ddp_scheduling": True,
        },
        "repository_disjoint_split": True,
        "train_repositories": len(train_repos),
        "val_repositories": len(val_repos),
        "repository_split_files": split_files,
        "source_corpus_ready": str(args.base_ready.resolve()),
        "source_corpus_ready_sha256": sha256(args.base_ready),
        "selection": str(args.selection.resolve()),
        "selection_sha256": sha256(args.selection),
        "freeze_root": str(args.freeze_root.resolve()),
        "freeze_manifest_sha256": sha256(args.freeze_root / "manifest.json"),
        "freeze_snapshot_index_sha256": sha256(args.freeze_root / "snapshot_index.parquet"),
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "chat_template": base_ready["chat_template"],
        "qa_token_budget": args.qa_pack_token_budget,
        "partitions": partition_records,
        "validation_panels": validation_panels,
        "validated_invariants": [
            "exactly_10000000_training_qas",
            "all_eligible_training_llm_qas_are_included",
            "remaining_training_budget_is_deterministic_ast",
            "all_selected_snapshots_appear_exactly_once",
            "repository_split_is_disjoint",
            "every_context_is_a_complete_question_independent_64k_snapshot",
            "every_snapshot_has_between_one_and_nine_canonical_chunks",
            "every_qa_pack_is_complete_disjoint_and_within_16384_tokens",
            "cost_similar_snapshots_share_each_eight_rank_ddp_round",
            "logical_and_source_ids_are_unique",
        ],
    }
    temporary = ready_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
    temporary.replace(ready_path)
    print(json.dumps(ready, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
