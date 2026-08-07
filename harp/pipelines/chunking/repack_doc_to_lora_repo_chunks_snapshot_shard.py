#!/usr/bin/env python3
"""Repack one contiguous snapshot range into canonical target-size chunks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_doc_to_lora_repo_chunks as base  # noqa: E402
import repack_doc_to_lora_repo_chunks_multires as repack  # noqa: E402


LARGE_REPO_INDICES = (
    103, 110, 168, 173, 176, 216, 241, 253, 359, 366, 371,
    385, 394, 402, 416, 421, 437, 439, 498, 535, 568, 570,
    610, 616, 617, 679, 714, 724, 743, 749, 801, 822, 832,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def select_contiguous_snapshot_shard(
    snapshots: list[tuple[tuple[str, str, str, str], list[str]]],
    shard_index: int,
    num_shards: int,
) -> list[tuple[tuple[str, str, str, str], list[str]]]:
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"Invalid snapshot shard {shard_index}/{num_shards}")
    total_cost = sum(max(1, len(chunk_ids)) for _, chunk_ids in snapshots)
    if total_cost <= 0:
        return []
    selected = []
    cumulative = 0
    for item in snapshots:
        cost = max(1, len(item[1]))
        # Assign by the midpoint of each snapshot's cost interval. This keeps
        # shards contiguous and approximately balances source chunk references.
        owner = min(num_shards - 1, int((cumulative + cost / 2) * num_shards / total_cost))
        if owner == shard_index:
            selected.append(item)
        cumulative += cost
    return selected


def load_needed_source_chunks(path: Path, needed_ids: set[str]) -> dict[str, repack.SourceChunk]:
    result: dict[str, repack.SourceChunk] = {}
    parquet = pq.ParquetFile(path, memory_map=True)
    for row_group in range(parquet.num_row_groups):
        id_table = parquet.read_row_group(row_group, columns=["chunk_id"])
        ids = [str(value) for value in id_table["chunk_id"].to_pylist()]
        take = [index for index, chunk_id in enumerate(ids) if chunk_id in needed_ids]
        if not take:
            continue
        table = parquet.read_row_group(
            row_group,
            columns=["chunk_id", "payload_text", "file_entries_json"],
        ).take(pa.array(take, type=pa.int64()))
        for chunk_id, payload, entries_json in zip(
            table["chunk_id"].to_pylist(),
            table["payload_text"].to_pylist(),
            table["file_entries_json"].to_pylist(),
        ):
            entries = json.loads(entries_json)
            parts = tuple(repack.parse_payload(payload, entries))
            result[str(chunk_id)] = repack.SourceChunk(str(chunk_id), parts)
    missing = needed_ids - result.keys()
    if missing:
        raise KeyError(f"Missing {len(missing)} source chunks; first={sorted(missing)[:3]}")
    return result


def build_partial(
    *,
    record: dict[str, Any],
    selected_snapshots: list[tuple[tuple[str, str, str, str], list[str]]],
    output_dir: Path,
    shard_index: int,
    num_shards: int,
    target_name: str,
) -> dict[str, Any]:
    audit_path = output_dir / "audit.json"
    if repack.output_complete(output_dir):
        return json.loads(audit_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_final = output_dir / "chunks.parquet"
    snapshots_final = output_dir / "snapshots.parquet"
    chunks_tmp = output_dir / f"chunks.parquet.tmp.{os.getpid()}"
    snapshots_tmp = output_dir / f"snapshots.parquet.tmp.{os.getpid()}"

    needed_ids = {chunk_id for _, chunk_ids in selected_snapshots for chunk_id in chunk_ids}
    source_chunks = load_needed_source_chunks(Path(record["chunks_parquet"]), needed_ids)
    config = repack.TARGETS[target_name]
    unique_payloads: set[str] = set()
    split_cache: dict[str, tuple[base.FilePart, ...]] = {}
    counters = {
        "processed_snapshots": 0,
        "source_chunk_references": 0,
        "snapshot_chunk_references": 0,
        "unique_chunks": 0,
        "deduplicated_chunk_references": 0,
        "context_tokens": 0,
        "file_parts": 0,
    }
    started = time.time()
    chunk_writer = base.BufferedParquetWriter(chunks_tmp, base.CHUNK_SCHEMA)
    snapshot_writer = base.BufferedParquetWriter(snapshots_tmp, base.SNAPSHOT_SCHEMA)
    try:
        for snapshot_key, source_ids in selected_snapshots:
            snapshot_repo, commit, base_commit, snapshot_id = snapshot_key
            all_parts: list[base.FilePart] = []
            for source_id in source_ids:
                counters["source_chunk_references"] += 1
                cached = split_cache.get(source_id)
                if cached is None:
                    expanded: list[base.FilePart] = []
                    for part in source_chunks[source_id].parts:
                        expanded.extend(repack.split_for_target(part, config["hard_tokens"]))
                    cached = tuple(expanded)
                    split_cache[source_id] = cached
                all_parts.extend(cached)
            packed = base.pack_file_parts(
                repack._TOKENIZER,
                all_parts,
                soft_tokens=config["soft_tokens"],
                hard_tokens=config["hard_tokens"],
                packing_safety_tokens=config["safety"],
            )
            total = len(packed)
            for index, chunk in enumerate(packed):
                if chunk.chunk_id not in unique_payloads:
                    unique_payloads.add(chunk.chunk_id)
                    chunk_writer.write(
                        {
                            "chunk_id": chunk.chunk_id,
                            "repo_id": snapshot_repo,
                            "payload_token_count": chunk.payload_token_count,
                            "payload_text": chunk.payload_text,
                            "file_entries_json": json.dumps(
                                [part.manifest_entry() for part in chunk.parts],
                                sort_keys=True,
                                ensure_ascii=False,
                            ),
                            "payload_sha256": chunk.chunk_id,
                        }
                    )
                    counters["unique_chunks"] += 1
                else:
                    counters["deduplicated_chunk_references"] += 1
                header = base.render_repository_header(snapshot_repo, commit, index, total)
                context_tokens = base.token_count(repack._TOKENIZER, header + chunk.payload_text) + 9
                if context_tokens > config["final_cap"]:
                    raise ValueError(
                        f"{target_name} context exceeds cap for {snapshot_id} chunk {index}: "
                        f"{context_tokens}>{config['final_cap']}"
                    )
                snapshot_writer.write(
                    {
                        "repo_id": snapshot_repo,
                        "commit_sha": commit,
                        "base_commit": base_commit,
                        "snapshot_id": snapshot_id,
                        "chunk_index": index,
                        "num_chunks": total,
                        "chunk_id": chunk.chunk_id,
                        "payload_token_count": chunk.payload_token_count,
                        "context_token_count": context_tokens,
                        "num_file_parts": len(chunk.parts),
                    }
                )
                counters["snapshot_chunk_references"] += 1
                counters["context_tokens"] += context_tokens
                counters["file_parts"] += len(chunk.parts)
            counters["processed_snapshots"] += 1
    except Exception:
        chunk_writer.close()
        snapshot_writer.close()
        chunks_tmp.unlink(missing_ok=True)
        snapshots_tmp.unlink(missing_ok=True)
        raise
    else:
        chunk_writer.close()
        snapshot_writer.close()
        chunks_tmp.replace(chunks_final)
        snapshots_tmp.replace(snapshots_final)

    audit = {
        "status": "complete",
        "repo_id": record["repo_id"],
        "repo_index": record["repo_index"],
        "target": target_name,
        "snapshot_shard_index": shard_index,
        "snapshot_num_shards": num_shards,
        "source_chunks_parquet": record["chunks_parquet"],
        "source_snapshots_parquet": record["snapshots_parquet"],
        "chunks_parquet": str(chunks_final),
        "snapshots_parquet": str(snapshots_final),
        "config": config,
        "counters": counters,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(audit_path, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repositories-jsonl", required=True, type=Path)
    parser.add_argument("--repo-index", required=True, type=int)
    parser.add_argument("--snapshot-shard-index", required=True, type=int)
    parser.add_argument("--snapshot-num-shards", required=True, type=int)
    parser.add_argument("--partial-root", required=True, type=Path)
    parser.add_argument("--target", required=True, choices=sorted(repack.TARGETS))
    parser.add_argument("--model-name", default="google/gemma-4-E2B-it")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only, use_fast=True
    )
    repack._TOKENIZER = tokenizer
    record = repack.load_repository_record(args.repositories_jsonl, args.repo_index)
    record["repo_index"] = args.repo_index
    snapshots = list(repack.iter_snapshots(Path(record["snapshots_parquet"])))
    selected = select_contiguous_snapshot_shard(
        snapshots, args.snapshot_shard_index, args.snapshot_num_shards
    )
    source_repo_dir = Path(record["chunks_parquet"]).parent.name
    output_dir = (
        args.partial_root
        / source_repo_dir
        / f"shard_{args.snapshot_shard_index:03d}_of_{args.snapshot_num_shards:03d}"
    )
    result = build_partial(
        record=record,
        selected_snapshots=selected,
        output_dir=output_dir,
        shard_index=args.snapshot_shard_index,
        num_shards=args.snapshot_num_shards,
        target_name=args.target,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
