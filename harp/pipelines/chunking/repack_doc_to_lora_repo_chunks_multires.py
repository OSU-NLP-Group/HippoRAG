#!/usr/bin/env python3
"""Derive smaller canonical Doc-to-LoRA chunks from the frozen 130K corpus.

This deliberately reads the immutable per-repository ``chunks.parquet`` and
``snapshots.parquet`` files instead of walking Git repositories again.  Each
invocation owns exactly one repository, making it safe to run as a large,
resumable Slurm array.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_doc_to_lora_repo_chunks as base  # noqa: E402


TARGETS = {
    "8k": {"soft_tokens": 7000, "hard_tokens": 8000, "final_cap": 8192, "safety": 256},
    "32k": {"soft_tokens": 30000, "hard_tokens": 32000, "final_cap": 32768, "safety": 512},
    "64k": {"soft_tokens": 60000, "hard_tokens": 64000, "final_cap": 65536, "safety": 1024},
}


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    parts: tuple[base.FilePart, ...]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_repository_record(path: Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise ValueError("repo-index must be non-negative")
    with path.open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                row = json.loads(line)
                if row.get("status") != "complete":
                    raise ValueError(f"Repository record {index} is not complete: {row}")
                return row
    raise IndexError(f"repo-index {index} is outside {path}")


def expected_opening(entry: dict[str, Any]) -> str:
    span = ""
    if entry["split_kind"] != "whole_file":
        span = f" lines={entry['start_line']}-{entry['end_line']}"
    return f"<<<FILE path={json.dumps(entry['path'], ensure_ascii=False)}{span}>>>\n"


def parse_payload(payload: str, entries: list[dict[str, Any]]) -> list[base.FilePart]:
    """Recover file-part text and verify that metadata exactly covers payload."""
    parts: list[base.FilePart] = []
    cursor = 0
    closing = "<<<END FILE>>>\n"
    for index, entry in enumerate(entries):
        opening = expected_opening(entry)
        if not payload.startswith(opening, cursor):
            raise ValueError(
                f"Payload/header mismatch at part {index} path={entry['path']!r} offset={cursor}"
            )
        body_start = cursor + len(opening)
        if index + 1 < len(entries):
            next_opening = expected_opening(entries[index + 1])
            boundary = closing + "\n" + next_opening
            boundary_at = payload.find(boundary, body_start)
            if boundary_at < 0:
                raise ValueError(f"Cannot find boundary after part {index} path={entry['path']!r}")
            body = payload[body_start:boundary_at]
            cursor = boundary_at + len(closing) + 1
        else:
            if not payload.endswith(closing):
                raise ValueError(f"Payload does not end after final part {entry['path']!r}")
            boundary_at = len(payload) - len(closing)
            if boundary_at < body_start:
                raise ValueError("Malformed final file part")
            body = payload[body_start:boundary_at]
            cursor = len(payload)

        part = base._make_part(
            tokenizer=_TOKENIZER,
            path=str(entry["path"]),
            blob_sha=str(entry["blob_sha"]),
            text=body,
            start_line=int(entry["start_line"]),
            end_line=int(entry["end_line"]),
            split_kind=str(entry["split_kind"]),
        )
        parts.append(part)
    if cursor != len(payload):
        raise ValueError(f"Unparsed payload suffix: {len(payload) - cursor} characters")
    return parts


def split_for_target(part: base.FilePart, hard_tokens: int) -> list[base.FilePart]:
    if part.token_count <= hard_tokens:
        return [part]
    if part.split_kind == "whole_file":
        result = base.parts_for_file(
            _TOKENIZER, part.path, part.blob_sha, part.text, hard_tokens
        )
    else:
        lines = part.text.splitlines(keepends=True) or [""]
        result = base.split_lines_to_parts(
            _TOKENIZER,
            part.path,
            part.blob_sha,
            lines,
            line_offset=part.start_line - 1,
            hard_tokens=hard_tokens,
            split_kind=f"multires_{part.split_kind}",
        )
    if "".join(item.text for item in result) != part.text:
        raise ValueError(f"Content changed while splitting {part.path!r}")
    if any(item.token_count > hard_tokens for item in result):
        raise ValueError(f"Oversized target part remained for {part.path!r}")
    return result


def load_source_chunks(path: Path) -> dict[str, SourceChunk]:
    table = pq.read_table(
        path,
        columns=["chunk_id", "payload_text", "file_entries_json"],
        memory_map=True,
    )
    result: dict[str, SourceChunk] = {}
    for chunk_id, payload, entries_json in zip(
        table["chunk_id"].to_pylist(),
        table["payload_text"].to_pylist(),
        table["file_entries_json"].to_pylist(),
    ):
        entries = json.loads(entries_json)
        parts = tuple(parse_payload(payload, entries))
        result[str(chunk_id)] = SourceChunk(str(chunk_id), parts)
    return result


def iter_snapshots(path: Path) -> Iterable[tuple[tuple[str, str, str, str], list[str]]]:
    table = pq.read_table(
        path,
        columns=[
            "repo_id", "commit_sha", "base_commit", "snapshot_id", "chunk_index", "chunk_id"
        ],
        memory_map=True,
    )
    rows = sorted(
        zip(*(table[name].to_pylist() for name in table.schema.names)),
        key=lambda row: (row[3], int(row[4])),
    )
    current_key: tuple[str, str, str, str] | None = None
    chunk_ids: list[str] = []
    for repo_id, commit, base_commit, snapshot_id, _chunk_index, chunk_id in rows:
        key = (str(repo_id), str(commit), str(base_commit), str(snapshot_id))
        if current_key is not None and key != current_key:
            yield current_key, chunk_ids
            chunk_ids = []
        current_key = key
        chunk_ids.append(str(chunk_id))
    if current_key is not None:
        yield current_key, chunk_ids


def output_complete(directory: Path) -> bool:
    audit = directory / "audit.json"
    if not audit.exists() or not (directory / "chunks.parquet").exists() or not (
        directory / "snapshots.parquet"
    ).exists():
        return False
    try:
        return json.loads(audit.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def build_target(
    *,
    record: dict[str, Any],
    source_chunks: dict[str, SourceChunk],
    snapshots_path: Path,
    output_root: Path,
    target_name: str,
    config: dict[str, int],
) -> dict[str, Any]:
    repo_id = str(record["repo_id"])
    source_repo_dir = Path(record["chunks_parquet"]).parent
    destination = output_root / target_name / "repositories" / source_repo_dir.name
    destination.mkdir(parents=True, exist_ok=True)
    if output_complete(destination):
        return json.loads((destination / "audit.json").read_text(encoding="utf-8"))

    chunks_final = destination / "chunks.parquet"
    snapshots_final = destination / "snapshots.parquet"
    chunks_tmp = destination / f"chunks.parquet.tmp.{os.getpid()}"
    snapshots_tmp = destination / f"snapshots.parquet.tmp.{os.getpid()}"
    for stale in (chunks_tmp, snapshots_tmp):
        stale.unlink(missing_ok=True)

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
        for snapshot_key, source_ids in iter_snapshots(snapshots_path):
            snapshot_repo, commit, base_commit, snapshot_id = snapshot_key
            all_parts: list[base.FilePart] = []
            for source_id in source_ids:
                counters["source_chunk_references"] += 1
                cached = split_cache.get(source_id)
                if cached is None:
                    expanded: list[base.FilePart] = []
                    for part in source_chunks[source_id].parts:
                        expanded.extend(split_for_target(part, config["hard_tokens"]))
                    cached = tuple(expanded)
                    split_cache[source_id] = cached
                all_parts.extend(cached)

            packed = base.pack_file_parts(
                _TOKENIZER,
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
                context_tokens = base.token_count(_TOKENIZER, header + chunk.payload_text) + 9
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
        "repo_id": repo_id,
        "target": target_name,
        "source_chunks_parquet": str(record["chunks_parquet"]),
        "source_snapshots_parquet": str(record["snapshots_parquet"]),
        "chunks_parquet": str(chunks_final),
        "snapshots_parquet": str(snapshots_final),
        "config": config,
        "counters": counters,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(destination / "audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repositories-jsonl", required=True, type=Path)
    parser.add_argument("--repo-index", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model-name", default="google/gemma-4-E2B-it")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted(TARGETS),
        default=("8k", "32k"),
    )
    args = parser.parse_args()

    global _TOKENIZER
    _TOKENIZER = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    record = load_repository_record(args.repositories_jsonl, args.repo_index)
    source_chunks = load_source_chunks(Path(record["chunks_parquet"]))
    results = []
    for target_name in args.targets:
        results.append(
            build_target(
                record=record,
                source_chunks=source_chunks,
                snapshots_path=Path(record["snapshots_parquet"]),
                output_root=args.output_root,
                target_name=target_name,
                config=TARGETS[target_name],
            )
        )
    print(json.dumps({"repo_index": args.repo_index, "results": results}, sort_keys=True), flush=True)


_TOKENIZER: Any = None


if __name__ == "__main__":
    main()
