#!/usr/bin/env python3
"""Freeze the exact valid-repository intersection across chunk resolutions.

The chunk roots may contain cancelled-task directories.  A repository enters
the freeze only when every requested resolution has a completed audit plus
readable ``chunks.parquet`` and ``snapshots.parquet`` files, and the exact set
of snapshot commits agrees at every resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


INDEX_SCHEMA = pa.schema(
    [
        ("repo_id", pa.string()),
        ("commit_sha", pa.string()),
        ("base_commit", pa.string()),
        ("resolution", pa.string()),
        ("num_chunks", pa.int32()),
        ("repo_dir", pa.string()),
    ]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_footer_sha256(path: Path) -> str:
    """Hash the compact Parquet footer without rereading every payload byte."""

    with path.open("rb") as handle:
        handle.seek(-8, os.SEEK_END)
        trailer = handle.read(8)
        if len(trailer) != 8 or trailer[4:] != b"PAR1":
            raise ValueError(f"Invalid Parquet trailer: {path}")
        footer_size = struct.unpack("<I", trailer[:4])[0]
        handle.seek(-(8 + footer_size), os.SEEK_END)
        footer = handle.read(footer_size)
    if len(footer) != footer_size:
        raise ValueError(f"Truncated Parquet footer: {path}")
    return hashlib.sha256(footer).hexdigest()


def parse_resolution(raw: str) -> tuple[str, Path]:
    name, separator, root = raw.partition("=")
    if not separator or not name or not root:
        raise argparse.ArgumentTypeError("resolution must be NAME=ROOT")
    return name, Path(root).resolve()


def scan_root(name: str, root: Path) -> dict[str, dict[str, Any]]:
    repositories = root / "repositories"
    if not repositories.is_dir():
        raise FileNotFoundError(repositories)
    found: dict[str, dict[str, Any]] = {}
    for repo_dir in sorted(path for path in repositories.iterdir() if path.is_dir()):
        chunks = repo_dir / "chunks.parquet"
        snapshots = repo_dir / "snapshots.parquet"
        audit_path = repo_dir / "audit.json"
        if not (chunks.is_file() and snapshots.is_file() and audit_path.is_file()):
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "complete":
            continue
        repo_id = str(audit.get("repo_id") or "").strip()
        if not repo_id:
            raise ValueError(f"Completed audit has no repo_id: {audit_path}")
        if repo_id in found:
            raise ValueError(f"Duplicate {name} repository id: {repo_id}")
        chunks_file = pq.ParquetFile(chunks, memory_map=True)
        snapshots_file = pq.ParquetFile(snapshots, memory_map=True)
        required_chunks = {"chunk_id", "payload_text"}
        required_snapshots = {"repo_id", "commit_sha", "chunk_id", "chunk_index"}
        if not required_chunks <= set(chunks_file.schema_arrow.names):
            raise ValueError(f"{chunks} lacks {sorted(required_chunks)}")
        if not required_snapshots <= set(snapshots_file.schema_arrow.names):
            raise ValueError(f"{snapshots} lacks {sorted(required_snapshots)}")
        snapshot_columns = ["repo_id", "commit_sha", "chunk_id"]
        if "base_commit" in snapshots_file.schema_arrow.names:
            snapshot_columns.append("base_commit")
        rows = pq.read_table(
            snapshots, columns=snapshot_columns, memory_map=True
        ).to_pylist()
        commits: dict[str, dict[str, Any]] = {}
        for row in rows:
            if str(row["repo_id"]) != repo_id:
                raise ValueError(f"repo_id mismatch in {snapshots}")
            commit = str(row["commit_sha"])
            entry = commits.setdefault(
                commit,
                {
                    "base_commit": str(row.get("base_commit") or commit),
                    "chunks": set(),
                },
            )
            entry["chunks"].add(str(row["chunk_id"]))
        if not commits:
            raise ValueError(f"No snapshots in {snapshots}")
        found[repo_id] = {
            "repo_dir": str(repo_dir.resolve()),
            "audit": str(audit_path.resolve()),
            "audit_sha256": sha256(audit_path),
            "chunks_parquet": str(chunks.resolve()),
            "chunks_bytes": chunks.stat().st_size,
            "chunks_rows": chunks_file.metadata.num_rows,
            "chunks_footer_sha256": parquet_footer_sha256(chunks),
            "snapshots_parquet": str(snapshots.resolve()),
            "snapshots_bytes": snapshots.stat().st_size,
            "snapshots_rows": snapshots_file.metadata.num_rows,
            "snapshots_sha256": sha256(snapshots),
            "commits": commits,
        }
    return found


def write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution", type=parse_resolution, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-repositories", type=int, required=True)
    args = parser.parse_args()
    resolutions = dict(args.resolution)
    if len(resolutions) != len(args.resolution):
        raise ValueError("Duplicate resolution name")
    if args.output.exists():
        raise FileExistsError(args.output)

    scanned = {name: scan_root(name, root) for name, root in resolutions.items()}
    valid_sets = {name: set(records) for name, records in scanned.items()}
    intersection = set.intersection(*valid_sets.values())
    if len(intersection) != args.expected_repositories:
        raise ValueError(
            f"Expected {args.expected_repositories} common repositories, "
            f"found {len(intersection)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.parent / f".{args.output.name}.tmp.{os.getpid()}"
    temporary.mkdir()
    try:
        repository_ids = "".join(f"{repo_id}\n" for repo_id in sorted(intersection))
        (temporary / "repository_ids.txt").write_text(
            repository_ids, encoding="utf-8"
        )
        records_handle = (temporary / "repositories.jsonl").open(
            "w", encoding="utf-8"
        )
        index_rows: list[dict[str, Any]] = []
        try:
            for repo_id in sorted(intersection):
                commit_sets = {
                    name: set(scanned[name][repo_id]["commits"])
                    for name in resolutions
                }
                first_name = next(iter(resolutions))
                expected_commits = commit_sets[first_name]
                disagreements = {
                    name: sorted(values ^ expected_commits)
                    for name, values in commit_sets.items()
                    if values != expected_commits
                }
                if disagreements:
                    raise ValueError(
                        f"Snapshot commit mismatch for {repo_id}: {disagreements}"
                    )
                per_resolution = {}
                for name in resolutions:
                    record = scanned[name][repo_id]
                    commits = record.pop("commits")
                    per_resolution[name] = record
                    for commit in sorted(commits):
                        index_rows.append(
                            {
                                "repo_id": repo_id,
                                "commit_sha": commit,
                                "base_commit": commits[commit]["base_commit"],
                                "resolution": name,
                                "num_chunks": len(commits[commit]["chunks"]),
                                "repo_dir": record["repo_dir"],
                            }
                        )
                records_handle.write(
                    json.dumps(
                        {"repo_id": repo_id, "resolutions": per_resolution},
                        sort_keys=True,
                    )
                    + "\n"
                )
        finally:
            records_handle.close()
        index_rows.sort(
            key=lambda row: (
                str(row["repo_id"]),
                str(row["commit_sha"]),
                str(row["resolution"]),
            )
        )
        pq.write_table(
            pa.Table.from_pylist(index_rows, schema=INDEX_SCHEMA),
            temporary / "snapshot_index.parquet",
            compression="zstd",
        )
        exclusions = {
            name: sorted(set().union(*valid_sets.values()) - values)
            for name, values in valid_sets.items()
        }
        manifest = {
            "format": "doc_to_lora_multires_freeze_v1",
            "status": "frozen",
            "created_unix": time.time(),
            "resolutions": {name: str(root) for name, root in resolutions.items()},
            "valid_repository_counts": {
                name: len(values) for name, values in valid_sets.items()
            },
            "intersection_repositories": len(intersection),
            "snapshot_index_rows": len(index_rows),
            "excluded_from_intersection_by_resolution": exclusions,
            "repository_ids_file": "repository_ids.txt",
            "repository_ids_sha256": sha256(temporary / "repository_ids.txt"),
            "repository_records_file": "repositories.jsonl",
            "repository_records_sha256": sha256(temporary / "repositories.jsonl"),
            "snapshot_index_file": "snapshot_index.parquet",
            "snapshot_index_sha256": sha256(temporary / "snapshot_index.parquet"),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
    except BaseException:
        # Leave the temporary directory for forensic inspection.  A subsequent
        # invocation uses a PID-specific path and will not overwrite it.
        raise
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
