#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


TEXT_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".sh", ".bash", ".zsh",
    ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".md",
    ".rst", ".txt",
}
EXCLUDED_PARTS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules",
    "dist", "build", "vendor", "third_party", "site-packages", ".tox",
    "_vendor", "vendored", "vendors", "thirdparty", "external", "extern",
    "deps", "libpsutil", ".eggs",
}
SKIP_BASENAMES = {"version.py", "_version.py"}


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def eligible_enriched(path: str) -> bool:
    p = Path(path)
    lowered = {part.lower() for part in p.parts}
    if lowered & EXCLUDED_PARTS or p.name.lower() in SKIP_BASENAMES:
        return False
    name = p.name.lower()
    if name.endswith("_generated.py") or name.endswith(".generated.py"):
        return False
    if name.endswith("_pb2.py") or name.endswith("_pb2_grpc.py"):
        return False
    return p.suffix.lower() in TEXT_EXTENSIONS


def read_rows(path: Path) -> list[dict[str, object]]:
    table = pq.read_table(path, columns=["repo_id", "base_commit", "repo_path", "usable_repo_snapshot"])
    cols = table.to_pydict()
    return [
        {name: cols[name][i] for name in cols}
        for i in range(table.num_rows)
        if bool(cols["usable_repo_snapshot"][i])
    ]


def select_rows(rows: list[dict[str, object]], all_snapshots: bool) -> list[dict[str, object]]:
    if all_snapshots:
        return rows
    by_repo: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_repo[str(row["repo_id"])].append(row)
    return [items[len(items) // 2] for _, items in sorted(by_repo.items())]


def ls_tree(repo: Path, commit: str) -> list[tuple[str, str, int]]:
    proc = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "-l", commit],
        capture_output=True, text=True, check=False, timeout=300,
    )
    if proc.returncode:
        return []
    out = []
    for line in proc.stdout.splitlines():
        try:
            meta, path = line.split("\t", 1)
            _mode, kind, blob, size_text = meta.split()
            size = int(size_text)
        except ValueError:
            continue
        if kind == "blob" and size > 0:
            out.append((blob, path, size))
    return out


class BlobReader:
    def __init__(self, repo: Path):
        self.proc = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

    def read(self, blob: str) -> bytes:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(blob.encode("ascii") + b"\n")
        self.proc.stdin.flush()
        header = self.proc.stdout.readline().decode("ascii", errors="replace").strip()
        fields = header.split()
        if len(fields) != 3 or fields[1] != "blob":
            return b""
        data = self.proc.stdout.read(int(fields[2]))
        self.proc.stdout.read(1)
        return data

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.terminate()
        self.proc.wait(timeout=10)


def summarize(records: list[dict[str, object]], field: str, chunk_tokens: int, rank: int) -> dict[str, object]:
    values = [int(row[field]) for row in records]
    chunks = [max(1, math.ceil(value / chunk_tokens)) for value in values]
    return {
        "snapshots": len(values),
        "tokens": {"min": min(values, default=0), "p25": percentile(values, .25),
                   "median": percentile(values, .5), "p75": percentile(values, .75),
                   "p90": percentile(values, .9), "p95": percentile(values, .95),
                   "max": max(values, default=0)},
        "chunks": {"min": min(chunks, default=0), "p25": percentile(chunks, .25),
                   "median": percentile(chunks, .5), "p75": percentile(chunks, .75),
                   "p90": percentile(chunks, .9), "p95": percentile(chunks, .95),
                   "max": max(chunks, default=0)},
        "chunk_histogram": {str(k): chunks.count(k) for k in sorted(set(chunks))},
        "fraction_one_chunk": (sum(k == 1 for k in chunks) / len(chunks)) if chunks else 0.0,
        "active_rank_without_bias": {"median": percentile(chunks, .5) * rank,
                                     "p95": percentile(chunks, .95) * rank,
                                     "max": max(chunks, default=0) * rank},
        "allocated_rank_with_d2l_bias": {"median": (percentile(chunks, .5) + 1) * rank,
                                         "p95": (percentile(chunks, .95) + 1) * rank,
                                         "max": (max(chunks, default=0) + 1) * rank},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--all-snapshots", action="store_true")
    ap.add_argument("--chunk-tokens", type=int, default=130000)
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    rows = select_rows(read_rows(args.manifest), args.all_snapshots)
    by_repo: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_repo[str(row["repo_id"])].append(row)

    records: list[dict[str, object]] = []
    for repo_idx, (repo_id, repo_rows) in enumerate(sorted(by_repo.items()), 1):
        repo = Path(str(repo_rows[0]["repo_path"]))
        reader = BlobReader(repo)
        token_cache: dict[tuple[str, str], int] = {}
        try:
            for row in repo_rows:
                commit = str(row["base_commit"])
                files = ls_tree(repo, commit)
                py_segments = []
                enriched_segments = []
                for blob, path, size in files:
                    use_py = path.endswith(".py") and size <= 2_000_000
                    use_enriched = size <= 524_288 and eligible_enriched(path)
                    if not use_py and not use_enriched:
                        continue
                    key = (blob, path)
                    if key not in token_cache:
                        raw = reader.read(blob)
                        try:
                            text = raw.decode("utf-8")
                        except UnicodeDecodeError:
                            token_cache[key] = 0
                        else:
                            segment = f"# path: {path}\n{text}\n\n"
                            token_cache[key] = len(tokenizer.encode(segment, add_special_tokens=False))
                    count = token_cache[key]
                    if use_py and count:
                        py_segments.append(count)
                    if use_enriched and count:
                        enriched_segments.append(count)
                records.append({
                    "repo_id": repo_id, "commit": commit,
                    "python_files": len(py_segments), "python_tokens": sum(py_segments),
                    "enriched_files": len(enriched_segments), "enriched_tokens": sum(enriched_segments),
                })
        finally:
            reader.close()
        if repo_idx % 10 == 0 or repo_idx == len(by_repo):
            print(f"repos {repo_idx}/{len(by_repo)} snapshots={len(records)}", flush=True)

    payload = {
        "manifest": str(args.manifest), "selection": "all" if args.all_snapshots else "one_median_row_per_repo",
        "chunk_tokens": args.chunk_tokens, "base_rank": args.rank,
        "unique_repos": len(by_repo), "records": records,
        "python_only": summarize(records, "python_tokens", args.chunk_tokens, args.rank),
        "enriched_extensions": summarize(records, "enriched_tokens", args.chunk_tokens, args.rank),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("selection", "chunk_tokens", "unique_repos")}, indent=2))
    print(json.dumps({"python_only": payload["python_only"], "enriched_extensions": payload["enriched_extensions"]}, indent=2))


if __name__ == "__main__":
    main()

