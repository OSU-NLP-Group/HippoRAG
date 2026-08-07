#!/usr/bin/env python3
"""Normalize SWE-Fixer into issue-fix rows for static Code2LoRA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


DEFAULT_DATASET = "internlm/SWE-Fixer-Train-110K"


def _get(row: Dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _prompt(repo: str, base_commit: str, problem_statement: str) -> str:
    return (
        "You are fixing an issue in a repository. The repository state has already "
        "been encoded into the adapter.\n\n"
        f"Repository: {repo}\n"
        f"Base commit: {base_commit}\n"
        "Language: python\n\n"
        "Issue:\n"
        "<issue>\n"
        f"{problem_statement.strip()}\n"
        "</issue>\n\n"
        "Return only a unified diff patch that applies to the repository at the base commit.\n"
    )


def normalize_row(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    repo = str(_get(row, "repo", "repo_id", "repository", default="")).strip()
    base_commit = str(_get(row, "base_commit", "commit", "commit_sha", default="")).strip()
    instance_id = str(_get(row, "instance_id", "id", default=f"swefixer_{index:08d}")).strip()
    problem = str(_get(row, "problem_statement", "issue", "description", default="")).strip()
    patch = str(_get(row, "patch", "gold_patch", default=""))
    test_patch = str(_get(row, "test_patch", default=""))
    hints_text = str(_get(row, "hints_text", "hints", default=""))
    created_at = _get(row, "created_at", default=None)
    usable = bool(repo and base_commit and problem and patch)
    missing = []
    for key, value in [
        ("repo", repo),
        ("base_commit", base_commit),
        ("problem_statement", problem),
        ("patch", patch),
    ]:
        if not value:
            missing.append(key)
    return {
        "source_dataset": "swe-fixer-train-110k",
        "row_index": index,
        "instance_id": instance_id,
        "repo_id": repo,
        "repo_slug": repo.replace("/", "__"),
        "base_commit": base_commit,
        "commit_sha": base_commit,
        "problem_statement": problem,
        "hints_text": hints_text,
        "created_at_raw": "" if created_at is None else str(created_at),
        "language": "python",
        "patch": patch,
        "target_patch": patch,
        "test_patch": test_patch,
        "prompt": _prompt(repo, base_commit, problem) if repo and base_commit and problem else "",
        "patch_char_len": len(patch),
        "test_patch_char_len": len(test_patch),
        "usable_for_train": usable,
        "unusable_reason": "" if usable else "missing_" + ",".join(missing),
        "metadata_json": json.dumps({
            "raw_keys": sorted(row.keys()),
            "created_at": created_at,
        }, sort_keys=True, default=str),
    }


def _write_parquet(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    tmp = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--schema-json", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    started = time.time()
    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    rows: List[Dict[str, Any]] = []
    schema_keys = None
    for idx, row in enumerate(ds):
        if schema_keys is None:
            schema_keys = sorted(row.keys())
        rows.append(normalize_row(dict(row), idx))
        if args.limit and len(rows) >= args.limit:
            break
        if len(rows) % 1000 == 0:
            print(f"normalized {len(rows)} rows", flush=True)

    output = Path(args.output)
    _write_parquet(rows, output)
    usable = sum(1 for row in rows if row["usable_for_train"])
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "rows": len(rows),
        "usable_for_train": usable,
        "schema_keys": schema_keys or [],
        "output": str(output),
        "elapsed_sec": round(time.time() - started, 3),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.schema_json:
        Path(args.schema_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.schema_json).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
