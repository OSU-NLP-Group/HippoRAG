#!/usr/bin/env python3
"""Merge validated LLM QA JSONL shards into one accepted JSONL artifact."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def shard_index(payload: dict[str, Any], fallback: int) -> int:
    args = payload.get("args") or {}
    source = args.get("packs_jsonl") or payload.get("packs_jsonl") or payload.get("output_jsonl") or ""
    text = str(source)
    marker = ".shard_"
    if marker in text:
        raw = text.split(marker, 1)[1].split("_of_", 1)[0]
        try:
            return int(raw)
        except ValueError:
            pass
    try:
        return int(args.get("shard_index", fallback))
    except (TypeError, ValueError):
        return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-glob", required=True)
    parser.add_argument("--audit-glob", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=0)
    args = parser.parse_args()

    accepted_paths = [Path(path) for path in sorted(glob.glob(args.accepted_glob))]
    audit_paths = [Path(path) for path in sorted(glob.glob(args.audit_glob))]
    if args.expected_shards and len(accepted_paths) != args.expected_shards:
        raise SystemExit(f"expected {args.expected_shards} accepted JSONL shards, found {len(accepted_paths)}")
    if args.expected_shards and len(audit_paths) != args.expected_shards:
        raise SystemExit(f"expected {args.expected_shards} LLM QA audit shards, found {len(audit_paths)}")
    if len(accepted_paths) != len(audit_paths):
        raise SystemExit(f"accepted/audit shard count mismatch: {len(accepted_paths)} vs {len(audit_paths)}")

    accepted_by_path = {path.resolve(): path for path in accepted_paths}
    rows: list[dict[str, Any]] = []
    seen_questions: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    audits: list[dict[str, Any]] = []
    indices: set[int] = set()
    for fallback, audit_path in enumerate(audit_paths):
        audit = read_json(audit_path)
        index = shard_index(audit, fallback)
        if index in indices:
            raise SystemExit(f"duplicate LLM QA shard index in audits: {index}")
        indices.add(index)
        audit_output = audit.get("output_jsonl")
        if not audit_output:
            raise SystemExit(f"audit missing output_jsonl: {audit_path}")
        accepted_path = accepted_by_path.pop(Path(str(audit_output)).resolve(), None)
        if accepted_path is None:
            raise SystemExit(f"audit output_jsonl does not match accepted shard: {audit_path}")
        accepted_count = 0
        for _line_no, row in iter_jsonl(accepted_path):
            key = (str(row.get("repo_id") or ""), str(row.get("question") or "").strip().lower())
            if key in seen_questions:
                raise SystemExit(f"duplicate accepted LLM QA question across shards: {key[0]} {key[1]}")
            seen_questions.add(key)
            rows.append(row)
            accepted_count += 1
        if int(audit.get("accepted") or 0) != accepted_count:
            raise SystemExit(f"audit accepted count does not match shard rows: {audit_path}")
        counts.update(audit.get("counts") or {})
        audits.append({
            "shard_index": index,
            "accepted_jsonl": str(accepted_path),
            "audit": str(audit_path),
            "accepted": accepted_count,
            "output_jsonl": str(audit_output),
        })

    if accepted_by_path:
        unused = ", ".join(str(path) for path in sorted(accepted_by_path.values()))
        raise SystemExit(f"accepted LLM QA shard has no matching validation audit output_jsonl: {unused}")

    if args.expected_shards:
        expected = set(range(args.expected_shards))
        if indices != expected:
            missing = sorted(expected - indices)
            extra = sorted(indices - expected)
            pieces = []
            if missing:
                pieces.append(f"missing shard indices {missing}")
            if extra:
                pieces.append(f"unexpected shard indices {extra}")
            raise SystemExit("; ".join(pieces))
    if not rows:
        raise SystemExit("no accepted LLM QA rows found across shards")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(args.output_jsonl)

    payload = {
        "merged": True,
        "accepted": len(rows),
        "counts": dict(counts),
        "output_jsonl": str(args.output_jsonl),
        "expected_shards": args.expected_shards,
        "shards": sorted(audits, key=lambda item: item["shard_index"]),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
