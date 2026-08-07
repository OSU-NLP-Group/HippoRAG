#!/usr/bin/env python3
"""Validate cross-file augmentation rows and append them to an existing final pool."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import finalize_code2lora_qa_hardreject_dedup as finalizer


def iter_jsonl(path: Path):
    yield from finalizer.iter_jsonl(path)


def content_key(row: dict[str, Any]) -> str:
    repo_id = finalizer.normalize_space(row.get("repo_id"))
    base_commit = finalizer.normalize_space(row.get("base_commit") or row.get("commit_sha") or row.get("commit"))
    question = finalizer.normalize_space(row.get("question"))
    answer = finalizer.normalize_space(row.get("answer"))
    return finalizer.stable_hash_json(
        {
            "repo_id": repo_id,
            "base_commit": base_commit,
            "question": finalizer.normalize_for_key(question),
            "answer": finalizer.normalize_for_key(answer),
        }
    )


def canonical_aug_row(row: dict[str, Any], source_file: str, line_no: int):
    rid = finalizer.row_id(row)
    return finalizer.canonical_row(
        row,
        source_family="deterministic_cross_file_aug",
        source_file=source_file,
        line_no=line_no,
        priority=15,
        replacement_key=f"deterministic_cross_file_aug:{rid}",
    )


def load_old_content_keys(old_final_root: Path) -> set[str]:
    keys: set[str] = set()
    for path in sorted(old_final_root.glob("final_qa.shard_*_of_*.jsonl")):
        for _line_no, row, err in iter_jsonl(path):
            if err or row is None:
                continue
            keys.add(content_key(row))
    return keys


def hardlink_old_shards(old_final_root: Path, out_final_root: Path) -> int:
    out_final_root.mkdir(parents=True, exist_ok=True)
    linked = 0
    for old_path in sorted(old_final_root.glob("final_qa.shard_*_of_*.jsonl")):
        new_path = out_final_root / f"base_{old_path.name}"
        if new_path.exists():
            continue
        try:
            os.link(old_path, new_path)
        except FileExistsError:
            pass
        linked += 1
    return linked


def load_old_summary(old_root: Path) -> dict[str, Any]:
    summary_path = old_root / "final_qa_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def load_old_category_counts(old_root: Path) -> dict[str, int]:
    path = old_root / "final_qa_category_counts_exact.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in (data.get("by_category") or {}).items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True, help="Existing final_qa_hardreject_dedup_* root")
    parser.add_argument("--new-generated-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-output-shards", type=int, default=64)
    args = parser.parse_args()

    started = time.time()
    old_final_root = args.old_root / "final_shards"
    out_final_root = args.output_root / "final_shards"
    out_aug_root = args.output_root / "accepted_cross_file_aug_shards"
    out_aug_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    old_keys = load_old_content_keys(old_final_root)
    handles = [
        (out_aug_root / f"cross_file_aug.accepted.shard_{idx:04d}_of_{args.num_output_shards:04d}.jsonl").open("w", encoding="utf-8")
        for idx in range(args.num_output_shards)
    ]
    counts: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    by_source_family: Counter[str] = Counter()
    seen_new: set[str] = set()
    try:
        for path in sorted(args.new_generated_root.glob("shards/*/cross_file_aug.generated.jsonl")):
            counts["input_files"] += 1
            for line_no, row, err in iter_jsonl(path):
                if err or row is None:
                    counts[f"hard_reject_{(err or 'invalid').split(':', 1)[0]}"] += 1
                    continue
                candidate, reason = canonical_aug_row(row, str(path), line_no)
                counts[f"reason_{reason}"] += 1
                if candidate is None:
                    continue
                key = str(candidate["content_key"])
                if key in old_keys:
                    counts["duplicate_against_old"] += 1
                    continue
                if key in seen_new:
                    counts["duplicate_within_new"] += 1
                    continue
                seen_new.add(key)
                clean = finalizer.strip_internal(candidate)
                shard = int(finalizer.stable_hash(key)[:12], 16) % args.num_output_shards
                handles[shard].write(json.dumps(clean, ensure_ascii=True, sort_keys=True) + "\n")
                counts["accepted_new"] += 1
                by_category[str(clean.get("task_category") or "unknown")] += 1
                by_source_family[str(clean.get("source_family") or "unknown")] += 1
    finally:
        for handle in handles:
            handle.close()

    linked = hardlink_old_shards(old_final_root, out_final_root)
    for path in sorted(out_aug_root.glob("cross_file_aug.accepted.shard_*.jsonl")):
        link_path = out_final_root / path.name
        if link_path.exists():
            continue
        os.link(path, link_path)
        linked += 1

    old_summary = load_old_summary(args.old_root)
    old_category = Counter(load_old_category_counts(args.old_root))
    old_source = Counter({str(k): int(v) for k, v in (old_summary.get("by_source_family") or {}).items()})
    old_rows = int(old_summary.get("counts", {}).get("rows_final") or old_summary.get("total_rows") or 0)
    combined_category = Counter(old_category)
    combined_category.update(by_category)
    combined_source = Counter(old_source)
    combined_source.update(by_source_family)
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - started, 3),
        "old_root": str(args.old_root),
        "new_generated_root": str(args.new_generated_root),
        "output_root": str(args.output_root),
        "old_rows": old_rows,
        "accepted_new_rows": int(counts["accepted_new"]),
        "combined_rows": old_rows + int(counts["accepted_new"]),
        "counts": dict(counts),
        "linked_files": linked,
        "new_by_source_family": dict(by_source_family),
        "combined_by_source_family": dict(combined_source),
        "new_by_category": dict(sorted(by_category.items())),
        "combined_by_category": dict(sorted(combined_category.items())),
        "top_new_categories": by_category.most_common(100),
        "top_combined_categories": combined_category.most_common(100),
    }
    (args.output_root / "final_qa_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_root / "final_qa_category_counts_exact.json").write_text(
        json.dumps(
            {
                "created_utc": summary["created_utc"],
                "total_rows": summary["combined_rows"],
                "num_categories": len(combined_category),
                "by_source_family": dict(combined_source),
                "by_category": dict(sorted(combined_category.items())),
                "top_categories": combined_category.most_common(200),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
