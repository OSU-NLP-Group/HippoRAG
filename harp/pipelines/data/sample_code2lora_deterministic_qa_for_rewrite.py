#!/usr/bin/env python3
"""Sample deterministic Code2LoRA QA rows for LLM rewriting.

The sampler is streaming and deterministic: a row is selected when a stable hash
of its task category and id falls below the requested fraction. This gives an
approximately 10% sample within every task_category without loading the full
7M+ row corpus into memory.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set


DEFAULT_INPUT_GLOBS = [
    "/path/to/ad-local/storage/code2lora_jina_enriched_qa/enriched_qa_only_full_20260711/"
    "ast_qa_generated_v1_20260715/shards/*/ast_qa.generated.jsonl",
    "/path/to/ad-local/storage/code2lora_jina_enriched_qa/enriched_qa_only_full_20260711/"
    "ast_qa_generated_v1_20260715_supplement_skip40/shards/*/ast_qa.generated.jsonl",
    "/path/to/ad-local/storage/code2lora_jina_enriched_qa/enriched_qa_only_full_20260711/"
    "swefixer_patch_qa_v1_20260716/shards/*/swefixer_deterministic_qa.generated.jsonl",
    "/path/to/ad-local/storage/code2lora_jina_enriched_qa/enriched_qa_only_full_20260711/"
    "swefixer_repo_static_qa_v1_20260716/shards/*/swefixer_deterministic_qa.generated.jsonl",
    "/path/to/ad-local/storage/code2lora_jina_enriched_qa/enriched_qa_only_full_20260711/"
    "deterministic_expansion_qa_v1_12m_20260717/shards/*/deterministic_expansion_qa.generated.jsonl",
    "/path/to/ad-local/storage/code2lora_jina_enriched_qa/enriched_qa_only_full_20260711/"
    "diverse_repo_coverage_qa_v1_10m_20260717/shards/*/diverse_repo_coverage_qa.generated.jsonl",
    "/path/to/ad-local/storage/code2lora_jina_enriched_qa/enriched_qa_only_full_20260711/"
    "deep_surface_qa_v1_20260718/*/shards/*/deep_surface_qa.generated.jsonl",
]


def expand_inputs(patterns: List[str]) -> List[Path]:
    paths = []  # type: List[Path]
    for pattern in patterns:
        matches = sorted(Path("/").glob(pattern.lstrip("/")))
        paths.extend(path for path in matches if path.is_file())
    return sorted(dict.fromkeys(paths))


def stable_int(*parts: str) -> int:
    key = "\0".join(parts).encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.sha1(key).digest()[:8], "big", signed=False)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                yield row


def row_id(row: dict) -> str:
    value = row.get("id") or row.get("qa_id")
    if value:
        return str(value)
    fallback = json.dumps(
        {
            "repo_id": row.get("repo_id"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "task_category": row.get("task_category"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(fallback.encode("utf-8", errors="replace")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", action="append", default=[], help="JSONL glob; may be repeated")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-numerator", type=int, default=10)
    parser.add_argument("--sample-denominator", type=int, default=100)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Lower bound of the selected hash bucket range. For disjoint 10%% bands, use 0, 10, 20, ... with denominator 100.",
    )
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--sample-version", default="deterministic_qa_rewrite_10pct_v1")
    args = parser.parse_args()

    if args.sample_numerator <= 0 or args.sample_denominator <= 0:
        raise SystemExit("sample numerator and denominator must be positive")
    if args.sample_numerator > args.sample_denominator:
        raise SystemExit("sample numerator cannot exceed denominator")
    if args.sample_offset < 0 or args.sample_offset >= args.sample_denominator:
        raise SystemExit("sample offset must be in [0, sample_denominator)")
    if args.sample_offset + args.sample_numerator > args.sample_denominator:
        raise SystemExit("sample offset + numerator cannot exceed sample denominator")

    patterns = args.input_glob or DEFAULT_INPUT_GLOBS
    input_paths = expand_inputs(patterns)
    if not input_paths:
        raise SystemExit("no input JSONL files matched")

    args.output_root.mkdir(parents=True, exist_ok=True)
    shards_dir = args.output_root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [
        shards_dir / f"rewrite_input.shard_{idx:02d}_of_{args.num_shards:02d}.jsonl"
        for idx in range(args.num_shards)
    ]
    tmp_paths = [path.with_suffix(path.suffix + ".tmp") for path in shard_paths]

    total_by_category: Counter[str] = Counter()
    selected_by_category: Counter[str] = Counter()
    total_by_source: Counter[str] = Counter()
    selected_by_source: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    seen_selected = set()  # type: Set[str]
    started = time.time()

    handles = [path.open("w", encoding="utf-8") for path in tmp_paths]
    try:
        for input_path in input_paths:
            source_label = input_path.parts[-4] if len(input_path.parts) >= 4 else input_path.parent.name
            for row in iter_jsonl(input_path):
                counts["rows_seen"] += 1
                category = str(row.get("task_category") or "unknown")
                rid = row_id(row)
                total_by_category[category] += 1
                total_by_source[source_label] += 1
                bucket = stable_int(category, rid) % args.sample_denominator
                if not (args.sample_offset <= bucket < args.sample_offset + args.sample_numerator):
                    continue
                if rid in seen_selected:
                    counts["duplicate_selected_skipped"] += 1
                    continue
                seen_selected.add(rid)
                selected_by_category[category] += 1
                selected_by_source[source_label] += 1
                counts["rows_selected"] += 1
                shard_idx = stable_int("shard", category, rid) % args.num_shards
                out_row = dict(row)
                out_row["rewrite_input_id"] = rid
                out_row["rewrite_sample_version"] = args.sample_version
                out_row["rewrite_sample_fraction"] = {
                    "numerator": args.sample_numerator,
                    "denominator": args.sample_denominator,
                    "offset": args.sample_offset,
                }
                out_row["rewrite_source_file"] = str(input_path)
                handles[shard_idx].write(json.dumps(out_row, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        for handle in handles:
            handle.close()

    for tmp_path, final_path in zip(tmp_paths, shard_paths):
        tmp_path.replace(final_path)

    audit = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - started, 3),
        "input_files": [str(path) for path in input_paths],
        "output_shards": [str(path) for path in shard_paths],
        "sample_fraction": {
            "numerator": args.sample_numerator,
            "denominator": args.sample_denominator,
            "offset": args.sample_offset,
        },
        "counts": dict(counts),
        "total_by_category": dict(sorted(total_by_category.items())),
        "selected_by_category": dict(sorted(selected_by_category.items())),
        "total_by_source": dict(sorted(total_by_source.items())),
        "selected_by_source": dict(sorted(selected_by_source.items())),
    }
    audit_path = args.output_root / "rewrite_sample.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
