#!/usr/bin/env python3
"""Generate deterministic cross-file QA augmentations from accepted final QA.

The source rows are already validated QA examples.  This script only uses rows
with required_context spanning at least two distinct files, then creates new
questions that teach path-to-path relationships, source/test pairings, and
multi-file evidence navigation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


GENERATOR = "deterministic_cross_file_aug_from_final_qa_v1"
TEST_RE = re.compile(r"(^|/)(tests?|testing|test)/|(^|/|_)test_[^/]*\.py$|(^|/)test[^/]*\.py$")
CROSS_HINT_RE = re.compile(
    r"(cross|multi|import|call|definition|symbol|test|implementation|source|patch|hunk|repair|localization|oracle)",
    re.I,
)


def stable_id(*parts: Any) -> str:
    return hashlib.sha1("\0".join(str(part) for part in parts).encode("utf-8", errors="replace")).hexdigest()


def short(text: Any, limit: int = 220) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    clipped = normalized[:limit].rsplit(" ", 1)[0] or normalized[:limit]
    return clipped.rstrip(" ,;:") + "..."


def is_test_path(path: str) -> bool:
    return bool(TEST_RE.search(path))


def paths_from_context(spans: Any) -> list[str]:
    out = []
    if not isinstance(spans, list):
        return out
    for span in spans:
        if not isinstance(span, dict):
            continue
        path = str(span.get("path") or "").strip()
        if path and path not in out:
            out.append(path)
    return out


def context_for_paths(spans: list[dict[str, Any]], paths: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(paths)
    out = []
    seen = set()
    for span in spans:
        path = str(span.get("path") or "")
        if path in wanted and path not in seen:
            copied = dict(span)
            copied.setdefault("reason", "cross-file evidence from source QA")
            out.append(copied)
            seen.add(path)
    return out


def row_source_id(row: dict[str, Any]) -> str:
    return str(row.get("source_id") or row.get("id") or stable_id(row.get("repo_id"), row.get("question"), row.get("answer")))


def should_use(row: dict[str, Any], paths: list[str]) -> bool:
    if len(paths) < 2:
        return False
    category = str(row.get("task_category") or "")
    question = str(row.get("question") or "")
    source_family = str(row.get("source_family") or "")
    if source_family != "deterministic_original":
        return False
    if CROSS_HINT_RE.search(category) or CROSS_HINT_RE.search(question):
        return True
    return False


def make_row(
    src: dict[str, Any],
    category: str,
    question: str,
    answer: str,
    required_context: list[dict[str, Any]],
    answer_style: str,
) -> dict[str, Any] | None:
    question = question.strip()
    if question and question[-1] not in "?!":
        question += "?"
    answer = answer.strip()
    if not question or not answer or not required_context:
        return None
    source_id = row_source_id(src)
    rid = stable_id(GENERATOR, source_id, category, question, answer)
    return {
        "id": rid,
        "repo_id": src.get("repo_id", ""),
        "base_commit": src.get("base_commit") or src.get("commit_sha") or "",
        "commit_sha": src.get("commit_sha") or src.get("base_commit") or "",
        "question": question,
        "answer": answer,
        "answer_style": answer_style,
        "task_category": category,
        "qa_source": "deterministic_cross_file_aug_from_final",
        "source_family": "deterministic_cross_file_aug",
        "required_context": required_context,
        "generator": GENERATOR,
        "source_final_qa_id": source_id,
        "source_final_task_category": src.get("task_category", ""),
        "source_final_question": src.get("question", ""),
        "source_final_answer": src.get("answer", ""),
    }


def rows_for_source(src: dict[str, Any], rows_per_source: int) -> list[dict[str, Any]]:
    spans = src.get("required_context") or []
    if not isinstance(spans, list):
        return []
    paths = paths_from_context(spans)
    if not should_use(src, paths):
        return []
    test_paths = [path for path in paths if is_test_path(path)]
    impl_paths = [path for path in paths if not is_test_path(path)]
    first = paths[0]
    second = paths[1]
    source_question = short(src.get("question"), 180)
    source_answer = short(src.get("answer"), 180)
    out: list[dict[str, Any]] = []

    def add(category: str, question: str, answer: str, ctx_paths: list[str], answer_style: str) -> None:
        if len(out) >= rows_per_source:
            return
        row = make_row(src, category, question, answer, context_for_paths(spans, ctx_paths), answer_style)
        if row is not None:
            out.append(row)

    add(
        "cross_file_evidence_paths_for_question",
        f"Which repository files should be read together to answer: {source_question}",
        "\n".join(paths[:6]),
        paths[:6],
        "path_list",
    )
    add(
        "cross_file_answer_grounding_paths",
        f"Across which files is the answer `{source_answer}` grounded?",
        "\n".join(paths[:6]),
        paths[:6],
        "path_list",
    )
    add(
        "cross_file_primary_secondary_context",
        f"What are the primary and secondary files linked by this repository fact: {source_question}",
        f"Primary context: {first}\nSecondary context: {second}",
        [first, second],
        "source_target_pair",
    )
    add(
        "cross_file_context_relation_summary",
        f"What cross-file relation should be followed between `{first}` and `{second}`?",
        f"Read `{first}` together with `{second}`; the original QA asks: {source_question}",
        [first, second],
        "explanatory",
    )
    if impl_paths and test_paths:
        add(
            "cross_file_test_to_implementation_pair_extra",
            f"Which implementation file should be paired with test file `{test_paths[0]}` for this repo-specific fact?",
            impl_paths[0],
            [test_paths[0], impl_paths[0]],
            "path",
        )
        add(
            "cross_file_implementation_to_test_pair_extra",
            f"Which test file should be read with implementation file `{impl_paths[0]}` for this repo-specific fact?",
            test_paths[0],
            [impl_paths[0], test_paths[0]],
            "path",
        )
        add(
            "cross_file_source_test_navigation_extra",
            f"What source-test navigation path is implied by this question: {source_question}",
            f"Implementation: {impl_paths[0]}\nTest: {test_paths[0]}",
            [impl_paths[0], test_paths[0]],
            "source_test_pair",
        )
    if len(paths) >= 3:
        add(
            "cross_file_three_way_context_extra",
            f"Which three files form the multi-file context for this repository fact?",
            "\n".join(paths[:3]),
            paths[:3],
            "path_list",
        )
    category = str(src.get("task_category") or "unknown")
    if any(token in category for token in ("import", "call", "definition", "symbol")):
        add(
            "cross_file_symbol_or_import_trace_extra",
            f"Which files participate in the symbol/import trace behind: {source_question}",
            "\n".join(paths[:4]),
            paths[:4],
            "path_list",
        )
    if any(token in category for token in ("patch", "hunk", "repair", "localization", "oracle")):
        add(
            "cross_file_patch_or_repair_trace_extra",
            f"Which files should be read as the repair or localization trace for: {source_question}",
            "\n".join(paths[:4]),
            paths[:4],
            "path_list",
        )
    return out[:rows_per_source]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield line_no, row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--target-rows", type=int, default=94000)
    parser.add_argument("--rows-per-source", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=250000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    seen = set()
    generated = 0
    if args.resume and args.output_jsonl.exists():
        for _line_no, row in iter_jsonl(args.output_jsonl):
            rid = str(row.get("id") or "")
            if rid:
                seen.add(rid)
        generated = len(seen)

    counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    started = time.time()
    input_files = sorted(args.input_root.glob("final_qa.shard_*_of_*.jsonl"))
    with args.output_jsonl.open(mode, encoding="utf-8") as out_fh:
        for file_index, path in enumerate(input_files):
            if file_index % args.num_shards != args.shard_index:
                continue
            counts["input_files_seen"] += 1
            for _line_no, src in iter_jsonl(path):
                counts["source_rows_seen"] += 1
                if generated >= args.target_rows:
                    break
                rows = rows_for_source(src, args.rows_per_source)
                if not rows:
                    counts["source_rows_skipped"] += 1
                    continue
                for row in rows:
                    rid = str(row["id"])
                    if rid in seen:
                        counts["duplicate_generated_skipped"] += 1
                        continue
                    seen.add(rid)
                    out_fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                    generated += 1
                    counts["generated"] += 1
                    category_counts[str(row.get("task_category") or "unknown")] += 1
                    if generated >= args.target_rows:
                        break
                if args.progress_every and counts["source_rows_seen"] % args.progress_every == 0:
                    print(json.dumps({"generated": generated, "source_rows_seen": counts["source_rows_seen"], "elapsed_sec": round(time.time() - started, 1)}, sort_keys=True), flush=True)
            if generated >= args.target_rows:
                break
    audit = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - started, 3),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "generated_total_in_output": generated,
        "counts": dict(counts),
        "category_counts": dict(category_counts),
        "generator": GENERATOR,
    }
    tmp = args.audit_output.with_suffix(args.audit_output.suffix + ".tmp")
    tmp.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.audit_output)
    print(json.dumps({"generated": generated, "audit": str(args.audit_output)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
