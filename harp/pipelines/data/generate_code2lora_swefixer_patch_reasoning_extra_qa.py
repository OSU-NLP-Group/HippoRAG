#!/usr/bin/env python3
"""Generate an extra deterministic SWE-fixer patch-reasoning QA wave.

This is a non-overlapping augmentation pass over the same SWE-fixer patch
metadata.  It asks new localization-heavy questions around triage, evidence,
source-test traceability, and repair planning instead of rerunning the first
patch-reasoning templates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import generate_code2lora_swefixer_patch_reasoning_qa as base


GENERATOR = "deterministic_swefixer_patch_reasoning_extra_qa_v1"
FAMILY_BUDGETS = {
    "localization": 0.50,
    "repair_planning": 0.18,
    "before_after": 0.12,
    "test_oracle": 0.09,
    "cross_file": 0.08,
    "contrastive": 0.03,
}


def stable_id(*parts: Any) -> str:
    return hashlib.sha1("\0".join(str(part) for part in parts).encode("utf-8", errors="replace")).hexdigest()


def compact(lines: Iterable[str], limit: int = 6) -> str:
    out = []
    for line in lines:
        text = str(line).strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return "\n".join(out)


def ctx(path: str, start: int, end: int, reason: str, evidence: str = "") -> dict[str, Any]:
    return base.ctx(path, start, end, reason, evidence)


def hctx(file_diff: base.FileDiff, hunk: base.Hunk, reason: str, evidence: str = "") -> dict[str, Any]:
    return base.hctx(file_diff, hunk, reason, evidence)


def make_row(
    src: dict[str, Any],
    category: str,
    question: str,
    answer: str,
    required_context: list[dict[str, Any]],
    answer_style: str,
) -> dict[str, Any] | None:
    question = question.strip()
    answer = answer.strip()
    if not question or not answer or not required_context:
        return None
    return {
        "id": stable_id(GENERATOR, src.get("instance_id", ""), src.get("base_commit", ""), category, question, answer),
        "repo_id": src.get("repo_id", ""),
        "base_commit": src.get("base_commit", ""),
        "commit_sha": src.get("commit_sha") or src.get("base_commit", ""),
        "instance_id": src.get("instance_id", ""),
        "source_dataset": src.get("source_dataset", ""),
        "qa_source": "swefixer_patch_reasoning_extra",
        "task_category": category,
        "question": question,
        "answer": answer,
        "answer_style": answer_style,
        "required_context": required_context,
        "generator": GENERATOR,
    }


def add(out: list[dict[str, Any]], seen: set[str], row: dict[str, Any], *args: Any) -> None:
    made = make_row(row, *args)
    if made is None:
        return
    rid = str(made["id"])
    if rid in seen:
        return
    seen.add(rid)
    out.append(made)


def category_family(category: str) -> str:
    if "test_oracle" in category:
        return "test_oracle"
    if category.startswith("cross_file") or "trace" in category or "source_test" in category:
        return "cross_file"
    if category.startswith("contrastive"):
        return "contrastive"
    if category.startswith("before_after") or "delta" in category:
        return "before_after"
    if category.startswith("repair_planning") or "triage" in category:
        return "repair_planning"
    return "localization"


def select_balanced(row: dict[str, Any], candidates: list[dict[str, Any]], rows_per_issue: int) -> list[dict[str, Any]]:
    if rows_per_issue <= 0 or len(candidates) <= rows_per_issue:
        return candidates[:rows_per_issue] if rows_per_issue > 0 else candidates
    buckets: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILY_BUDGETS}
    buckets["other"] = []
    for item in candidates:
        buckets.setdefault(category_family(str(item.get("task_category") or "")), []).append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: stable_id("extra-select", row.get("instance_id", ""), item["id"]))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for family, fraction in FAMILY_BUDGETS.items():
        want = max(1, int(rows_per_issue * fraction))
        for item in buckets.get(family, [])[:want]:
            selected.append(item)
            selected_ids.add(str(item["id"]))
            if len(selected) >= rows_per_issue:
                return selected
    remainder = [item for item in candidates if str(item["id"]) not in selected_ids]
    remainder.sort(key=lambda item: stable_id("extra-rem", row.get("instance_id", ""), item["id"]))
    selected.extend(remainder[: rows_per_issue - len(selected)])
    return selected[:rows_per_issue]


def path_role_summary(source_paths: list[str], test_paths: list[str], config_paths: list[str]) -> str:
    parts = []
    if source_paths:
        parts.append("Implementation:\n" + "\n".join(source_paths[:8]))
    if test_paths:
        parts.append("Tests:\n" + "\n".join(test_paths[:8]))
    if config_paths:
        parts.append("Configuration:\n" + "\n".join(config_paths[:4]))
    return "\n".join(parts)


def rows_for_issue(row: dict[str, Any], rows_per_issue: int) -> list[dict[str, Any]]:
    issue = base.issue_title(row)
    target_files = base.parse_diff(str(row.get("target_patch") or row.get("patch") or ""))
    test_files_diff = base.parse_diff(str(row.get("test_patch") or ""))
    all_files = target_files + test_files_diff
    source_paths = base.changed_paths(target_files, tests=False)
    test_paths = base.changed_paths(test_files_diff, tests=True)
    all_paths = base.changed_paths(all_files)
    config_paths = [path for path in all_paths if base.is_config_path(path)]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    paired_contexts = base.pair_contexts(source_paths, test_paths)
    if source_paths:
        add(
            out,
            seen,
            row,
            "triage_first_repair_file_with_reason",
            f"When starting to debug `{issue}`, which implementation file should be opened first and what patch evidence supports that choice?",
            f"Open `{source_paths[0]}` first; it is the first implementation-side file changed by the SWE-fixer patch.",
            [ctx(source_paths[0], 1, 1, "first implementation-side changed file")],
            "explanatory_path",
        )
        add(
            out,
            seen,
            row,
            "localization_all_implementation_repair_files",
            f"List the implementation-side files that define the repair surface for `{issue}`.",
            "\n".join(source_paths[:12]),
            [ctx(path, 1, 1, "implementation-side repair surface") for path in source_paths[:8]],
            "list",
        )
    if test_paths:
        add(
            out,
            seen,
            row,
            "test_oracle_first_file_with_reason",
            f"When validating the fix for `{issue}`, which test file gives the first regression signal and why?",
            f"Inspect `{test_paths[0]}`; it is the first test-side file changed by the SWE-fixer patch.",
            [ctx(test_paths[0], 1, 1, "first test-side changed file")],
            "explanatory_path",
        )
    if all_paths:
        add(
            out,
            seen,
            row,
            "localization_changed_path_role_map",
            f"For `{issue}`, how do the changed paths divide into implementation, tests, and configuration?",
            path_role_summary(source_paths, test_paths, config_paths),
            [ctx(path, 1, 1, "changed path used for role map") for path in all_paths[:10]],
            "grouped_list",
        )
    if source_paths and test_paths:
        add(
            out,
            seen,
            row,
            "cross_file_source_test_trace",
            f"What source-test trace should be followed to understand the repair for `{issue}`?",
            f"Start from implementation `{source_paths[0]}` and validate against test `{test_paths[0]}`.",
            paired_contexts,
            "source_test_trace",
        )
        add(
            out,
            seen,
            row,
            "contrastive_source_vs_oracle_file",
            f"For `{issue}`, which file is repair implementation rather than regression oracle: `{source_paths[0]}` or `{test_paths[0]}`?",
            f"`{source_paths[0]}` is the repair implementation file; `{test_paths[0]}` is test-oracle evidence.",
            paired_contexts,
            "contrastive_explanation",
        )

    for file_diff in all_files:
        role = base.likely_role(file_diff.new_path)
        for hunk in file_diff.hunks:
            plus = base.added_lines(hunk)
            minus = base.removed_lines(hunk)
            ctx_lines = base.context_lines(hunk)
            evidence = compact(plus, 8) or compact(minus, 8) or compact(ctx_lines, 6)
            span = f"{file_diff.new_path}:{hunk.new_start}-{hunk.new_start + max(0, hunk.new_len - 1)}"
            hc = [hctx(file_diff, hunk, "patch hunk evidence for extra reasoning QA", evidence)]
            add(out, seen, row, "localization_patch_hunk_to_open", f"Which exact patch hunk should be opened to inspect evidence for `{issue}` in `{file_diff.new_path}`?", span, hc, "location")
            add(out, seen, row, "repair_planning_hunk_evidence_summary", f"What local code evidence at `{span}` should guide a repair plan for `{issue}`?", evidence, hc, "code_evidence")
            if plus:
                add(out, seen, row, "localization_added_code_span", f"Where does the patch add code for `{issue}` in `{file_diff.new_path}`?", span, hc, "location")
                add(out, seen, row, "repair_planning_added_code_to_review", f"What added code at `{span}` should be reviewed when explaining the fix for `{issue}`?", compact(plus, 8), hc, "code")
            if minus:
                add(out, seen, row, "localization_removed_code_span", f"Where does the patch remove or replace code for `{issue}` in `{file_diff.new_path}`?", span, hc, "location")
            if plus and minus:
                add(out, seen, row, "before_after_line_delta_extra", f"At `{span}`, what line-level before/after delta does the patch make for `{issue}`?", "Before:\n" + compact(minus, 6) + "\nAfter:\n" + compact(plus, 6), hc, "diff_summary")
            syms = base.symbol_names(plus + minus + ctx_lines)
            for symbol in syms[:3]:
                cat = "localization_implementation_symbol_from_hunk" if role == "implementation" else "test_oracle_symbol_from_hunk_extra"
                q = (
                    f"Which implementation symbol should be inspected at `{span}` for `{issue}`?"
                    if role == "implementation"
                    else f"Which test-side symbol at `{span}` records expected behavior for `{issue}`?"
                )
                add(out, seen, row, cat, q, symbol, hc, "symbolic")
            conditions = [line.strip() for line in plus if base.CONDITION_LINE_RE.match(line)]
            if conditions:
                add(out, seen, row, "before_after_added_branch_extra", f"Which added branch or guard at `{span}` changes the repair behavior for `{issue}`?", compact(conditions, 6), hc, "list")
            assertions = [line.strip() for line in plus if base.ASSERT_LINE_RE.search(line)]
            if assertions:
                add(out, seen, row, "test_oracle_expected_behavior_extra", f"What added assertion at `{span}` captures expected behavior for `{issue}`?", compact(assertions, 6), hc, "list")
            imports = [line.strip() for line in plus if base.IMPORT_LINE_RE.match(line)]
            if imports:
                add(out, seen, row, "cross_file_dependency_trace_extra", f"What newly imported dependency at `{span}` should be traced while understanding `{issue}`?", compact(imports, 6), hc, "list")
            call_candidates = base.calls(plus)
            if call_candidates:
                add(out, seen, row, "cross_file_api_trace_extra", f"Which added call or API at `{span}` may connect this edit to other repository code for `{issue}`?", "\n".join(call_candidates[:6]), hc, "list")
            if role == "implementation" and test_paths:
                add(out, seen, row, "cross_file_impl_hunk_to_test_file_extra", f"After inspecting implementation hunk `{span}` for `{issue}`, which test file should be read next?", test_paths[0], hc + [ctx(test_paths[0], 1, 1, "test-side changed file paired with implementation hunk")], "path")
            if role == "test" and source_paths:
                add(out, seen, row, "cross_file_test_hunk_to_impl_file_extra", f"After reading test hunk `{span}` for `{issue}`, which implementation file should be inspected next?", source_paths[0], hc + [ctx(source_paths[0], 1, 1, "implementation-side changed file paired with test hunk")], "path")

    return select_balanced(row, out, rows_per_issue)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def load_seen(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    for row in iter_jsonl(path):
        rid = str(row.get("id") or "")
        if rid:
            seen.add(rid)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-metadata", type=Path, default=Path("/path/to/ad-local/storage/issuefix_swefixer/metadata/swefixer_full.parquet"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=24)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--target-rows", type=int, default=84000)
    parser.add_argument("--rows-per-issue", type=int, default=48)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(args.output_jsonl) if args.resume else set()
    generated = len(seen)
    counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    started = time.time()
    mode = "a" if args.resume else "w"

    with args.output_jsonl.open(mode, encoding="utf-8") as out_fh:
        for row in base.iter_patch_rows(args.input_metadata):
            row_index = int(row.get("row_index") or 0)
            if args.num_shards > 1 and row_index % args.num_shards != args.shard_index:
                continue
            if row.get("usable_for_train") is False:
                counts["unusable_skipped"] += 1
                continue
            if args.target_rows and generated >= args.target_rows:
                break
            counts["issues_seen"] += 1
            for item in rows_for_issue(row, args.rows_per_issue):
                rid = str(item["id"])
                if rid in seen:
                    counts["duplicate_skipped"] += 1
                    continue
                seen.add(rid)
                out_fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                generated += 1
                counts["generated"] += 1
                category_counts[str(item.get("task_category") or "unknown")] += 1
                if args.target_rows and generated >= args.target_rows:
                    break
            if args.progress_every and counts["issues_seen"] % args.progress_every == 0:
                print(json.dumps({"issues_seen": counts["issues_seen"], "generated": generated, "elapsed_sec": round(time.time() - started, 1)}, sort_keys=True), flush=True)

    audit = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "elapsed_sec": round(time.time() - started, 3),
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
