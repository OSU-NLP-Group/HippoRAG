#!/usr/bin/env python3
"""Generate diverse deterministic QA from SWE-fixer patch data.

This wave focuses on patch reasoning rather than restating patch facts.  It
intentionally over-samples localization questions while adding before/after
semantic deltas, test-oracle links, cross-file causality, and contrastive
source/test/file-role questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


GENERATOR = "deterministic_swefixer_patch_reasoning_qa_v2"
FAMILY_BUDGETS = {
    "localization": 0.45,
    "before_after": 0.22,
    "test_oracle": 0.14,
    "cross_file": 0.10,
    "contrastive": 0.06,
    "repair_planning": 0.03,
}
DIFF_FILE_RE = re.compile(r"^diff --git a/(.*?) b/(.*)$")
HUNK_RE = re.compile(r"^@@ -(?P<old_start>\d+)(?:,(?P<old_len>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_len>\d+))? @@(?P<header>.*)$")
TEST_RE = re.compile(r"(^|/)(tests?|testing|test)/|(^|/|_)test_[^/]*\.py$|(^|/)test[^/]*\.py$")
SYMBOL_LINE_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
TEST_DEF_RE = re.compile(r"^\s*(?:async\s+def|def)\s+(test_[A-Za-z0-9_]+)")
IMPORT_LINE_RE = re.compile(r"^\s*(?:from\s+[\w.]+\s+import\s+.+|import\s+.+)")
CONDITION_LINE_RE = re.compile(r"^\s*(?:if|elif|while|for)\b")
ASSERT_LINE_RE = re.compile(r"\b(?:assert|pytest\.raises|self\.assert\w+|assertRaises|assertEqual|assertTrue|assertFalse)\b")
RETURN_LINE_RE = re.compile(r"^\s*return\b")
EXCEPTION_LINE_RE = re.compile(r"\b(?:raise|except)\b")
CALL_LINE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*\(")
CONFIG_PATH_RE = re.compile(r"(^|/)(pyproject\.toml|setup\.cfg|setup\.py|tox\.ini|pytest\.ini|mypy\.ini|ruff\.toml|.*\.(?:toml|cfg|ini|ya?ml|json))$")


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    header: str
    lines: list[str]


@dataclass
class FileDiff:
    old_path: str
    new_path: str
    hunks: list[Hunk]


def stable_id(*parts: Any) -> str:
    return hashlib.sha1("\0".join(str(part) for part in parts).encode("utf-8", errors="replace")).hexdigest()


def short(text: Any, limit: int = 240) -> str:
    return " ".join(str(text or "").split())[:limit].rstrip()


def compact(lines: Iterable[str], limit: int = 8) -> str:
    out = []
    for line in lines:
        text = str(line).strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return "\n".join(out)


def parse_diff(diff_text: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    current: FileDiff | None = None
    current_hunk: Hunk | None = None
    for line in str(diff_text or "").splitlines():
        match = DIFF_FILE_RE.match(line)
        if match:
            if current is not None:
                files.append(current)
            current = FileDiff(match.group(1), match.group(2), [])
            current_hunk = None
            continue
        if current is None:
            continue
        hunk_match = HUNK_RE.match(line)
        if hunk_match:
            current_hunk = Hunk(
                int(hunk_match.group("old_start")),
                int(hunk_match.group("old_len") or "1"),
                int(hunk_match.group("new_start")),
                int(hunk_match.group("new_len") or "1"),
                hunk_match.group("header").strip(),
                [],
            )
            current.hunks.append(current_hunk)
            continue
        if current_hunk is not None:
            current_hunk.lines.append(line)
    if current is not None:
        files.append(current)
    return files


def added_lines(hunk: Hunk) -> list[str]:
    return [line[1:] for line in hunk.lines if line.startswith("+") and not line.startswith("+++")]


def removed_lines(hunk: Hunk) -> list[str]:
    return [line[1:] for line in hunk.lines if line.startswith("-") and not line.startswith("---")]


def context_lines(hunk: Hunk) -> list[str]:
    return [line[1:] if line[:1] in {" ", "+", "-"} else line for line in hunk.lines if line.startswith(" ")]


def is_test_path(path: str) -> bool:
    return bool(TEST_RE.search(path))


def is_config_path(path: str) -> bool:
    return bool(CONFIG_PATH_RE.search(path))


def changed_paths(files: list[FileDiff], tests: bool | None = None) -> list[str]:
    paths = []
    for file_diff in files:
        path = file_diff.new_path
        if tests is True and not is_test_path(path):
            continue
        if tests is False and is_test_path(path):
            continue
        paths.append(path)
    return sorted(dict.fromkeys(paths))


def issue_title(row: dict[str, Any]) -> str:
    lines = [line.strip() for line in str(row.get("problem_statement") or "").splitlines() if line.strip()]
    return short(lines[0] if lines else row.get("instance_id") or "this issue", 180)


def ctx(path: str, start: int, end: int, reason: str, evidence: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": path,
        "start_line": max(1, int(start or 1)),
        "end_line": max(1, int(end or start or 1)),
        "reason": reason,
    }
    if evidence:
        out["evidence"] = evidence[:2000]
    return out


def hctx(file_diff: FileDiff, hunk: Hunk, reason: str, evidence: str = "") -> dict[str, Any]:
    return ctx(file_diff.new_path, hunk.new_start, hunk.new_start + max(0, hunk.new_len - 1), reason, evidence)


def make_row(
    src: dict[str, Any],
    category: str,
    question: str,
    answer: str,
    required_context: list[dict[str, Any]],
    answer_style: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    question = question.strip()
    answer = answer.strip()
    if not question or not answer or not required_context:
        return None
    payload: dict[str, Any] = {
        "id": stable_id(GENERATOR, src.get("instance_id", ""), src.get("base_commit", ""), category, question, answer),
        "repo_id": src.get("repo_id", ""),
        "base_commit": src.get("base_commit", ""),
        "commit_sha": src.get("commit_sha") or src.get("base_commit", ""),
        "instance_id": src.get("instance_id", ""),
        "source_dataset": src.get("source_dataset", ""),
        "qa_source": "swefixer_patch_reasoning",
        "task_category": category,
        "question": question,
        "answer": answer,
        "answer_style": answer_style,
        "required_context": required_context,
        "generator": GENERATOR,
    }
    if extra:
        payload.update(extra)
    return payload


def likely_role(path: str) -> str:
    if is_test_path(path):
        return "test"
    if is_config_path(path):
        return "configuration"
    return "implementation"


def symbol_names(lines: Iterable[str]) -> list[str]:
    out = []
    for line in lines:
        match = SYMBOL_LINE_RE.match(line)
        if match:
            out.append(match.group(1))
    return sorted(dict.fromkeys(out))


def test_names(lines: Iterable[str]) -> list[str]:
    out = []
    for line in lines:
        match = TEST_DEF_RE.match(line)
        if match:
            out.append(match.group(1))
    return sorted(dict.fromkeys(out))


def calls(lines: Iterable[str]) -> list[str]:
    out = []
    for line in lines:
        for match in CALL_LINE_RE.finditer(line):
            name = match.group(1)
            if name not in {"if", "for", "while", "return", "assert"}:
                out.append(name)
    return sorted(dict.fromkeys(out))


def pair_contexts(source_paths: list[str], test_paths: list[str]) -> list[dict[str, Any]]:
    contexts = [ctx(path, 1, 1, "implementation file changed by SWE-fixer patch") for path in source_paths[:4]]
    contexts.extend(ctx(path, 1, 1, "test file changed by SWE-fixer patch") for path in test_paths[:4])
    return contexts


def add(out: list[dict[str, Any]], seen: set[str], row: dict[str, Any], *args: Any, **kwargs: Any) -> None:
    made = make_row(row, *args, **kwargs)
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
    if category.startswith("cross_file") or "source_to_test" in category or "source_edit_to_test" in category:
        return "cross_file"
    if category.startswith("contrastive"):
        return "contrastive"
    if category.startswith("before_after") or "semantic_delta" in category or "behavior_change" in category:
        return "before_after"
    if category.startswith("repair_planning"):
        return "repair_planning"
    return "localization"


def rank_for_selection(row: dict[str, Any], item: dict[str, Any]) -> str:
    return stable_id(
        "select",
        row.get("instance_id", ""),
        row.get("base_commit", ""),
        item.get("task_category", ""),
        item.get("question", ""),
    )


def select_balanced_rows(row: dict[str, Any], candidates: list[dict[str, Any]], rows_per_issue: int) -> list[dict[str, Any]]:
    if rows_per_issue <= 0 or len(candidates) <= rows_per_issue:
        return candidates[:rows_per_issue] if rows_per_issue > 0 else candidates
    buckets: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILY_BUDGETS}
    buckets["other"] = []
    for item in candidates:
        buckets.setdefault(category_family(str(item.get("task_category") or "")), []).append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: rank_for_selection(row, item))

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
    remainder.sort(key=lambda item: rank_for_selection(row, item))
    selected.extend(remainder[: rows_per_issue - len(selected)])
    return selected[:rows_per_issue]


def rows_for_issue(row: dict[str, Any], rows_per_issue: int) -> list[dict[str, Any]]:
    issue = issue_title(row)
    target_files = parse_diff(str(row.get("target_patch") or row.get("patch") or ""))
    test_files_diff = parse_diff(str(row.get("test_patch") or ""))
    all_files = target_files + test_files_diff
    source_paths = changed_paths(target_files, tests=False)
    test_paths = changed_paths(test_files_diff, tests=True)
    all_paths = changed_paths(all_files)
    config_paths = [path for path in all_paths if is_config_path(path)]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    issue_ref = f"SWE-fixer instance `{row.get('instance_id')}`"

    # Localization: deliberately broad and plentiful.
    if source_paths:
        add(out, seen, row, "patch_localization_implementation_files", f"Which implementation files are likely repair targets for `{issue}`?", "\n".join(source_paths[:12]), pair_contexts(source_paths, test_paths), "list")
        add(out, seen, row, "repair_localization_primary_source_file", f"What is the first implementation file to inspect when repairing `{issue}`?", source_paths[0], [ctx(source_paths[0], 1, 1, "first implementation path in patch")], "path")
        add(out, seen, row, "repair_localization_source_file_count", f"How many implementation files does the fix for `{issue}` touch?", str(len(source_paths)), [ctx(path, 1, 1, "implementation file changed by patch") for path in source_paths[:8]], "count")
    if test_paths:
        add(out, seen, row, "patch_localization_test_files", f"Which regression-test files localize the expected behavior for `{issue}`?", "\n".join(test_paths[:12]), [ctx(path, 1, 1, "test file changed by patch") for path in test_paths[:8]], "list")
        add(out, seen, row, "repair_localization_primary_test_file", f"What is the first test file to inspect for the regression signal in `{issue}`?", test_paths[0], [ctx(test_paths[0], 1, 1, "first test path in patch")], "path")
    if source_paths and test_paths:
        add(out, seen, row, "repair_localization_source_test_pair", f"For `{issue}`, what implementation/test file pair best localizes the repair?", f"Implementation: {source_paths[0]}\nTest: {test_paths[0]}", pair_contexts(source_paths, test_paths), "source_test_pair")
        add(out, seen, row, "patch_causality_source_to_test_files", f"For `{issue}`, which implementation files and tests should be read together to understand the fix?", "Implementation files:\n" + "\n".join(source_paths[:8]) + "\nTest files:\n" + "\n".join(test_paths[:8]), pair_contexts(source_paths, test_paths), "list")
        add(out, seen, row, "contrastive_repair_file_not_test_file", f"For `{issue}`, which changed file is implementation-side rather than test-side?", source_paths[0], pair_contexts(source_paths, test_paths), "path")
        add(out, seen, row, "contrastive_test_file_not_repair_file", f"For `{issue}`, which changed file is test-side rather than the implementation repair?", test_paths[0], pair_contexts(source_paths, test_paths), "path")
    if config_paths:
        add(out, seen, row, "repair_localization_config_file", f"Which configuration file is implicated by the patch for `{issue}`?", "\n".join(config_paths[:8]), [ctx(path, 1, 1, "configuration file changed by patch") for path in config_paths[:8]], "list")

    for file_diff in all_files:
        role = likely_role(file_diff.new_path)
        add(out, seen, row, "changed_file_role_localization", f"In the fix for `{issue}`, what role does `{file_diff.new_path}` play?", role, [ctx(file_diff.new_path, 1, 1, "changed path role inferred from path")], "label")
        add(out, seen, row, "repair_localization_changed_file_reason", f"Why should `{file_diff.new_path}` be inspected for `{issue}`?", f"It is a changed {role} file in the SWE-fixer patch.", [ctx(file_diff.new_path, 1, 1, "file appears in patch")], "explanatory")
        for hunk_index, hunk in enumerate(file_diff.hunks):
            plus = added_lines(hunk)
            minus = removed_lines(hunk)
            ctx_lines = context_lines(hunk)
            added = compact(plus, 10)
            removed = compact(minus, 10)
            span = f"{file_diff.new_path}:{hunk.new_start}-{hunk.new_start + max(0, hunk.new_len - 1)}"
            evidence = added or removed or compact(ctx_lines, 8)
            hc = [hctx(file_diff, hunk, "patch hunk evidence", evidence)]
            add(out, seen, row, "hunk_localization_repair_span", f"Which hunk span localizes a concrete edit for `{issue}` in `{file_diff.new_path}`?", span, hc, "location")
            if hunk.header:
                add(out, seen, row, "hunk_localization_header", f"What hunk header localizes the edit for `{issue}` at `{span}`?", hunk.header, hc, "symbol_or_header")
            add(out, seen, row, "repair_planning_context_span", f"What context span should be included when planning the repair for `{issue}` in `{file_diff.new_path}`?", span, hc, "location")
            if added:
                add(out, seen, row, "repair_planning_added_code_evidence", f"What added code evidence should guide the repair plan for `{issue}` at `{span}`?", added, hc, "code")
            if removed and added:
                add(out, seen, row, "before_after_semantic_delta_hunk", f"What before/after code change is made for `{issue}` at `{span}`?", "Before:\n" + removed + "\nAfter:\n" + added, hc, "diff_summary")
                add(out, seen, row, "patch_intent_behavior_changed_at_hunk", f"What behavior appears to change for `{issue}` at `{span}`?", "The patch replaces prior behavior with:\n" + added, hc, "explanatory")
            syms = symbol_names(plus + minus + ctx_lines)
            for symbol in syms[:4]:
                add(out, seen, row, "symbol_localization_from_patch_hunk", f"Which symbol is localized by the patch hunk for `{issue}` in `{file_diff.new_path}`?", symbol, hc, "symbolic")
                if role == "implementation":
                    add(out, seen, row, "repair_target_symbol_candidate", f"What function or class is a likely repair target for `{issue}` in `{file_diff.new_path}`?", symbol, hc, "symbolic")
                elif role == "test":
                    add(out, seen, row, "test_oracle_symbol_candidate", f"What test-side symbol helps localize the regression oracle for `{issue}` in `{file_diff.new_path}`?", symbol, hc, "symbolic")
            tests = test_names(plus + minus + ctx_lines)
            for test_name in tests[:4]:
                add(out, seen, row, "test_oracle_added_or_changed_test_name", f"Which test function records the regression signal for `{issue}` in `{file_diff.new_path}`?", test_name, hc, "symbolic")
                add(out, seen, row, "test_oracle_localization", f"Where is the test oracle for `{issue}` localized?", f"{file_diff.new_path}:{hunk.new_start}-{hunk.new_start + max(0, hunk.new_len - 1)}", hc, "location")
            conditions_plus = [line.strip() for line in plus if CONDITION_LINE_RE.match(line)]
            conditions_minus = [line.strip() for line in minus if CONDITION_LINE_RE.match(line)]
            if conditions_plus:
                add(out, seen, row, "before_after_added_guard_condition", f"What new guard condition is introduced for `{issue}` in `{file_diff.new_path}`?", "\n".join(conditions_plus[:8]), hc, "list")
                add(out, seen, row, "repair_intent_input_validation", f"What input or edge-case validation does the patch add for `{issue}`?", "\n".join(conditions_plus[:8]), hc, "list")
            if conditions_minus and conditions_plus:
                add(out, seen, row, "before_after_condition_delta", f"How does the conditional logic change for `{issue}` in `{file_diff.new_path}`?", "Before:\n" + compact(conditions_minus, 8) + "\nAfter:\n" + compact(conditions_plus, 8), hc, "diff_summary")
            returns_plus = [line.strip() for line in plus if RETURN_LINE_RE.match(line)]
            returns_minus = [line.strip() for line in minus if RETURN_LINE_RE.match(line)]
            if returns_plus:
                add(out, seen, row, "before_after_added_return_behavior", f"What return behavior is introduced for `{issue}` in `{file_diff.new_path}`?", "\n".join(returns_plus[:8]), hc, "list")
            if returns_minus and returns_plus:
                add(out, seen, row, "before_after_return_delta", f"How does return behavior change for `{issue}` in `{file_diff.new_path}`?", "Before:\n" + compact(returns_minus, 8) + "\nAfter:\n" + compact(returns_plus, 8), hc, "diff_summary")
            exceptions_plus = [line.strip() for line in plus if EXCEPTION_LINE_RE.search(line)]
            exceptions_minus = [line.strip() for line in minus if EXCEPTION_LINE_RE.search(line)]
            if exceptions_plus:
                add(out, seen, row, "before_after_added_exception_behavior", f"What exception-handling behavior is introduced for `{issue}` in `{file_diff.new_path}`?", "\n".join(exceptions_plus[:8]), hc, "list")
            if exceptions_minus and exceptions_plus:
                add(out, seen, row, "before_after_exception_delta", f"How does exception behavior change for `{issue}` in `{file_diff.new_path}`?", "Before:\n" + compact(exceptions_minus, 8) + "\nAfter:\n" + compact(exceptions_plus, 8), hc, "diff_summary")
            imports_plus = [line.strip() for line in plus if IMPORT_LINE_RE.match(line)]
            if imports_plus:
                add(out, seen, row, "cross_file_added_import_dependency", f"Which added import may support the fix for `{issue}` in `{file_diff.new_path}`?", "\n".join(imports_plus[:8]), hc, "list")
                add(out, seen, row, "repair_planning_dependency_to_inspect", f"What new dependency should be inspected to understand the repair for `{issue}`?", "\n".join(imports_plus[:8]), hc, "list")
            assertions_plus = [line.strip() for line in plus if ASSERT_LINE_RE.search(line)]
            assertions_minus = [line.strip() for line in minus if ASSERT_LINE_RE.search(line)]
            if assertions_plus:
                add(out, seen, row, "test_oracle_added_assertion", f"What assertion or expected failure is added for `{issue}`?", "\n".join(assertions_plus[:8]), hc, "list")
                add(out, seen, row, "test_oracle_behavior_under_test", f"What behavior does the test patch make observable for `{issue}`?", "\n".join(assertions_plus[:8]), hc, "list")
            if assertions_minus and assertions_plus:
                add(out, seen, row, "test_oracle_assertion_delta", f"How does the regression assertion change for `{issue}` in `{file_diff.new_path}`?", "Before:\n" + compact(assertions_minus, 8) + "\nAfter:\n" + compact(assertions_plus, 8), hc, "diff_summary")
            call_candidates = calls(plus)
            if call_candidates:
                add(out, seen, row, "cross_file_call_or_api_surface", f"Which call or API surface appears in the added code for `{issue}` in `{file_diff.new_path}`?", "\n".join(call_candidates[:8]), hc, "list")
            if role == "test" and source_paths:
                add(out, seen, row, "test_oracle_to_source_localization", f"Which implementation file should be paired with this test hunk for `{issue}`?", source_paths[0], hc + [ctx(source_paths[0], 1, 1, "implementation file changed by patch")], "path")
            if role == "implementation" and test_paths:
                add(out, seen, row, "source_edit_to_test_oracle_link", f"Which test file should be checked against this implementation hunk for `{issue}`?", test_paths[0], hc + [ctx(test_paths[0], 1, 1, "test file changed by patch")], "path")
            if hunk_index < 2 and all_paths:
                others = [path for path in all_paths if path != file_diff.new_path]
                if others:
                    add(out, seen, row, "contrastive_hunk_file_vs_other_changed_file", f"For `{issue}`, which changed file contains this hunk rather than `{others[0]}`?", file_diff.new_path, hc + [ctx(others[0], 1, 1, "other changed file for contrast")], "path")
    return select_balanced_rows(row, out, rows_per_issue)


def iter_patch_rows(path: Path) -> Iterable[dict[str, Any]]:
    columns = [
        "source_dataset",
        "row_index",
        "instance_id",
        "repo_id",
        "base_commit",
        "commit_sha",
        "problem_statement",
        "patch",
        "target_patch",
        "test_patch",
        "usable_for_train",
    ]
    table = pq.read_table(path, columns=columns)
    yield from table.to_pylist()


def load_seen(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    for row in iter_jsonl(path):
        rid = str(row.get("id") or "")
        if rid:
            seen.add(rid)
    return seen


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-metadata", type=Path, default=Path("/path/to/ad-local/storage/issuefix_swefixer/metadata/swefixer_full.parquet"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=24)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--target-rows", type=int, default=300000)
    parser.add_argument("--rows-per-issue", type=int, default=96)
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
        for row in iter_patch_rows(args.input_metadata):
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
                print(
                    json.dumps(
                        {
                            "issues_seen": counts["issues_seen"],
                            "generated": generated,
                            "elapsed_sec": round(time.time() - started, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

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
