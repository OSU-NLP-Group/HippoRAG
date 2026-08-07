#!/usr/bin/env python3
"""Expand deterministic Code2LoRA QA into additional high-precision templates.

This is a CPU-only, packless generator.  It reads existing deterministic QA rows
and emits disjoint derived QA rows for three scale-up lanes:

* patch/path issue QA
* AST/code-graph navigation QA
* packless chunk/file context QA
"""

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


GENERATOR = "deterministic_expansion_qa_v1"
DEFAULT_INPUT_GLOBS = [
    "/path/to/ad-local/storage/code2lora_jina_enriched_qa/enriched_qa_only_full_20260711/"
    "ast_qa_generated_v1_20260715/shards/*/ast_qa.generated.jsonl",
    "/path/to/ad-local/storage/code2lora_jina_enriched_qa/enriched_qa_only_full_20260711/"
    "ast_qa_generated_v1_20260715_supplement_skip40/shards/*/ast_qa.generated.jsonl",
    "/path/to/ad-local/storage/code2lora_jina_enriched_qa/enriched_qa_only_full_20260711/"
    "swefixer_patch_qa_v1_20260716/shards/*/swefixer_deterministic_qa.generated.jsonl",
    "/path/to/ad-local/storage/code2lora_jina_enriched_qa/enriched_qa_only_full_20260711/"
    "swefixer_repo_static_qa_v1_20260716/shards/*/swefixer_deterministic_qa.generated.jsonl",
]


def stable_id(*parts: Any) -> str:
    raw = "\0".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()


def expand_inputs(patterns: List[str]) -> List[Path]:
    paths = []  # type: List[Path]
    for pattern in patterns:
        matches = sorted(Path("/").glob(pattern.lstrip("/")))
        paths.extend(path for path in matches if path.is_file())
    return sorted(dict.fromkeys(paths))


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%s invalid JSON: %s" % (path, line_no, exc))
            if isinstance(row, dict):
                yield row


def short(text: Any, limit: int = 220) -> str:
    value = " ".join(str(text or "").split())
    return value[:limit].rstrip()


def source_row_id(row: Dict[str, Any]) -> str:
    rid = str(row.get("id") or row.get("rewrite_input_id") or "")
    if rid:
        return rid
    return stable_id(row.get("repo_id", ""), row.get("question", ""), row.get("answer", ""), row.get("task_category", ""))


def context_items(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = row.get("required_context")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def context_lines(row: Dict[str, Any], limit: int = 8) -> List[str]:
    lines = []  # type: List[str]
    for item in context_items(row)[:limit]:
        path = str(item.get("path") or "")
        if not path:
            continue
        start = item.get("start_line")
        end = item.get("end_line")
        reason = str(item.get("reason") or "").strip()
        span = path
        if start is not None:
            span += ":%s" % start
            if end is not None and end != start:
                span += "-%s" % end
        if reason:
            span += " (%s)" % short(reason, 120)
        lines.append(span)
    return lines


def first_context(row: Dict[str, Any]) -> Tuple[str, int, int]:
    for item in context_items(row):
        path = str(item.get("path") or "")
        if path:
            start = int(item.get("start_line") or 1)
            end = int(item.get("end_line") or start)
            return path, start, end
    path = str(row.get("source_path") or "")
    return path, 1, 1


def make_row(
    row: Dict[str, Any],
    category: str,
    question: str,
    answer: str,
    answer_style: str,
    family: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    question = short(question, 520)
    answer = str(answer or "").strip()
    if not question or not answer:
        return None
    parent_id = source_row_id(row)
    payload = {
        "id": stable_id(GENERATOR, parent_id, category, question, answer),
        "parent_qa_id": parent_id,
        "repo_id": row.get("repo_id", ""),
        "base_commit": row.get("base_commit", ""),
        "commit_sha": row.get("commit_sha", row.get("base_commit", "")),
        "instance_id": row.get("instance_id", ""),
        "source_instance_id": row.get("source_instance_id", ""),
        "source_dataset": row.get("source_dataset", ""),
        "qa_source": "%s_expanded" % str(row.get("qa_source") or "deterministic"),
        "source_task_category": row.get("task_category", ""),
        "task_category": category,
        "question": question,
        "answer": answer,
        "answer_style": answer_style,
        "required_context": row.get("required_context", []),
        "generator": GENERATOR,
        "generator_family": family,
    }
    for key in (
        "source_path",
        "symbol_kind",
        "symbol_name",
        "qualified_name",
        "import_statement",
        "resolved_path",
        "rewrite_sample_version",
    ):
        if key in row:
            payload[key] = row[key]
    if extra:
        payload.update(extra)
    return payload


def ast_expansions(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []  # type: List[Dict[str, Any]]
    cat = str(row.get("task_category") or "")
    repo = str(row.get("repo_id") or "the repository")
    answer = str(row.get("answer") or "").strip()
    qname = str(row.get("qualified_name") or row.get("symbol_name") or "").strip("`")
    symbol = str(row.get("symbol_name") or qname.split(".")[-1] or "").strip("`")
    path, _start, _end = first_context(row)

    def add(category: str, question: str, ans: str, style: str = "concise") -> None:
        made = make_row(row, category, question, ans, style, "ast_code_graph")
        if made:
            out.append(made)

    if cat == "qa_symbol_path" and qname and answer:
        add("ast_edit_location", "Where should you look to inspect or edit the definition of `%s` in `%s`?" % (qname, repo), answer, "path")
        add("ast_definition_file_reverse", "Which symbol is defined in `%s` at the recorded context span?" % answer, qname, "symbol")
        add("ast_symbol_navigation", "What repository path contains the implementation for `%s`?" % qname, answer, "path")
    elif cat == "qa_signature" and qname and answer:
        add("ast_signature_to_symbol", "Which callable in `%s` has the signature `%s`?" % (repo, answer), qname, "symbol")
        add("ast_callable_signature_grounding", "What exact signature should be used for `%s` when reasoning about this repository?" % qname, answer, "code")
        if path:
            add("ast_signature_file_location", "Which file contains the callable with signature `%s`?" % answer, path, "path")
    elif cat == "qa_method_class" and symbol and answer:
        add("ast_method_owner_reverse", "Which method is defined on class `%s` in `%s`?" % (answer, repo), symbol, "symbol")
        add("ast_method_class_navigation", "When editing method `%s`, which class should be inspected?" % symbol, answer, "symbol")
    elif cat == "qa_class_bases" and qname and answer:
        add("ast_base_to_class", "Which class declares `%s` as its recorded base-class list?" % answer, qname, "symbol")
        add("ast_class_inheritance_fact", "What inheritance relationship is recorded for `%s`?" % qname, answer, "list")
    elif cat == "qa_constant" and qname and answer:
        add("ast_constant_reverse_lookup", "Which constant in `%s` has value `%s`?" % (repo, short(answer, 160)), qname, "symbol")
        add("ast_constant_value_grounding", "What value is assigned to `%s` in this snapshot?" % qname, answer, "literal")
    elif cat == "qa_enum_members" and qname and answer:
        add("ast_enum_reverse_lookup", "Which enum defines the members `%s`?" % short(answer, 180), qname, "symbol")
        add("ast_enum_member_grounding", "What members should be associated with enum `%s`?" % qname, answer, "list")
    elif cat == "qa_import_resolution" and answer:
        stmt = str(row.get("import_statement") or "").strip()
        if stmt:
            add("ast_import_statement_to_target", "What target does the import statement `%s` resolve to?" % short(stmt, 240), answer, "path")
        if path:
            add("ast_import_location_grounding", "Which file contains the import resolution fact `%s`?" % short(answer, 200), path, "path")
    return out


def patch_expansions(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []  # type: List[Dict[str, Any]]
    cat = str(row.get("task_category") or "")
    answer = str(row.get("answer") or "").strip()
    question = str(row.get("question") or "").strip()
    instance_id = str(row.get("instance_id") or row.get("source_instance_id") or "").strip()
    path, _start, _end = first_context(row)

    def add(category: str, q: str, ans: str, style: str = "concise") -> None:
        made = make_row(row, category, q, ans, style, "issue_patch_path")
        if made:
            out.append(made)

    if not answer:
        return out
    issue_label = "SWE-fixer instance `%s`" % instance_id if instance_id else "the issue"
    if cat in {"issue_patch_changed_files", "issue_patch_source_files", "issue_patch_test_files"}:
        kind = "changed files"
        if cat.endswith("source_files"):
            kind = "source files"
        elif cat.endswith("test_files"):
            kind = "test files"
        add("issue_patch_file_plan", "For %s, what %s are implicated by the patch?" % (issue_label, kind), answer, "list")
        add("issue_patch_path_localization", "Which repository paths should be inspected for %s based on the patch metadata?" % issue_label, answer, "list")
        if path:
            add("issue_patch_first_context_path", "Which path provides the first recorded context span for %s?" % issue_label, path, "path")
    elif cat.startswith("patch_") or cat.startswith("test_patch_"):
        add("issue_patch_exact_fact_restated", "What deterministic patch fact is recorded for %s by `%s`?" % (issue_label, cat), answer, row.get("answer_style", "concise"))
        if path:
            add("issue_patch_fact_location", "Which file contains the `%s` patch fact for %s?" % (cat, issue_label), path, "path")
        if cat == "patch_hunk_replacement":
            add("issue_patch_before_after_change", "What before/after replacement does the patch apply for %s?" % issue_label, answer, "diff_summary")
        elif cat == "patch_hunk_added_lines":
            add("issue_patch_added_code", "What exact code lines are added for %s?" % issue_label, answer, "code")
        elif cat == "patch_added_conditions":
            add("issue_patch_guard_conditions", "Which guard or branch conditions are introduced for %s?" % issue_label, answer, "list")
        elif cat == "patch_added_exceptions":
            add("issue_patch_exception_behavior", "Which exception-handling behavior is introduced for %s?" % issue_label, answer, "list")
        elif cat == "patch_added_returns":
            add("issue_patch_return_behavior", "Which return behavior is introduced for %s?" % issue_label, answer, "list")
        elif cat == "patch_added_imports":
            add("issue_patch_import_dependencies", "Which imports are introduced by the patch for %s?" % issue_label, answer, "list")
        elif cat == "patch_added_symbols":
            add("issue_patch_new_symbols", "Which new functions or classes are introduced by the patch for %s?" % issue_label, answer, "list")
        elif cat in {"test_patch_assertions", "test_patch_added_tests"}:
            add("issue_patch_test_signal", "What test-side signal is added for %s?" % issue_label, answer, "list")
    elif question and instance_id:
        add("issue_patch_instance_fact", "For SWE-fixer instance `%s`, what is the answer to: %s" % (instance_id, short(question, 300)), answer, row.get("answer_style", "concise"))
    return out


def chunk_expansions(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []  # type: List[Dict[str, Any]]
    answer = str(row.get("answer") or "").strip()
    question = str(row.get("question") or "").strip()
    paths = context_lines(row)
    path, start, end = first_context(row)
    repo = str(row.get("repo_id") or "the repository")
    cat = str(row.get("task_category") or "deterministic QA")

    def add(category: str, q: str, ans: str, style: str = "concise") -> None:
        made = make_row(row, category, q, ans, style, "packless_chunk_file")
        if made:
            out.append(made)

    if paths and question:
        add("packless_required_context_paths", "Which repository context supports answering `%s`?" % short(question, 300), "\n".join(paths), "list")
    if path and answer:
        span = "%s:%s-%s" % (path, start, end)
        add("packless_answer_evidence_span", "Where in `%s` is the deterministic answer `%s` grounded?" % (repo, short(answer, 160)), span, "location")
    if path and cat:
        add("packless_task_context_file", "Which file should be inspected for a `%s` question in `%s`?" % (cat, repo), path, "path")
    return out


def expansions(row: Dict[str, Any], families: set) -> List[Dict[str, Any]]:
    out = []  # type: List[Dict[str, Any]]
    source = str(row.get("qa_source") or "")
    cat = str(row.get("task_category") or "")
    if "ast" in families and (source.startswith("ast_") or cat.startswith("qa_")):
        out.extend(ast_expansions(row))
    if "patch" in families and (source.startswith("swefixer_") or cat.startswith("patch_") or cat.startswith("issue_patch") or cat.startswith("test_patch")):
        out.extend(patch_expansions(row))
    if "chunk" in families:
        out.extend(chunk_expansions(row))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--families", default="patch,ast,chunk")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--target-rows", type=int, default=500000)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    families = set(item.strip() for item in args.families.split(",") if item.strip())
    unknown = families - {"patch", "ast", "chunk"}
    if unknown:
        raise SystemExit("unknown families: %s" % sorted(unknown))

    input_paths = expand_inputs(args.input_glob or DEFAULT_INPUT_GLOBS)
    if not input_paths:
        raise SystemExit("no input files matched")
    assigned_paths = [path for idx, path in enumerate(input_paths) if idx % args.num_shards == args.shard_index]
    if not assigned_paths:
        raise SystemExit("no input files assigned to shard %s/%s" % (args.shard_index, args.num_shards))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    generated = 0
    category_counts = Counter()
    if args.resume and args.output_jsonl.exists():
        for existing in iter_jsonl(args.output_jsonl):
            rid = str(existing.get("id") or "")
            if rid:
                seen.add(rid)
                generated += 1
                category_counts[str(existing.get("task_category") or "unknown")] += 1

    started = time.time()
    counts = Counter()
    mode = "a" if args.resume else "w"
    with args.output_jsonl.open(mode, encoding="utf-8") as out_fh:
        for path in assigned_paths:
            counts["input_files"] += 1
            for row in iter_jsonl(path):
                counts["source_rows_seen"] += 1
                for item in expansions(row, families):
                    rid = str(item.get("id") or "")
                    if not rid or rid in seen:
                        counts["duplicates_skipped"] += 1
                        continue
                    seen.add(rid)
                    out_fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                    generated += 1
                    counts["generated"] += 1
                    category_counts[str(item.get("task_category") or "unknown")] += 1
                    if args.progress_every and generated % args.progress_every == 0:
                        print(json.dumps({"generated": generated, "source_rows_seen": counts["source_rows_seen"]}, sort_keys=True), flush=True)
                    if args.target_rows and generated >= args.target_rows:
                        break
                if args.target_rows and generated >= args.target_rows:
                    break
            if args.target_rows and generated >= args.target_rows:
                break

    audit = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "assigned_input_files": [str(path) for path in assigned_paths],
        "all_input_file_count": len(input_paths),
        "elapsed_sec": round(time.time() - started, 3),
        "counts": dict(counts),
        "generated_total_in_output": generated,
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
