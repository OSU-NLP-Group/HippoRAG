#!/usr/bin/env python3
"""Hard-reject and deduplicate Code2LoRA QA candidates.

This intentionally does not do semantic judging.  It keeps rows with the basic
fields needed for training, collapses deterministic originals against their LLM
rewrites, and deduplicates exact normalized QA content.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path(
    "/path/to/ad-local/storage/code2lora_jina_enriched_qa/"
    "enriched_qa_only_full_20260711"
)

DETERMINISTIC_GLOBS = [
    "ast_qa_generated_v1_20260715/shards/*/ast_qa.generated.jsonl",
    "ast_qa_generated_v1_20260715_supplement_skip40/shards/*/ast_qa.generated.jsonl",
    "swefixer_patch_qa_v1_20260716/shards/*/swefixer_deterministic_qa.generated.jsonl",
    "swefixer_repo_static_qa_v1_20260716/shards/*/swefixer_deterministic_qa.generated.jsonl",
    "deterministic_expansion_qa_v1_12m_20260717/shards/*/deterministic_expansion_qa.generated.jsonl",
    "diverse_repo_coverage_qa_v1_10m_20260717/shards/*/diverse_repo_coverage_qa.generated.jsonl",
    "deep_surface_qa_v1_20260718/*/shards/*/deep_surface_qa.generated.jsonl",
    "swefixer_patch_reasoning_qa_v2_20260720/shards/*/swefixer_patch_reasoning_qa.generated.jsonl",
    "swefixer_patch_reasoning_extra_qa_v1_20260720/shards/*/swefixer_patch_reasoning_extra_qa.generated.jsonl",
    "cross_file_aug_from_final_qa_v1_6m_20260720/shards/*/cross_file_aug.generated.jsonl",
]
REWRITE_GLOBS = [
    "deterministic_qa_rewritten_gemma4_12b*/rewrite_output*.jsonl",
]
LLM_RAW_GLOBS = [
    "llm_qa_raw_gemma4_12b_repair_20260716/**/llm_qa.raw_generations.jsonl",
    "llm_qa_raw_gemma4_12b_vllm_residual_20260717/**/llm_qa.raw_generations.jsonl",
    "llm_qa_raw_gemma4_12b_vllm_followon_20260717/**/llm_qa.raw_generations.jsonl",
]
BAD_TEXT_PATTERNS = [
    re.compile(r"\b(?:traceback \(most recent call last\)|exception in ASGI application)\b", re.I),
    re.compile(r"\b(?:i cannot|i can't|as an ai language model|i do not have access)\b", re.I),
    re.compile(r"\b(?:vllm|openai)\s+(?:error|exception)\b", re.I),
]


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def stable_hash_json(value: Any) -> str:
    return stable_hash(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_for_key(text: Any) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"[`'\"“”‘’]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def row_id(row: dict[str, Any]) -> str:
    value = row.get("id") or row.get("qa_id") or row.get("rewrite_input_id")
    if value:
        return str(value)
    return stable_hash_json(
        {
            "repo_id": row.get("repo_id"),
            "base_commit": row.get("base_commit") or row.get("commit_sha") or row.get("commit"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "task_category": row.get("task_category") or row.get("category"),
        }
    )


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any] | None, str | None]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, None, f"invalid_json:{exc.msg}"
                continue
            if not isinstance(row, dict):
                yield line_no, None, "not_object"
                continue
            yield line_no, row, None


def expand_patterns(root: Path, patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        out.extend(glob.glob(str(root / pattern), recursive=True))
    return sorted(dict.fromkeys(path for path in out if Path(path).is_file()))


def parse_llm_response(row: dict[str, Any]) -> list[dict[str, Any]]:
    payload = row.get("response")
    if payload is None:
        raw = row.get("raw_response")
        if isinstance(raw, dict):
            try:
                payload = raw["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                payload = None
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        payload = json.loads(text)
    if isinstance(payload, dict):
        for key in ("qa", "qas", "qa_pairs", "questions", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            if "question" in payload and "answer" in payload:
                payload = [payload]
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def context_is_valid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for span in value:
        if not isinstance(span, dict):
            return False
        path = normalize_space(span.get("path"))
        if not path:
            return False
        try:
            start = int(span.get("start_line"))
            end = int(span.get("end_line"))
        except (TypeError, ValueError):
            return False
        if start < 1 or end < start:
            return False
    return True


def hard_reject_reason(row: dict[str, Any]) -> str | None:
    required = {
        "repo_id": normalize_space(row.get("repo_id")),
        "base_commit": normalize_space(row.get("base_commit") or row.get("commit_sha") or row.get("commit")),
        "question": normalize_space(row.get("question")),
        "answer": normalize_space(row.get("answer")),
    }
    for field, value in required.items():
        if not value:
            return f"missing_{field}"
    if not context_is_valid(row.get("required_context")):
        return "missing_or_malformed_required_context"
    question = required["question"]
    answer = required["answer"]
    if len(question) < 8:
        return "question_too_short"
    if len(answer) < 1:
        return "answer_too_short"
    if len(question) > 4000 or len(answer) > 12000:
        return "question_or_answer_too_long"
    text = f"{question}\n{answer}"
    for pattern in BAD_TEXT_PATTERNS:
        if pattern.search(text):
            return "bad_error_or_refusal_text"
    return None


def canonical_row(
    row: dict[str, Any],
    *,
    source_family: str,
    source_file: str,
    line_no: int,
    priority: int,
    replacement_key: str,
) -> tuple[dict[str, Any] | None, str]:
    reason = hard_reject_reason(row)
    if reason:
        return None, reason
    repo_id = normalize_space(row.get("repo_id"))
    base_commit = normalize_space(row.get("base_commit") or row.get("commit_sha") or row.get("commit"))
    question = normalize_space(row.get("question"))
    answer = normalize_space(row.get("answer"))
    task_category = normalize_space(row.get("task_category") or row.get("category") or "unknown")
    qa_source = normalize_space(row.get("qa_source") or source_family)
    source_id = row_id(row)
    content_key = stable_hash_json(
        {
            "repo_id": repo_id,
            "base_commit": base_commit,
            "question": normalize_for_key(question),
            "answer": normalize_for_key(answer),
        }
    )
    out = {
        "id": stable_hash_json({"content_key": content_key, "replacement_key": replacement_key, "source_id": source_id}),
        "source_id": source_id,
        "replacement_key": replacement_key,
        "content_key": content_key,
        "source_family": source_family,
        "qa_source": qa_source,
        "repo_id": repo_id,
        "base_commit": base_commit,
        "commit_sha": normalize_space(row.get("commit_sha") or base_commit),
        "question": question,
        "answer": answer,
        "task_category": task_category,
        "answer_style": normalize_space(row.get("answer_style") or ""),
        "required_context": row.get("required_context"),
        "generator": normalize_space(row.get("generator") or row.get("generator_model") or row.get("rewriter_model") or ""),
        "validation_status": "accepted_hardreject_dedup_candidate",
        "_priority": priority,
        "_source_file": source_file,
        "_source_line": line_no,
    }
    for key in (
        "instance_id",
        "source_instance_id",
        "source_dataset",
        "pack_id",
        "prompt_version",
        "generator_model",
        "rewriter_model",
        "rewrite_input_id",
        "rewrite_sample_version",
        "rewrite_prompt_version",
        "original_question",
        "original_answer",
    ):
        if key in row and row.get(key) not in (None, ""):
            out[key] = row.get(key)
    return out, "accepted"


def normalize_deterministic(row: dict[str, Any], source_file: str, line_no: int) -> tuple[dict[str, Any] | None, str]:
    rid = row_id(row)
    source_family = normalize_space(row.get("source_family") or "deterministic_original")
    if not source_family.startswith("deterministic"):
        source_family = "deterministic_original"
    return canonical_row(
        row,
        source_family=source_family,
        source_file=source_file,
        line_no=line_no,
        priority=10,
        replacement_key=f"deterministic_original:{rid}",
    )


def normalize_rewrite(row: dict[str, Any], source_file: str, line_no: int) -> tuple[dict[str, Any] | None, str]:
    if row.get("rewrite_status") != "rewritten":
        return None, "rewrite_not_successful"
    rewrite_input_id = normalize_space(row.get("rewrite_input_id"))
    if not rewrite_input_id:
        return None, "missing_rewrite_input_id"
    out = dict(row)
    out["question"] = normalize_space(row.get("rewritten_question") or row.get("question"))
    out["answer"] = normalize_space(row.get("rewritten_answer") or row.get("answer"))
    return canonical_row(
        out,
        source_family="llm_rewrite",
        source_file=source_file,
        line_no=line_no,
        priority=30,
        replacement_key=f"deterministic_original:{rewrite_input_id}",
    )


def normalize_llm_items(row: dict[str, Any], source_file: str, line_no: int) -> Iterable[tuple[dict[str, Any] | None, str]]:
    if normalize_space(row.get("status")) != "generated":
        yield None, "llm_status_not_generated"
        return
    try:
        items = parse_llm_response(row)
    except (json.JSONDecodeError, TypeError, ValueError):
        yield None, "llm_response_invalid_json"
        return
    if not items:
        yield None, "llm_no_qa_items"
        return
    for item_index, item in enumerate(items):
        out = dict(item)
        for key in ("repo_id", "base_commit", "commit_sha", "pack_id", "prompt_version", "generator_model", "decoding"):
            if key not in out and key in row:
                out[key] = row.get(key)
        if "task_category" not in out and row.get("category"):
            out["task_category"] = row.get("category")
        pack_id = normalize_space(row.get("pack_id") or f"{source_file}:{line_no}")
        replacement_key = f"llm_generated:{pack_id}:{item_index}"
        yield canonical_row(
            out,
            source_family="llm_generated",
            source_file=source_file,
            line_no=line_no,
            priority=20,
            replacement_key=replacement_key,
        )


def better(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Return the preferred row. Higher priority wins; ties are deterministic."""
    pa = int(a.get("_priority") or 0)
    pb = int(b.get("_priority") or 0)
    if pa != pb:
        return a if pa > pb else b
    ka = (len(str(a.get("answer") or "")), str(a.get("id") or ""))
    kb = (len(str(b.get("answer") or "")), str(b.get("id") or ""))
    return a if ka >= kb else b


def bucket_for(key: str, buckets: int) -> int:
    return int(stable_hash(key)[:12], 16) % buckets


def open_bucket_handles(root: Path, prefix: str, buckets: int):
    root.mkdir(parents=True, exist_ok=True)
    handles = []
    for idx in range(buckets):
        path = root / f"{prefix}_{idx:04d}.jsonl"
        handles.append(path.open("w", encoding="utf-8"))
    return handles


def close_handles(handles: list[Any]) -> None:
    for handle in handles:
        handle.close()


def create_manifest(args: argparse.Namespace) -> int:
    root = Path(args.root)
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "deterministic": expand_patterns(root, DETERMINISTIC_GLOBS),
        "rewrites": expand_patterns(root, REWRITE_GLOBS),
        "llm_raw": expand_patterns(root, LLM_RAW_GLOBS),
    }
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_root) / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: len(value) if isinstance(value, list) else value for key, value in manifest.items()}, indent=2))
    print(f"manifest_path={out_path}")
    return 0


def load_manifest(output_root: Path) -> dict[str, Any]:
    return json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))


def map_candidates(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    manifest = load_manifest(output_root)
    map_root = output_root / "stage0_replacement_buckets" / f"map_{args.map_index:03d}_of_{args.num_maps:03d}"
    handles = open_bucket_handles(map_root, "replace_bucket", args.num_buckets)
    counts: Counter[str] = Counter()
    started = time.time()

    def write_candidate(candidate: dict[str, Any] | None, reason: str) -> None:
        counts[f"reason_{reason}"] += 1
        if candidate is None:
            return
        idx = bucket_for(str(candidate["replacement_key"]), args.num_buckets)
        handles[idx].write(json.dumps(candidate, ensure_ascii=True, sort_keys=True) + "\n")
        counts["written_candidates"] += 1

    try:
        file_items: list[tuple[str, str]] = []
        for source_kind in ("deterministic", "rewrites", "llm_raw"):
            file_items.extend((source_kind, path) for path in manifest.get(source_kind, []))
        for file_index, (source_kind, path_str) in enumerate(file_items):
            if file_index % args.num_maps != args.map_index:
                continue
            path = Path(path_str)
            counts[f"files_{source_kind}"] += 1
            for line_no, row, err in iter_jsonl(path):
                counts[f"lines_{source_kind}"] += 1
                if err:
                    counts[f"hard_reject_{err.split(':', 1)[0]}"] += 1
                    continue
                assert row is not None
                if source_kind == "deterministic":
                    write_candidate(*normalize_deterministic(row, path_str, line_no))
                elif source_kind == "rewrites":
                    write_candidate(*normalize_rewrite(row, path_str, line_no))
                else:
                    for candidate, reason in normalize_llm_items(row, path_str, line_no):
                        write_candidate(candidate, reason)
    finally:
        close_handles(handles)

    audit = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - started, 3),
        "map_index": args.map_index,
        "num_maps": args.num_maps,
        "num_buckets": args.num_buckets,
        "counts": dict(counts),
    }
    (map_root / "map.audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def reduce_replacements(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    stage0 = output_root / "stage0_replacement_buckets"
    stage1 = output_root / "stage1_content_buckets" / f"replace_{args.bucket_index:04d}"
    handles = open_bucket_handles(stage1, "content_bucket", args.num_buckets)
    best_by_replacement: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    started = time.time()
    try:
        for path in sorted(stage0.glob(f"map_*_of_*/replace_bucket_{args.bucket_index:04d}.jsonl")):
            counts["input_files"] += 1
            for _line_no, row, err in iter_jsonl(path):
                if err or row is None:
                    counts["invalid_stage0"] += 1
                    continue
                key = str(row.get("replacement_key") or "")
                if not key:
                    counts["missing_replacement_key"] += 1
                    continue
                old = best_by_replacement.get(key)
                best_by_replacement[key] = row if old is None else better(row, old)
                counts["rows_read"] += 1
        for row in best_by_replacement.values():
            idx = bucket_for(str(row["content_key"]), args.num_buckets)
            handles[idx].write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            counts["rows_after_replacement"] += 1
    finally:
        close_handles(handles)
    counts["replacement_duplicates_removed"] = counts["rows_read"] - counts["rows_after_replacement"]
    audit = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - started, 3),
        "bucket_index": args.bucket_index,
        "num_buckets": args.num_buckets,
        "counts": dict(counts),
    }
    (stage1 / "replace_reduce.audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def strip_internal(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_") and key not in {"replacement_key", "content_key"}}


def reduce_content(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    stage1 = output_root / "stage1_content_buckets"
    final_root = output_root / "final_shards"
    final_root.mkdir(parents=True, exist_ok=True)
    final_path = final_root / f"final_qa.shard_{args.bucket_index:04d}_of_{args.num_buckets:04d}.jsonl"
    best_by_content: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    by_source_family: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    started = time.time()
    for path in sorted(stage1.glob(f"replace_*/content_bucket_{args.bucket_index:04d}.jsonl")):
        counts["input_files"] += 1
        for _line_no, row, err in iter_jsonl(path):
            if err or row is None:
                counts["invalid_stage1"] += 1
                continue
            key = str(row.get("content_key") or "")
            if not key:
                counts["missing_content_key"] += 1
                continue
            old = best_by_content.get(key)
            best_by_content[key] = row if old is None else better(row, old)
            counts["rows_read"] += 1
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in best_by_content.values():
            clean = strip_internal(row)
            by_source_family[str(clean.get("source_family") or "unknown")] += 1
            by_category[str(clean.get("task_category") or "unknown")] += 1
            handle.write(json.dumps(clean, ensure_ascii=True, sort_keys=True) + "\n")
    tmp_path.replace(final_path)
    counts["rows_final"] = len(best_by_content)
    counts["content_duplicates_removed"] = counts["rows_read"] - counts["rows_final"]
    audit = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - started, 3),
        "bucket_index": args.bucket_index,
        "num_buckets": args.num_buckets,
        "final_path": str(final_path),
        "counts": dict(counts),
        "by_source_family": dict(by_source_family),
        "by_category_top100": by_category.most_common(100),
    }
    audit_path = final_root / f"final_qa.shard_{args.bucket_index:04d}_of_{args.num_buckets:04d}.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def summarize(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    final_root = output_root / "final_shards"
    counts: Counter[str] = Counter()
    by_source_family: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    for audit_path in sorted(final_root.glob("final_qa.shard_*_of_*.audit.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for key, value in audit.get("counts", {}).items():
            counts[key] += int(value)
        by_source_family.update({k: int(v) for k, v in audit.get("by_source_family", {}).items()})
        for category, value in audit.get("by_category_top100", []):
            by_category[category] += int(value)
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_root": str(output_root),
        "final_shards": len(list(final_root.glob("final_qa.shard_*_of_*.jsonl"))),
        "counts": dict(counts),
        "by_source_family": dict(by_source_family),
        "top_categories_partial_from_shard_top100": by_category.most_common(100),
    }
    summary_path = output_root / "final_qa_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["manifest", "map", "reduce-replacement", "reduce-content", "summarize"], required=True)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-maps", type=int, default=16)
    parser.add_argument("--map-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--num-buckets", type=int, default=64)
    parser.add_argument("--bucket-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    args = parser.parse_args()

    if args.mode == "manifest":
        return create_manifest(args)
    if args.mode == "map":
        return map_candidates(args)
    if args.mode == "reduce-replacement":
        return reduce_replacements(args)
    if args.mode == "reduce-content":
        return reduce_content(args)
    if args.mode == "summarize":
        return summarize(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
