#!/usr/bin/env python3
"""Rewrite deterministic Code2LoRA QA rows with an OpenAI-compatible LLM."""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


SYSTEM_PROMPT = (
    "You are a careful repository QA rewriter. Your job is to enrich deterministic "
    "question-answer pairs without changing their meaning. Preserve every factual "
    "claim, path, symbol name, signature, line reference, exception, test name, and "
    "patch fact exactly. Do not add unsupported details. Return JSON only."
)


REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "answer_style": {"type": "string"},
        "rewrite_notes": {"type": "string"},
    },
    "required": ["question", "answer", "answer_style"],
}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def response_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return ""


def row_id(row: Dict[str, Any]) -> str:
    return str(row.get("rewrite_input_id") or row.get("id") or "")


def parse_rewrite_response(text: str) -> Dict[str, Any]:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("rewrite response is not a JSON object")
    missing = {"question", "answer", "answer_style"} - set(parsed)
    if missing:
        raise ValueError(f"rewrite response missing required fields: {sorted(missing)}")
    for field in ("question", "answer", "answer_style"):
        if not isinstance(parsed.get(field), str) or not parsed[field].strip():
            raise ValueError(f"rewrite response field {field!r} is empty or non-string")
    return parsed


def is_valid_rewrite(row: Dict[str, Any]) -> bool:
    if row.get("rewrite_status") != "rewritten":
        return False
    return bool(str(row.get("rewritten_question") or "").strip()) and bool(
        str(row.get("rewritten_answer") or "").strip()
    )


def load_seen(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    seen = set()  # type: Set[str]
    for row in iter_jsonl(path):
        rid = row_id(row)
        if rid and is_valid_rewrite(row):
            seen.add(rid)
    return seen


def context_summary(row: Dict[str, Any], max_chars: int) -> str:
    pieces = []  # type: List[str]
    required_context = row.get("required_context")
    if isinstance(required_context, list):
        for item in required_context[:8]:
            if isinstance(item, dict):
                pieces.append(
                    " - "
                    + str(item.get("path", ""))
                    + (
                        f":{item.get('start_line')}-{item.get('end_line')}"
                        if item.get("start_line") is not None
                        else ""
                    )
                    + (f" ({item.get('reason')})" if item.get("reason") else "")
                )
    hint = row.get("context_hint")
    if isinstance(hint, str) and hint.strip():
        pieces.append("\nContext hint:\n" + hint[:max_chars])
    patch = row.get("patch")
    if isinstance(patch, str) and patch.strip():
        pieces.append("\nPatch excerpt:\n" + patch[:max_chars])
    return "\n".join(pieces)[:max_chars]


def rewrite_prompt(row: Dict[str, Any], max_context_chars: int) -> str:
    original_question = str(row.get("question") or "")
    original_answer = str(row.get("answer") or "")
    answer_style = str(row.get("answer_style") or "concise")
    task_category = str(row.get("task_category") or "unknown")
    repo_id = str(row.get("repo_id") or "")
    ctx = context_summary(row, max_context_chars)
    return (
        "Rewrite this deterministic repository QA pair for a training dataset.\n\n"
        "Goals:\n"
        "- Keep the exact same meaning and answer facts.\n"
        "- Make the question a little clearer and more natural when useful.\n"
        "- Make short answers slightly more explanatory, usually 1-3 sentences.\n"
        "- If the original answer is a path, symbol, signature, list, or literal value, include that exact "
        "literal unchanged in the rewritten answer.\n"
        "- Do not infer new behavior, motivations, causes, or patch effects beyond the provided QA/context.\n"
        "- Preserve task_category semantics; do not convert the QA into a different task.\n\n"
        "Return a JSON object with question, answer, answer_style, and rewrite_notes. "
        "The answer_style may be the original style or a slightly more descriptive style label.\n\n"
        f"Repository: {repo_id}\n"
        f"Task category: {task_category}\n"
        f"Original answer_style: {answer_style}\n"
        f"Original question: {original_question}\n"
        f"Original answer: {original_answer}\n\n"
        f"Available grounding context:\n{ctx}\n"
    )


def request_payload(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rewrite_prompt(row, args.max_context_chars)},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    elif args.response_format == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "deterministic_qa_rewrite",
                "schema": REWRITE_SCHEMA,
            },
        }
    return payload


def post_chat(base_url: str, payload: Dict[str, Any], timeout: float, api_key: str) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def rewrite_one(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    rid = row_id(row)
    payload = request_payload(row, args)
    last_error = ""
    for attempt in range(args.retries + 1):
        try:
            raw = post_chat(args.base_url, payload, timeout=args.timeout, api_key=args.api_key)
            text = response_text(raw)
            parsed = parse_rewrite_response(text)
            out = dict(row)
            out["original_question"] = row.get("question", "")
            out["original_answer"] = row.get("answer", "")
            out["original_answer_style"] = row.get("answer_style", "")
            out["question"] = parsed["question"].strip()
            out["answer"] = parsed["answer"].strip()
            out["answer_style"] = parsed["answer_style"].strip()
            out["rewritten_question"] = parsed["question"].strip()
            out["rewritten_answer"] = parsed["answer"].strip()
            out["rewrite_notes"] = str(parsed.get("rewrite_notes") or "").strip()
            out["rewrite_status"] = "rewritten"
            out["rewriter_model"] = args.model
            out["rewrite_prompt_version"] = "deterministic_qa_meaning_preserving_enrichment_v1"
            out["rewrite_decoding"] = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            }
            return out
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
            last_error = str(exc)
            if attempt < args.retries:
                time.sleep(args.retry_sleep)
    out = dict(row)
    out["rewrite_status"] = "error"
    out["rewrite_error"] = last_error
    out["rewriter_model"] = args.model
    out["rewrite_prompt_version"] = "deterministic_qa_meaning_preserving_enrichment_v1"
    if rid:
        out["rewrite_input_id"] = rid
    return out


def redacted_args(args: argparse.Namespace) -> Dict[str, Any]:
    result = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    result["api_key_provided"] = bool(args.api_key and args.api_key != "EMPTY")
    result.pop("api_key", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--response-format", choices=("none", "json_object", "json_schema"), default="json_schema")
    parser.add_argument("--max-context-chars", type=int, default=5000)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.base_url.startswith(("http://", "https://")):
        raise SystemExit("--base-url must be an explicit OpenAI-compatible endpoint")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(args.output_jsonl) if args.resume else set()
    mode = "a" if args.resume else "w"
    counts: Counter[str] = Counter()
    rows = []  # type: List[Dict[str, Any]]
    started = time.time()

    for row in iter_jsonl(args.input_jsonl):
        rid = row_id(row)
        if rid and rid in seen:
            counts["skipped"] += 1
            continue
        if not row.get("question") or not row.get("answer"):
            counts["missing_required_input"] += 1
            continue
        rows.append(row)
        if args.limit_rows and len(rows) >= args.limit_rows:
            break

    with args.output_jsonl.open(mode, encoding="utf-8") as out_fh:
        concurrency = max(1, args.concurrency)
        if concurrency == 1:
            for row in rows:
                result = rewrite_one(row, args)
                out_fh.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                out_fh.flush()
                counts[str(result.get("rewrite_status") or "unknown")] += 1
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(rewrite_one, row, args) for row in rows]
                for future in as_completed(futures):
                    result = future.result()
                    out_fh.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    out_fh.flush()
                    counts[str(result.get("rewrite_status") or "unknown")] += 1

    audit = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "counts": dict(counts),
        "elapsed_sec": round(time.time() - started, 3),
        "args": redacted_args(args),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0 if counts.get("error", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
