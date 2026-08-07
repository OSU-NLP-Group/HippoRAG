#!/usr/bin/env python3
"""Generate raw LLM QA JSONL from prepared Code2LoRA context packs.

This client is intentionally runtime-agnostic: it talks to an already-approved
OpenAI-compatible endpoint, such as vLLM or SGLang, and writes raw generations
for validate_code2lora_jina_llm_qa.py. It does not start or choose a serving
stack.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are a perceptive QA generator that creates context-dependent QA pairs "
    "exposing a wide surface of repository-specific knowledge. Generate questions "
    "whose answers require the supplied repository context, not generic programming "
    "knowledge. Return JSON only. If a JSON schema is requested, return an object "
    "with a qa array; otherwise return a list of objects. Each QA object must have "
    "question, answer, answer_style, task_category, and required_context. Each "
    "required_context item must cite path, start_line, end_line, and reason from "
    "the supplied context. Do not include markdown fences."
)


QA_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "qa": {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "answer_style": {"type": "string"},
                    "task_category": {"type": "string"},
                    "required_context": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "start_line": {"type": "integer"},
                                "end_line": {"type": "integer"},
                                "reason": {"type": "string"},
                            },
                            "required": ["path", "start_line", "end_line", "reason"],
                        },
                    },
                },
                "required": ["question", "answer", "answer_style", "task_category", "required_context"],
            },
        },
    },
    "required": ["qa"],
}


def refreshed_pack_prompt(pack: dict[str, Any]) -> str:
    contexts = pack.get("contexts") or []
    context_blocks = []
    for item in contexts:
        context_blocks.append(
            f"# path: {item.get('path', '')}\n# lines: {item.get('start_line')}-{item.get('end_line')}\n"
            f"# reason: {item.get('reason', '')}\n{item.get('text', '')}"
        )
    category = str(pack.get("category") or "llm_semantic_qa")
    return (
        "You are a perceptive QA generator that creates context-dependent QA pairs exposing a wide "
        "surface of repository-specific knowledge for a Code2LoRA dataset.\n"
        "Generate 1 or 2 concise, high-quality QA pairs. Use the pack category exactly as task_category: "
        f"{category}.\n"
        "The best questions should require reading and synthesizing the supplied repository context. "
        "Avoid generic documentation questions, shallow summaries, trivia, and narrow lookup questions "
        "already covered by AST templates, such as file-defines-symbol, exact signature, or simple "
        "import-location questions.\n"
        "Prefer questions that teach a model how this repository works. Good QA types include cross-file "
        "control/data flow, issue and patch reasoning, test-to-implementation reasoning, configuration/"
        "runtime behavior, and lifecycle/invariant reasoning.\n"
        "For cross_file_semantic packs, every QA pair must cite required_context spans from at least two "
        "different file paths. For other categories, cite every file needed to support the answer.\n"
        "Write answers that are specific, grounded, and explanatory, usually one or two sentences. "
        "Return JSON only. If a JSON schema is requested, return an object with a qa array; otherwise "
        "return a list of objects with question, answer, answer_style, task_category, and required_context.\n\n"
        f"Repository: {pack.get('repo_id', '')}\n"
        f"Pack category: {category}\n\n"
        + "\n\n---\n\n".join(context_blocks)
    )


def generator_instructions(pack: dict[str, Any], args: argparse.Namespace) -> str:
    if args.refresh_generator_instructions:
        return refreshed_pack_prompt(pack)
    return str(pack["generator_instructions"])


def prompt_version(pack: dict[str, Any], args: argparse.Namespace) -> str:
    if args.refresh_generator_instructions:
        return "repoqa_aug_context_pack_v2_runtime"
    return str(pack.get("prompt_version", ""))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_qa_response(text: str) -> list[dict[str, Any]]:
    parsed = json.loads(text)
    if isinstance(parsed, list):
        qa = parsed
    elif isinstance(parsed, dict):
        qa = parsed.get("qa")
    else:
        raise ValueError("response JSON is neither an object nor a list")
    if not isinstance(qa, list) or not qa:
        raise ValueError("response JSON has no nonempty qa list")
    for index, item in enumerate(qa):
        if not isinstance(item, dict):
            raise ValueError(f"qa[{index}] is not an object")
        missing = {
            "question",
            "answer",
            "answer_style",
            "task_category",
            "required_context",
        } - set(item)
        if missing:
            raise ValueError(f"qa[{index}] missing required fields: {sorted(missing)}")
    return qa


def is_parseable_generated(row: dict[str, Any]) -> bool:
    if row.get("status") != "generated":
        return False
    response = row.get("response")
    if not isinstance(response, str) or not response.strip():
        return False
    try:
        parse_qa_response(response)
    except (json.JSONDecodeError, ValueError, TypeError):
        return False
    return True


def load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    for row in iter_jsonl(path):
        pack_id = str(row.get("pack_id") or "")
        if pack_id and is_parseable_generated(row):
            seen.add(pack_id)
    return seen


def request_payload(pack: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": generator_instructions(pack, args)},
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
                "name": "repoqa_batch",
                "schema": QA_RESPONSE_SCHEMA,
            },
        }
    return payload


def post_chat(base_url: str, payload: dict[str, Any], *, timeout: float, api_key: str) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def response_text(response: dict[str, Any]) -> str:
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


def redacted_audit_args(args: argparse.Namespace) -> dict[str, Any]:
    audit_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "api_key"}
    audit_args["api_key_provided"] = bool(args.api_key and args.api_key != "EMPTY")
    return audit_args


def generate_one(pack: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = request_payload(pack, args)
    last_error = ""
    for attempt in range(args.retries + 1):
        try:
            response = post_chat(args.base_url, payload, timeout=args.timeout, api_key=args.api_key)
            text = response_text(response)
            parse_qa_response(text)
            return {
                "pack_id": pack["pack_id"],
                "repo_id": pack.get("repo_id", ""),
                "base_commit": pack.get("base_commit", ""),
                "category": pack.get("category", ""),
                "response": text,
                "raw_response": response,
                "generator_model": args.model,
                "prompt_version": prompt_version(pack, args),
                "decoding": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                },
                "status": "generated",
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
            last_error = str(exc)
            if attempt < args.retries:
                time.sleep(args.retry_sleep)
    return {
        "pack_id": pack.get("pack_id", ""),
        "repo_id": pack.get("repo_id", ""),
        "base_commit": pack.get("base_commit", ""),
        "category": pack.get("category", ""),
        "response": "",
        "generator_model": args.model,
        "prompt_version": prompt_version(pack, args),
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        "status": "error",
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--response-format", choices=("none", "json_object", "json_schema"), default="none")
    parser.add_argument("--refresh-generator-instructions", action="store_true")
    parser.add_argument("--limit-packs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.base_url.startswith(("http://", "https://")):
        raise SystemExit("--base-url must be an explicit OpenAI-compatible http(s) endpoint")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(args.output_jsonl) if args.resume else set()
    mode = "a" if args.resume else "w"
    counts: Counter[str] = Counter()
    started = time.time()
    packs: list[dict[str, Any]] = []
    for pack in iter_jsonl(args.packs_jsonl):
        pack_id = str(pack.get("pack_id") or "")
        if not pack_id or pack_id in seen:
            counts["skipped"] += 1
            continue
        if not pack.get("generator_instructions"):
            counts["missing_prompt"] += 1
            continue
        packs.append(pack)
        if args.limit_packs and len(packs) >= args.limit_packs:
            break

    concurrency = max(1, int(args.concurrency))
    with args.output_jsonl.open(mode, encoding="utf-8") as out_fh:
        if concurrency == 1:
            for pack in packs:
                row = generate_one(pack, args)
                out_fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                out_fh.flush()
                counts[str(row["status"])] += 1
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(generate_one, pack, args) for pack in packs]
                for future in as_completed(futures):
                    row = future.result()
                    out_fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    out_fh.flush()
                    counts[str(row["status"])] += 1
    audit = {
        "packs_jsonl": str(args.packs_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "counts": dict(counts),
        "elapsed_sec": round(time.time() - started, 3),
        "args": redacted_audit_args(args),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0 if counts.get("error", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
