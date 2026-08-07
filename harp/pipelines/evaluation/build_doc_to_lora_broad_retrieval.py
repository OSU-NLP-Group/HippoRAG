#!/usr/bin/env python3
"""Build frozen, deterministic code/path-aware BM25 retrieval bundles.

One Slurm array task handles one repository snapshot.  Retrieval is constructed
without looking at gold answers, evidence spans, MCQ options, or patch metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from broad_eval_common import atomic_write_json, read_jsonl, repo_path  # noqa: E402


MODEL_NAME = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
BUDGET_TOKENS = {
    "500": 500,
    "1k": 1024,
    "2k": 2048,
    "8k": 8192,
}
TEXT_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html",
    ".ini", ".java", ".js", ".json", ".jsx", ".md", ".php", ".py", ".rb",
    ".rs", ".rst", ".scala", ".sh", ".sql", ".toml", ".ts", ".tsx", ".txt",
    ".xml", ".yaml", ".yml",
}
SKIP_PARTS = {
    ".git", ".github", ".idea", ".mypy_cache", ".pytest_cache", ".tox",
    ".venv", "__pycache__", "build", "dist", "node_modules", "vendor",
}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


@dataclass(frozen=True)
class Span:
    path: str
    start_line: int
    end_line: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--snapshot-index", type=int)
    parser.add_argument("--window-lines", type=int, default=120)
    parser.add_argument("--overlap-lines", type=int, default=30)
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    parser.add_argument(
        "--token-budgets",
        nargs="+",
        choices=sorted(BUDGET_TOKENS, key=lambda key: BUDGET_TOKENS[key]),
        default=["2k", "8k"],
        help="Named token budgets to freeze. Defaults preserve the original run.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.snapshot_index is None:
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        args.snapshot_index = int(task) if task is not None else 0
    if args.overlap_lines >= args.window_lines:
        parser.error("--overlap-lines must be smaller than --window-lines")
    return args


def git_bytes(repo: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        check=True,
    ).stdout


def eligible_path(path: str) -> bool:
    parsed = Path(path)
    return (
        parsed.suffix.lower() in TEXT_EXTENSIONS
        and not any(part in SKIP_PARTS for part in parsed.parts)
    )


def snapshot_spans(
    repo: Path,
    commit: str,
    window_lines: int,
    overlap_lines: int,
    max_file_bytes: int,
) -> tuple[list[Span], dict[str, Any]]:
    paths = [
        value.decode("utf-8", errors="replace")
        for value in git_bytes(repo, "ls-tree", "-r", "-z", "--name-only", commit).split(b"\0")
        if value
    ]
    spans: list[Span] = []
    skipped_large = 0
    skipped_binary = 0
    step = window_lines - overlap_lines
    for path in paths:
        if not eligible_path(path):
            continue
        blob = git_bytes(repo, "show", f"{commit}:{path}")
        if len(blob) > max_file_bytes:
            skipped_large += 1
            continue
        if b"\0" in blob[:8192]:
            skipped_binary += 1
            continue
        lines = blob.decode("utf-8", errors="replace").splitlines()
        for offset in range(0, max(1, len(lines)), step):
            selected = lines[offset : offset + window_lines]
            if not selected:
                break
            spans.append(
                Span(
                    path=path,
                    start_line=offset + 1,
                    end_line=offset + len(selected),
                    text="\n".join(selected),
                )
            )
            if offset + window_lines >= len(lines):
                break
    return spans, {
        "tree_path_count": len(paths),
        "indexed_span_count": len(spans),
        "skipped_large_files": skipped_large,
        "skipped_binary_files": skipped_binary,
    }


def terms(text: str) -> list[str]:
    output: list[str] = []
    for token in TOKEN_RE.findall(text):
        lowered = token.lower()
        output.append(lowered)
        if "_" in lowered:
            output.extend(part for part in lowered.split("_") if part)
        camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token).lower().split()
        if len(camel) > 1:
            output.extend(camel)
    return output


class BM25:
    def __init__(self, spans: list[Span]) -> None:
        self.spans = spans
        self.documents = [
            terms(span.path.replace("/", " ") + " " + span.text) for span in spans
        ]
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.lengths = [len(document) for document in self.documents]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency: Counter[str] = Counter()
        for document in self.term_frequencies:
            document_frequency.update(document.keys())
        count = len(spans)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def rank(self, query: str) -> list[tuple[float, int]]:
        query_terms = Counter(terms(query))
        ranked: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length_normalizer = 1.2 * (
                1.0 - 0.75 + 0.75 * self.lengths[index] / max(1, self.average_length)
            )
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                score += (
                    self.idf.get(term, 0.0)
                    * (frequency * 2.2)
                    / (frequency + length_normalizer)
                    * (1.0 + math.log(query_frequency))
                )
                if term in self.spans[index].path.lower():
                    score += 0.35 * self.idf.get(term, 0.0)
            ranked.append((score, index))
        return sorted(ranked, key=lambda value: (-value[0], self.spans[value[1]].path, self.spans[value[1]].start_line))


def render_span(span: Span) -> str:
    return (
        f'<file path="{span.path}" lines="{span.start_line}-{span.end_line}">\n'
        f"{span.text}\n</file>"
    )


def trim_text(tokenizer: Any, text: str, limit: int) -> tuple[str, int]:
    ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    if len(ids) <= limit:
        return text, len(ids)
    return tokenizer.decode(
        ids[:limit],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ), limit


def bundle(
    ranker: BM25,
    tokenizer: Any,
    query: str,
    token_budget: int,
) -> dict[str, Any]:
    started = time.time()
    selected: list[str] = []
    selected_metadata: list[dict[str, Any]] = []
    token_count = 0
    seen = set()
    for score, index in ranker.rank(query):
        span = ranker.spans[index]
        identity = (span.path, span.start_line, span.end_line)
        if identity in seen:
            continue
        remaining = token_budget - token_count
        if remaining <= 16:
            break
        rendered, count = trim_text(tokenizer, render_span(span), remaining)
        if count <= 0:
            continue
        selected.append(rendered)
        selected_metadata.append(
            {
                "path": span.path,
                "start_line": span.start_line,
                "end_line": span.end_line,
                "bm25_score": score,
                "token_count": count,
            }
        )
        token_count += count
        seen.add(identity)
        if token_count >= token_budget:
            break
    return {
        "text": "\n\n".join(selected),
        "token_count": token_count,
        "chunk_count": len(selected),
        "paths": list(dict.fromkeys(item["path"] for item in selected_metadata)),
        "spans": selected_metadata,
        "retrieval_seconds": time.time() - started,
    }


def main() -> int:
    args = parse_args()
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    rows = read_jsonl(args.dataset_jsonl)
    snapshots = sorted(
        {(str(row["repo_id"]), str(row["commit_sha"])) for row in rows}
    )
    if not 0 <= args.snapshot_index < len(snapshots):
        raise ValueError(
            f"snapshot index {args.snapshot_index} outside 0..{len(snapshots) - 1}"
        )
    repo_id, commit = snapshots[args.snapshot_index]
    selected = [
        row for row in rows
        if str(row["repo_id"]) == repo_id and str(row["commit_sha"]) == commit
    ]
    repository = repo_path(args.repo_root, repo_id)
    phase_started = time.time()
    index_started = time.time()
    spans, index_metadata = snapshot_spans(
        repository,
        commit,
        args.window_lines,
        args.overlap_lines,
        args.max_file_bytes,
    )
    ranker = BM25(spans)
    index_seconds = time.time() - index_started
    from transformers import AutoTokenizer
    tokenizer_started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
    )
    tokenizer_load_seconds = time.time() - tokenizer_started
    output_rows = []
    for row in selected:
        question = str(row["question"])
        output_rows.append(
            {
                "format": "doc_to_lora_broad_retrieval_v1",
                "fact_id": str(row["fact_id"]),
                "repo_id": repo_id,
                "commit_sha": commit,
                "query": question,
                "retrieval_inputs": ["question", "repo_id"],
                "gold_blind": True,
                "bundles": {
                    label: bundle(
                        ranker,
                        tokenizer,
                        question,
                        BUDGET_TOKENS[label],
                    )
                    for label in args.token_budgets
                },
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = args.output_dir / f"retrieval-{args.snapshot_index:03d}.jsonl"
    temporary = shard_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, shard_path)
    atomic_write_json(
        args.output_dir / f"retrieval-{args.snapshot_index:03d}.manifest.json",
        {
            "format": "doc_to_lora_broad_retrieval_manifest_v1",
            "repo_id": repo_id,
            "commit_sha": commit,
            "fact_count": len(output_rows),
            "index_seconds": index_seconds,
            "tokenizer_load_seconds": tokenizer_load_seconds,
            "total_phase_seconds": time.time() - phase_started,
            "window_lines": args.window_lines,
            "overlap_lines": args.overlap_lines,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "token_budgets": {
                label: BUDGET_TOKENS[label] for label in args.token_budgets
            },
            **index_metadata,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
