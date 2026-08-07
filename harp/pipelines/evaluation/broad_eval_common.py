#!/usr/bin/env python3
"""Shared utilities for the paper-grade Doc-to-LoRA broad evaluation.

The module intentionally depends only on the Python standard library so that
dataset validation and scoring can run on CPU-only Slurm nodes without loading
the model environment.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Sequence


CHOICE_LABELS = ("A", "B", "C", "D")
TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|testing)(/|$)|(^|/)(test_[^/]+|[^/]+_test)\.py$",
    re.IGNORECASE,
)
DIFF_PATH_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
HUNK_RE = re.compile(r"^@@[^@]*@@\s*(.*)$", re.MULTILINE)


def stable_hex(*parts: Any, length: int = 20) -> str:
    raw = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def stable_int(*parts: Any) -> int:
    return int(stable_hex(*parts, length=16), 16)


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip().strip("\"'`")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def git(
    repo: Path,
    arguments: Sequence[str],
    *,
    text: bool = True,
    timeout: int = 300,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=text,
        errors="replace" if text else None,
        timeout=timeout,
    )


def repo_path(repo_root: Path, repo_id: str) -> Path:
    owner, name = repo_id.split("/", 1)
    nested = repo_root / owner / name
    if (nested / ".git").exists():
        return nested
    slugged = repo_root / repo_id.replace("/", "__")
    if (slugged / ".git").exists():
        return slugged
    return nested


def git_file_exists(repo: Path, commit: str, path: str) -> bool:
    result = git(repo, ["cat-file", "-e", f"{commit}:{path}"])
    return result.returncode == 0


def git_show(repo: Path, commit: str, path: str, *, max_bytes: int = 4_000_000) -> str | None:
    result = git(repo, ["show", f"{commit}:{path}"], text=False)
    if result.returncode != 0 or len(result.stdout) > max_bytes:
        return None
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def list_repo_paths(
    repo: Path,
    commit: str,
    *,
    suffixes: set[str] | None = None,
    include_tests: bool = True,
) -> list[str]:
    result = git(repo, ["ls-tree", "-r", "--name-only", commit])
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    paths = []
    for raw in result.stdout.splitlines():
        path = raw.strip()
        if not path:
            continue
        if suffixes is not None and PurePosixPath(path).suffix.lower() not in suffixes:
            continue
        if not include_tests and is_test_path(path):
            continue
        paths.append(path)
    return sorted(paths)


def is_test_path(path: str) -> bool:
    return bool(TEST_PATH_RE.search(path))


def parse_diff_paths(diff_text: str) -> list[str]:
    paths = [match.group(1) for match in DIFF_PATH_RE.finditer(diff_text or "")]
    if not paths:
        paths = [match.group(2) for match in DIFF_FILE_RE.finditer(diff_text or "")]
    return list(dict.fromkeys(path for path in paths if path != "/dev/null"))


def parse_hunk_symbols(diff_text: str) -> list[str]:
    symbols: list[str] = []
    for match in HUNK_RE.finditer(diff_text or ""):
        header = match.group(1).strip()
        candidates = re.findall(
            r"(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)|"
            r"([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            header,
        )
        for pair in candidates:
            value = next((item for item in pair if item), "")
            if value and value not in symbols:
                symbols.append(value)
    return symbols


def render_signature(node: ast.AST) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    try:
        arguments = ast.unparse(node.args)
    except (AttributeError, ValueError):
        return ""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({arguments}):"


def source_segment_line(text: str, line: int, end_line: int | None = None) -> str:
    lines = text.splitlines()
    start = max(1, line)
    end = min(len(lines), end_line or line)
    return "\n".join(lines[start - 1 : end])


@dataclass(frozen=True)
class Fact:
    repo_id: str
    commit_sha: str
    family: str
    subtype: str
    question: str
    answer: str
    answer_type: str
    evidence_path: str
    start_line: int
    end_line: int
    negative: bool = False
    metadata: dict[str, Any] | None = None

    @property
    def fact_id(self) -> str:
        return stable_hex(
            "broad_eval_fact_v1",
            self.repo_id,
            self.commit_sha,
            self.family,
            self.subtype,
            self.question,
            self.answer,
        )

    def as_row(self, split: str) -> dict[str, Any]:
        return {
            "format": "doc_to_lora_broad_eval_fact_v1",
            "benchmark_id": self.fact_id,
            "fact_id": self.fact_id,
            "split": split,
            "repo_id": self.repo_id,
            "repo": self.repo_id,
            "commit_sha": self.commit_sha,
            "base_commit": self.commit_sha,
            "family": self.family,
            "subtype": self.subtype,
            "question": self.question,
            "gold_answer": self.answer,
            "answer_type": self.answer_type,
            "negative": self.negative,
            "evidence_paths": [self.evidence_path] if self.evidence_path else [],
            "evidence_spans": (
                [
                    {
                        "path": self.evidence_path,
                        "start_line": self.start_line,
                        "end_line": self.end_line,
                    }
                ]
                if self.evidence_path
                else []
            ),
            "metadata": self.metadata or {},
        }


def choose_hard_distractors(
    *,
    gold: str,
    candidates: Iterable[str],
    answer_type: str,
    seed_parts: Sequence[Any],
    count: int = 3,
) -> list[str]:
    """Choose deterministic, format-matched distractors.

    For paths, candidates at the same depth and with the same suffix are
    preferred. Other answer types prefer similar character length.
    """
    gold_norm = normalize_answer(gold)
    unique: dict[str, str] = {}
    for candidate in candidates:
        norm = normalize_answer(candidate)
        if not norm or norm == gold_norm:
            continue
        unique.setdefault(norm, str(candidate))
    values = list(unique.values())
    if answer_type in {"path", "repo_relative_path"}:
        gold_path = PurePosixPath(gold)
        gold_depth = len(gold_path.parts)
        gold_suffix = gold_path.suffix
        gold_parent = gold_path.parent.parts

        def key(value: str) -> tuple[int, int, int, int, str]:
            path = PurePosixPath(value)
            parent = path.parent.parts
            common_parent = 0
            for left, right in zip(gold_parent, parent):
                if left != right:
                    break
                common_parent += 1
            return (
                int(path.suffix != gold_suffix),
                -common_parent,
                abs(len(path.parts) - gold_depth),
                abs(len(value) - len(gold)),
                value,
            )

    else:

        def key(value: str) -> tuple[int, int, str]:
            same_shape = int(
                (value.startswith("def ") != gold.startswith("def "))
                or (value.startswith("raise ") != gold.startswith("raise "))
                or (value.startswith("return ") != gold.startswith("return "))
            )
            return (same_shape, abs(len(value) - len(gold)), value)

    ordered = sorted(values, key=key)
    if len(ordered) < count:
        raise ValueError(
            f"Need {count} distractors for {answer_type} gold={gold!r}; "
            f"found {len(ordered)}"
        )
    # Randomize only within the strongest candidate band, preserving hardness.
    band = ordered[: max(count, min(len(ordered), count * 4))]
    rng = random.Random(stable_int(*seed_parts))
    rng.shuffle(band)
    return band[:count]


def attach_mcq(
    row: dict[str, Any],
    *,
    distractors: Sequence[str],
    position_index: int,
    add_permutation: bool,
) -> dict[str, Any]:
    if len(distractors) != 3:
        raise ValueError("Exactly three MCQ distractors are required")
    gold = str(row["gold_answer"])
    if any(normalize_answer(value) == normalize_answer(gold) for value in distractors):
        raise ValueError("MCQ distractor duplicates the gold answer")
    if len({normalize_answer(value) for value in distractors}) != 3:
        raise ValueError("MCQ distractors are not unique")

    correct_index = position_index % len(CHOICE_LABELS)
    choices: list[str] = []
    distractor_iter = iter(distractors)
    for index in range(len(CHOICE_LABELS)):
        choices.append(gold if index == correct_index else next(distractor_iter))
    output = dict(row)
    output["mcq"] = {
        "kind": "single_choice",
        "labels": list(CHOICE_LABELS),
        "choices": choices,
        "correct_label": CHOICE_LABELS[correct_index],
        "correct_index": correct_index,
    }
    if add_permutation:
        shift = 1 + stable_int(row["fact_id"], "permutation") % 3
        permuted = choices[shift:] + choices[:shift]
        permuted_index = permuted.index(gold)
        output["mcq_permutation"] = {
            "kind": "single_choice",
            "labels": list(CHOICE_LABELS),
            "choices": permuted,
            "correct_label": CHOICE_LABELS[permuted_index],
            "correct_index": permuted_index,
        }
    else:
        output["mcq_permutation"] = None
    return output


def token_f1(prediction: str, gold: str) -> float:
    predicted = normalize_answer(prediction).split()
    target = normalize_answer(gold).split()
    if not predicted or not target:
        return float(predicted == target)
    overlap = sum((Counter(predicted) & Counter(target)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(target)
    return 2.0 * precision * recall / (precision + recall)


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def extract_mcq_label(text: str) -> str:
    match = re.search(r"\b([ABCD])\b", str(text).strip().upper())
    return match.group(1) if match else ""
