#!/usr/bin/env python3
"""Generate deterministic QA from SWE-fixer issues, patches, and repo snapshots.

The output is a raw JSONL pool intended for later selection and optional LLM
rewriting.  It deliberately favors high-precision facts grounded in either the
SWE-fixer patch/test patch or the repository state at the issue base commit.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import hashlib
import json
import random
import re
import subprocess
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


GENERATOR = "deterministic_swefixer_qa_v1"
TEXT_EXTS = {".py", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json"}
PY_EXTS = {".py"}
SKIP_PARTS = {
    ".git",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "site-packages",
    "_vendor",
    "vendor",
    "vendored",
    "third_party",
    "thirdparty",
    "build",
    "dist",
}
TEST_RE = re.compile(r"(^|/)(tests?|testing|test)/|(^|/|_)test_[^/]*\.py$|(^|/)test[^/]*\.py$")
DIFF_FILE_RE = re.compile(r"^diff --git a/(.*?) b/(.*)$")
HUNK_RE = re.compile(r"^@@ -(?P<old_start>\d+)(?:,(?P<old_len>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_len>\d+))? @@(?P<header>.*)$")
SYMBOL_LINE_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
IMPORT_LINE_RE = re.compile(r"^\s*(?:from\s+[\w.]+\s+import\s+.+|import\s+.+)")
CONDITION_LINE_RE = re.compile(r"^\s*(?:if|elif|while|for)\b")
EXCEPTION_LINE_RE = re.compile(r"\b(?:raise|except)\b")
ASSERT_LINE_RE = re.compile(r"\b(?:assert|pytest\.raises|self\.assert\w+)\b")
LOG_LINE_RE = re.compile(r"\b(?:log(?:ger)?\.(?:debug|info|warning|error|exception)|warnings\.warn)\b")
RETURN_LINE_RE = re.compile(r"^\s*return\b")
warnings.filterwarnings("ignore", category=SyntaxWarning)


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
    raw = "\0".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def repo_path(repo_root: Path, repo_id: str) -> Path:
    direct = repo_root / repo_id
    if direct.exists():
        return direct
    return repo_root / repo_id.replace("/", "__")


def git(repo: Path, args: list[str], *, text: bool = True, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text, timeout=timeout, check=False)


def eligible_path(path: str) -> bool:
    p = Path(path)
    parts = {part.lower() for part in p.parts}
    return not (parts & SKIP_PARTS) and p.suffix.lower() in TEXT_EXTS


def is_test_path(path: str) -> bool:
    return bool(TEST_RE.search(path))


def short_issue(problem: str) -> str:
    lines = [line.strip() for line in str(problem or "").splitlines() if line.strip()]
    if not lines:
        return "this issue"
    title = lines[0]
    return title[:180]


def compact_lines(lines: Iterable[str], *, limit: int = 8) -> str:
    cleaned = []
    for line in lines:
        text = line.strip()
        if text:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return "\n".join(cleaned)


def parse_diff(diff_text: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    current: FileDiff | None = None
    current_hunk: Hunk | None = None
    for line in str(diff_text or "").splitlines():
        m = DIFF_FILE_RE.match(line)
        if m:
            if current is not None:
                files.append(current)
            current = FileDiff(m.group(1), m.group(2), [])
            current_hunk = None
            continue
        if current is None:
            continue
        hm = HUNK_RE.match(line)
        if hm:
            current_hunk = Hunk(
                int(hm.group("old_start")),
                int(hm.group("old_len") or "1"),
                int(hm.group("new_start")),
                int(hm.group("new_len") or "1"),
                hm.group("header").strip(),
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


def all_added(fd: FileDiff) -> list[str]:
    out: list[str] = []
    for hunk in fd.hunks:
        out.extend(added_lines(hunk))
    return out


def changed_paths(files: list[FileDiff], *, tests: bool | None = None) -> list[str]:
    paths = []
    for fd in files:
        path = fd.new_path
        if tests is True and not is_test_path(path):
            continue
        if tests is False and is_test_path(path):
            continue
        paths.append(path)
    return sorted(dict.fromkeys(paths))


def context(path: str, start: int, end: int, reason: str, *, evidence: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"path": path, "start_line": max(1, start), "end_line": max(1, end), "reason": reason}
    if evidence:
        payload["evidence"] = evidence[:2000]
    return payload


def make_row(
    src: dict[str, Any],
    category: str,
    question: str,
    answer: str,
    required_context: list[dict[str, Any]],
    *,
    source: str,
    answer_style: str = "concise",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": stable_id(GENERATOR, src.get("instance_id", ""), src.get("base_commit", ""), category, question, answer),
        "repo_id": src.get("repo_id", ""),
        "base_commit": src.get("base_commit", ""),
        "commit_sha": src.get("commit_sha", src.get("base_commit", "")),
        "instance_id": src.get("instance_id", ""),
        "source_dataset": src.get("source_dataset", ""),
        "qa_source": source,
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


def patch_rows(src: dict[str, Any], *, max_rows: int) -> list[dict[str, Any]]:
    issue = short_issue(str(src.get("problem_statement") or ""))
    patch_files = parse_diff(str(src.get("target_patch") or src.get("patch") or ""))
    test_files = parse_diff(str(src.get("test_patch") or ""))
    all_files = patch_files + test_files
    out: list[dict[str, Any]] = []
    source_paths = changed_paths(patch_files, tests=False)
    test_paths = changed_paths(test_files, tests=True)
    every_path = changed_paths(all_files, tests=None)

    def add(category: str, question: str, answer: str, ctx: list[dict[str, Any]], style: str = "concise", extra: dict[str, Any] | None = None) -> None:
        if answer and len(out) < max_rows:
            out.append(make_row(src, category, question, answer, ctx, source="swefixer_patch", answer_style=style, extra=extra))

    if every_path:
        add(
            "issue_patch_changed_files",
            f"Which files does the patch for `{issue}` modify?",
            "\n".join(every_path),
            [context(path, 1, 1, "file appears in the SWE-fixer patch") for path in every_path[:8]],
            "list",
        )
    if source_paths:
        add(
            "issue_patch_source_files",
            f"Which source files are changed to address `{issue}`?",
            "\n".join(source_paths),
            [context(path, 1, 1, "source file appears in target patch") for path in source_paths[:8]],
            "list",
        )
    if test_paths:
        add(
            "issue_patch_test_files",
            f"Which test files are updated for `{issue}`?",
            "\n".join(test_paths),
            [context(path, 1, 1, "test file appears in test patch") for path in test_paths[:8]],
            "list",
        )

    for fd in all_files:
        for hunk in fd.hunks:
            plus = added_lines(hunk)
            minus = removed_lines(hunk)
            evidence = compact_lines(plus)
            if not evidence:
                continue
            hdr = hunk.header or fd.new_path
            ctx = [context(fd.new_path, hunk.new_start, hunk.new_start + max(0, hunk.new_len - 1), f"patch hunk {hdr}", evidence=evidence)]
            add(
                "patch_hunk_added_lines",
                f"What code is added in `{fd.new_path}` around `{hdr}` for `{issue}`?",
                evidence,
                ctx,
                "code",
            )
            if minus:
                add(
                    "patch_hunk_replacement",
                    f"What does the patch replace in `{fd.new_path}` around `{hdr}` for `{issue}`?",
                    "Removed:\n" + compact_lines(minus) + "\nAdded:\n" + evidence,
                    ctx,
                    "diff_summary",
                )
            imports = [line.strip() for line in plus if IMPORT_LINE_RE.match(line)]
            if imports:
                add("patch_added_imports", f"Which imports are added in `{fd.new_path}` for `{issue}`?", "\n".join(imports[:8]), ctx, "list")
            syms = [line.strip() for line in plus if SYMBOL_LINE_RE.match(line)]
            if syms:
                add("patch_added_symbols", f"Which functions or classes are introduced in `{fd.new_path}` for `{issue}`?", "\n".join(syms[:8]), ctx, "list")
            conditions = [line.strip() for line in plus if CONDITION_LINE_RE.match(line)]
            if conditions:
                add("patch_added_conditions", f"Which new conditional checks are added in `{fd.new_path}` for `{issue}`?", "\n".join(conditions[:8]), ctx, "list")
            raises = [line.strip() for line in plus if EXCEPTION_LINE_RE.search(line)]
            if raises:
                add("patch_added_exceptions", f"Which exception-handling lines are added in `{fd.new_path}` for `{issue}`?", "\n".join(raises[:8]), ctx, "list")
            returns = [line.strip() for line in plus if RETURN_LINE_RE.match(line)]
            if returns:
                add("patch_added_returns", f"Which return behavior is added in `{fd.new_path}` for `{issue}`?", "\n".join(returns[:8]), ctx, "list")
            assertions = [line.strip() for line in plus if ASSERT_LINE_RE.search(line)]
            if assertions:
                add("test_patch_assertions", f"What assertions or expected failures are added for `{issue}`?", "\n".join(assertions[:8]), ctx, "list")
            test_defs = [line.strip() for line in plus if re.match(r"^\s*(?:async\s+def|def)\s+test_", line)]
            if test_defs:
                add("test_patch_added_tests", f"Which test functions are added or updated for `{issue}`?", "\n".join(test_defs[:8]), ctx, "list")
            if len(out) >= max_rows:
                break
        if len(out) >= max_rows:
            break
    return out[:max_rows]


def list_repo_files(repo: Path, commit: str, *, max_file_bytes: int) -> list[tuple[str, int]]:
    proc = git(repo, ["ls-tree", "-r", "-l", commit], timeout=300)
    if proc.returncode != 0:
        return []
    out: list[tuple[str, int]] = []
    for line in proc.stdout.splitlines():
        try:
            meta, path = line.split("\t", 1)
            _mode, ftype, _blob, size_text = meta.split()
            size = int(size_text)
        except ValueError:
            continue
        if ftype == "blob" and 0 < size <= max_file_bytes and eligible_path(path):
            out.append((path, size))
    return out


def read_file(repo: Path, commit: str, path: str, *, max_file_bytes: int) -> str | None:
    proc = git(repo, ["show", f"{commit}:{path}"], text=False, timeout=120)
    if proc.returncode != 0 or len(proc.stdout) > max_file_bytes:
        return None
    try:
        return proc.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def func_context(stack: list[str]) -> str:
    return ".".join(stack) if stack else "<module>"


class RepoVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.stack: list[str] = []
        self.imports: list[tuple[int, str, str]] = []
        self.calls: list[tuple[int, str, str]] = []
        self.raises: list[tuple[int, str, str]] = []
        self.logs: list[tuple[int, str, str]] = []
        self.tests: list[tuple[int, str, list[str]]] = []
        self.schemas: list[tuple[int, str, list[str]]] = []
        self.routes: list[tuple[int, str, str]] = []
        self.cli_args: list[tuple[int, str]] = []

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            self.imports.append((node.lineno, alias.name, ast.unparse(node)))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        mod = "." * int(node.level or 0) + (node.module or "")
        self.imports.append((node.lineno, mod, ast.unparse(node)))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        test_asserts: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                test_asserts.append(ast.unparse(child.test))
            elif isinstance(child, ast.Call) and ASSERT_LINE_RE.search(ast.unparse(child.func)):
                test_asserts.append(ast.unparse(child))
        if node.name.startswith("test_") or is_test_path(self.path):
            if test_asserts:
                self.tests.append((node.lineno, node.name, test_asserts[:5]))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        bases = [ast.unparse(base) for base in node.bases]
        decorators = [ast.unparse(dec) for dec in node.decorator_list]
        fields = []
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                fields.append(f"{child.target.id}: {ast.unparse(child.annotation)}")
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        fields.append(target.id)
        schema_like = (
            any(base.endswith(("BaseModel", "TypedDict", "Model", "Serializer")) or base in {"dataclass"} for base in bases)
            or any("dataclass" in dec or "attrs" in dec for dec in decorators)
        )
        if schema_like and fields:
            self.schemas.append((node.lineno, node.name, fields[:12]))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> Any:
        name = call_name(node.func)
        if name:
            self.calls.append((node.lineno, name, func_context(self.stack)))
        rendered = ast.unparse(node)
        if name.endswith(".add_argument") and node.args:
            self.cli_args.append((node.lineno, rendered[:300]))
        if any(name.endswith(suffix) for suffix in [".debug", ".info", ".warning", ".error", ".exception"]) or name == "warnings.warn":
            self.logs.append((node.lineno, name, rendered[:300]))
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> Any:
        self.raises.append((node.lineno, func_context(self.stack), ast.unparse(node)[:300]))
        self.generic_visit(node)


def route_decorators(tree: ast.Module, path: str) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                text = ast.unparse(dec)
                if re.search(r"\.(?:get|post|put|patch|delete|route)\(", text):
                    out.append((node.lineno, node.name, text[:300]))
    return out


def repo_rows(src: dict[str, Any], repo_root: Path, *, max_rows: int, max_files: int, max_file_bytes: int, seed: int) -> list[dict[str, Any]]:
    repo = repo_path(repo_root, str(src.get("repo_id", "")))
    commit = str(src.get("base_commit") or src.get("commit_sha") or "")
    if not repo.exists() or not commit:
        return []
    files = list_repo_files(repo, commit, max_file_bytes=max_file_bytes)
    if not files:
        return []
    rng = random.Random(int(hashlib.sha1(f"{src.get('repo_id')}:{commit}:{seed}".encode()).hexdigest()[:8], 16))
    rng.shuffle(files)
    files = files[:max_files]

    visitors: list[RepoVisitor] = []
    texts: dict[str, str] = {}
    config_findings: list[tuple[str, int, str, str]] = []
    for path, _size in files:
        text = read_file(repo, commit, path, max_file_bytes=max_file_bytes)
        if text is None:
            continue
        texts[path] = text
        suffix = Path(path).suffix.lower()
        if suffix in PY_EXTS:
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError):
                continue
            visitor = RepoVisitor(path)
            visitor.visit(tree)
            visitor.routes.extend(route_decorators(tree, path))
            visitors.append(visitor)
        elif Path(path).name in {"pyproject.toml", "setup.cfg", "tox.ini"} or suffix in {".cfg", ".ini", ".toml", ".yaml", ".yml"}:
            for idx, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if stripped and re.search(r"(entry_points|console_scripts|scripts|dependencies|tool\.|pytest|flake8|mypy)", stripped):
                    config_findings.append((path, idx, stripped[:240], Path(path).name))
                    if len(config_findings) >= 16:
                        break

    out: list[dict[str, Any]] = []

    def add(category: str, question: str, answer: str, path: str, line: int, reason: str, style: str = "concise") -> None:
        if answer and len(out) < max_rows:
            out.append(
                make_row(
                    src,
                    category,
                    question,
                    answer,
                    [context(path, line, line, reason)],
                    source="repo_static_scan",
                    answer_style=style,
                )
            )

    imports_by_mod: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    calls_by_name: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for v in visitors:
        for line, mod, stmt in v.imports:
            imports_by_mod[mod].append((v.path, line, stmt))
        for line, name, where in v.calls:
            calls_by_name[name].append((v.path, line, where))

        for line, where, stmt in v.raises[:4]:
            add("repo_exception_paths", f"Where can `{stmt}` be raised in `{src.get('repo_id')}`?", f"{v.path}:{line} in {where}", v.path, line, "raise statement in repository code")
        for line, name, stmt in v.logs[:4]:
            add("repo_logging_paths", f"Where does the repository call `{name}`?", f"{v.path}:{line} -> {stmt}", v.path, line, "logging or warning call")
        for line, test_name, asserts in v.tests[:3]:
            add("repo_test_assertions", f"What does test `{test_name}` assert in `{v.path}`?", "\n".join(asserts), v.path, line, "test body assertion")
        for line, cls, fields in v.schemas[:3]:
            add("repo_schema_fields", f"Which fields are declared on schema/model `{cls}` in `{v.path}`?", "\n".join(fields), v.path, line, "schema-like class declaration", "list")
        for line, func, dec in v.routes[:3]:
            add("repo_route_handlers", f"Which route decorator is attached to `{func}` in `{v.path}`?", dec, v.path, line, "route decorator")
        for line, arg in v.cli_args[:3]:
            add("repo_cli_arguments", f"Which CLI argument is registered in `{v.path}`?", arg, v.path, line, "argparse add_argument call")
        if len(out) >= max_rows:
            return out[:max_rows]

    for mod, hits in list(imports_by_mod.items())[:12]:
        if len(hits) >= 2:
            answer = "\n".join(f"{path}:{line} -> {stmt}" for path, line, stmt in hits[:8])
            path, line, _stmt = hits[0]
            add("repo_import_graph", f"Which files import `{mod}` in `{src.get('repo_id')}`?", answer, path, line, "import graph evidence", "list")
    for name, hits in list(calls_by_name.items())[:20]:
        if len(hits) >= 2 and "." not in name:
            answer = "\n".join(f"{path}:{line} in {where}" for path, line, where in hits[:8])
            path, line, _where = hits[0]
            add("repo_call_sites", f"Where is `{name}` called in `{src.get('repo_id')}`?", answer, path, line, "call-site evidence", "list")
    for path, line, finding, fname in config_findings[:12]:
        add("repo_config_entrypoints", f"What repository configuration appears in `{fname}`?", finding, path, line, "configuration file line")
    return out[:max_rows]


def iter_rows(path: Path, columns: list[str]) -> Iterable[dict[str, Any]]:
    table = pq.read_table(path, columns=columns)
    for row in table.to_pylist():
        yield row


def load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(row.get("id") or "")
            if rid:
                seen.add(rid)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-metadata", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--families", default="patch,repo", help="comma-separated: patch,repo")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--target-rows", type=int, default=0)
    parser.add_argument("--qa-rows-per-issue", type=int, default=24)
    parser.add_argument("--repo-rows-per-snapshot", type=int, default=24)
    parser.add_argument("--repo-scan-every", type=int, default=1)
    parser.add_argument("--max-files-per-snapshot", type=int, default=32)
    parser.add_argument("--max-file-bytes", type=int, default=524288)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    families = {item.strip() for item in args.families.split(",") if item.strip()}
    unknown = families - {"patch", "repo"}
    if unknown:
        raise SystemExit(f"unknown families: {sorted(unknown)}")

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
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(args.output_jsonl) if args.resume else set()
    mode = "a" if args.resume else "w"
    counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    started = time.time()
    emitted = 0
    processed = 0

    with args.output_jsonl.open(mode, encoding="utf-8") as out_fh:
        for row in iter_rows(args.input_metadata, columns):
            idx = int(row.get("row_index") or 0)
            if args.num_shards > 1 and idx % args.num_shards != args.shard_index:
                continue
            if row.get("usable_for_train") is False:
                counts["unusable_skipped"] += 1
                continue
            processed += 1
            candidates: list[dict[str, Any]] = []
            if "patch" in families:
                candidates.extend(patch_rows(row, max_rows=args.qa_rows_per_issue))
            if "repo" in families and (processed - 1) % max(1, args.repo_scan_every) == 0:
                candidates.extend(
                    repo_rows(
                        row,
                        args.repo_root,
                        max_rows=args.repo_rows_per_snapshot,
                        max_files=args.max_files_per_snapshot,
                        max_file_bytes=args.max_file_bytes,
                        seed=args.seed,
                    )
                )
            for candidate in candidates:
                if candidate["id"] in seen:
                    counts["seen_skipped"] += 1
                    continue
                out_fh.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")
                seen.add(candidate["id"])
                counts["generated"] += 1
                category_counts[str(candidate["task_category"])] += 1
                emitted += 1
                if args.target_rows and emitted >= args.target_rows:
                    break
            if args.progress_every and processed % args.progress_every == 0:
                print(json.dumps({"processed": processed, "generated": emitted, "elapsed_sec": round(time.time() - started, 1)}), flush=True)
            if args.target_rows and emitted >= args.target_rows:
                break

    audit = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "counts": dict(counts),
        "category_counts": dict(category_counts),
        "processed_rows": processed,
        "generated": emitted,
        "elapsed_sec": round(time.time() - started, 3),
        "generator": GENERATOR,
        "output_jsonl": str(args.output_jsonl),
    }
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
