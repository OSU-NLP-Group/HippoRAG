#!/usr/bin/env python3
"""Generate deterministic deep-surface QA for Code2LoRA.

This generator targets coverage gaps left by the earlier AST, SWE-fixer, and
diverse repo-coverage waves:

* multi-hop symbol/test/config relationships
* patch-teaching facts from SWE-fixer diffs
* runtime/config/entrypoint surfaces
* test-as-spec details such as fixtures, parametrization, and mocks
* control-flow microfacts linking conditions to returns/raises

It is intentionally CPU-only and writes JSONL rows with explicit provenance and
multi-span context where possible.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import subprocess
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


GENERATOR = "deterministic_deep_surface_qa_v1"
warnings.filterwarnings("ignore", category=SyntaxWarning)
PY_EXT = ".py"
CONFIG_NAMES = {"pyproject.toml", "setup.cfg", "setup.py", "tox.ini", "pytest.ini", "mypy.ini", "ruff.toml"}
CONFIG_EXTS = {".toml", ".cfg", ".ini", ".yaml", ".yml", ".json"}
TEST_RE = re.compile(r"(^|/)(tests?|testing|test)/|(^|/|_)test_[^/]*\.py$|(^|/)test[^/]*\.py$")
DIFF_FILE_RE = re.compile(r"^diff --git a/(.*?) b/(.*)$")
HUNK_RE = re.compile(r"^@@ -(?P<old_start>\d+)(?:,(?P<old_len>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_len>\d+))? @@(?P<header>.*)$")
SYMBOL_LINE_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
IMPORT_LINE_RE = re.compile(r"^\s*(?:from\s+[\w.]+\s+import\s+.+|import\s+.+)")
CONDITION_LINE_RE = re.compile(r"^\s*(?:if|elif|while)\b")
ASSERT_LINE_RE = re.compile(r"\b(?:assert|pytest\.raises|self\.assert\w+)\b")
RETURN_LINE_RE = re.compile(r"^\s*return\b")
EXCEPTION_LINE_RE = re.compile(r"\b(?:raise|except)\b")


@dataclass
class Symbol:
    kind: str
    name: str
    qualified_name: str
    path: str
    start_line: int
    end_line: int
    signature: str = ""
    parent: str = ""
    bases: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)


@dataclass
class ImportFact:
    path: str
    line: int
    statement: str
    module: str
    name: str
    asname: str


@dataclass
class CallFact:
    path: str
    line: int
    name: str
    scope: str
    statement: str


@dataclass
class TestFact:
    path: str
    line: int
    name: str
    args: list[str]
    assertions: list[str]
    calls: list[str]
    decorators: list[str]
    mocks: list[str]


@dataclass
class ControlFact:
    path: str
    line: int
    scope: str
    condition: str
    consequence: str
    kind: str


@dataclass
class FileFacts:
    path: str
    text: str
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[ImportFact] = field(default_factory=list)
    calls: list[CallFact] = field(default_factory=list)
    tests: list[TestFact] = field(default_factory=list)
    controls: list[ControlFact] = field(default_factory=list)
    env_vars: list[tuple[int, str, str]] = field(default_factory=list)
    configs: list[tuple[int, str, str]] = field(default_factory=list)
    exports: list[tuple[int, list[str]]] = field(default_factory=list)
    route_decorators: list[tuple[int, str, str]] = field(default_factory=list)
    cli_args: list[tuple[int, str, str]] = field(default_factory=list)


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
    return hashlib.sha1("\0".join(str(p) for p in parts).encode("utf-8", errors="replace")).hexdigest()


def short(text: Any, limit: int = 360) -> str:
    return " ".join(str(text or "").split())[:limit].rstrip()


def is_test_path(path: str) -> bool:
    return bool(TEST_RE.search(path))


def repo_path(repo_root: Path, repo_id: str) -> Path:
    direct = repo_root / repo_id
    if direct.exists():
        return direct
    return repo_root / repo_id.replace("/", "__")


def git(repo: Path, args: list[str], *, text: bool = True, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text, timeout=timeout, check=False)


def read_file(repo: Path, commit: str, path: str, max_file_bytes: int) -> str | None:
    proc = git(repo, ["show", f"{commit}:{path}"], text=False, timeout=120)
    if proc.returncode != 0 or len(proc.stdout) > max_file_bytes:
        return None
    try:
        return proc.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def module_name(path: str) -> str:
    p = path[:-3] if path.endswith(".py") else path
    if p.endswith("/__init__"):
        p = p[: -len("/__init__")]
    return p.replace("/", ".")


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def source_signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        try:
            args = ast.unparse(node.args)
            returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
            return f"{prefix} {node.name}({args}){returns}"
        except Exception:
            return f"{prefix} {node.name}(...)"
    return ""


def arg_names(args: ast.arguments) -> list[str]:
    out = []
    for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
        out.append(arg.arg)
    if args.vararg:
        out.append(args.vararg.arg)
    if args.kwarg:
        out.append(args.kwarg.arg)
    return out


def literal_string_list(node: ast.AST) -> list[str]:
    try:
        value = ast.literal_eval(node)
    except Exception:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if isinstance(item, str)]
    return []


def ctx(path: str, start: int, end: int, reason: str, evidence: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"path": path, "start_line": max(1, int(start)), "end_line": max(1, int(end)), "reason": reason}
    if evidence:
        payload["evidence"] = evidence[:2000]
    return payload


def make_row(snapshot: dict[str, Any], category: str, question: str, answer: str, contexts: list[dict[str, Any]], style: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "id": stable_id(GENERATOR, snapshot.get("repo_id", ""), snapshot.get("base_commit", ""), snapshot.get("instance_id", ""), category, question, answer),
        "repo_id": snapshot.get("repo_id", ""),
        "base_commit": snapshot.get("base_commit", ""),
        "commit_sha": snapshot.get("commit_sha", snapshot.get("base_commit", "")),
        "source_instance_id": snapshot.get("instance_id", ""),
        "source_dataset": snapshot.get("source_dataset", ""),
        "qa_source": "deep_surface_scan",
        "task_category": category,
        "question": question,
        "answer": answer,
        "answer_style": style,
        "required_context": contexts,
        "generator": GENERATOR,
    }
    if extra:
        payload.update(extra)
    return payload


class Visitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.stack: list[str] = []
        self.symbols: list[Symbol] = []
        self.imports: list[ImportFact] = []
        self.calls: list[CallFact] = []
        self.tests: list[TestFact] = []
        self.controls: list[ControlFact] = []
        self.env_vars: list[tuple[int, str, str]] = []
        self.exports: list[tuple[int, list[str]]] = []
        self.route_decorators: list[tuple[int, str, str]] = []
        self.cli_args: list[tuple[int, str, str]] = []

    def scope(self) -> str:
        return ".".join(self.stack) if self.stack else "<module>"

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            self.imports.append(ImportFact(self.path, node.lineno, ast.unparse(node), alias.name, "", alias.asname or ""))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        mod = "." * int(node.level or 0) + (node.module or "")
        for alias in node.names:
            self.imports.append(ImportFact(self.path, node.lineno, ast.unparse(node), mod, alias.name, alias.asname or ""))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        mod = module_name(self.path)
        parent = self.stack[-1] if self.stack else ""
        qn = f"{mod}.{parent}.{node.name}" if parent else f"{mod}.{node.name}"
        decorators = [short(ast.unparse(dec), 220) for dec in node.decorator_list]
        self.symbols.append(
            Symbol(
                "method" if parent else "function",
                node.name,
                qn,
                self.path,
                node.lineno,
                getattr(node, "end_lineno", node.lineno),
                source_signature(node),
                parent,
                [],
                decorators,
                arg_names(node.args),
            )
        )
        calls, assertions, mocks = [], [], []
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                assertions.append(short(ast.unparse(child.test), 260))
            elif isinstance(child, ast.Call):
                cname = call_name(child.func)
                if cname:
                    calls.append(cname)
                    rendered = short(ast.unparse(child), 320)
                    if ASSERT_LINE_RE.search(rendered):
                        assertions.append(rendered)
                    if any(part in rendered for part in ("monkeypatch.", "mock.", "Mock(", "patch(", "patch.object(")):
                        mocks.append(rendered)
        if node.name.startswith("test_") or is_test_path(self.path):
            self.tests.append(TestFact(self.path, node.lineno, node.name, arg_names(node.args), assertions[:8], calls[:16], decorators[:8], mocks[:8]))
        for dec in decorators:
            if re.search(r"\.(?:get|post|put|patch|delete|route)\(", dec) or dec.startswith(("pytest.fixture", "fixture", "click.", "app.")):
                self.route_decorators.append((node.lineno, node.name, dec))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        mod = module_name(self.path)
        bases = [short(ast.unparse(base), 160) for base in node.bases]
        decorators = [short(ast.unparse(dec), 220) for dec in node.decorator_list]
        self.symbols.append(Symbol("class", node.name, f"{mod}.{node.name}", self.path, node.lineno, getattr(node, "end_lineno", node.lineno), "", "", bases, decorators, []))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Assign(self, node: ast.Assign) -> Any:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                values = literal_string_list(node.value)
                if values:
                    self.exports.append((node.lineno, values[:30]))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        name = call_name(node.func)
        rendered = short(ast.unparse(node), 360)
        if name:
            self.calls.append(CallFact(self.path, node.lineno, name, self.scope(), rendered))
            if name.endswith(".add_argument"):
                self.cli_args.append((node.lineno, name, rendered))
            if name in {"os.getenv", "getenv"} or name.startswith("os.environ."):
                self.env_vars.append((node.lineno, name, rendered))
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> Any:
        cond = short(ast.unparse(node.test), 260)
        for child in node.body[:4]:
            if isinstance(child, ast.Raise):
                self.controls.append(ControlFact(self.path, node.lineno, self.scope(), cond, short(ast.unparse(child), 320), "raise"))
            elif isinstance(child, ast.Return):
                self.controls.append(ControlFact(self.path, node.lineno, self.scope(), cond, short(ast.unparse(child), 320), "return"))
        self.generic_visit(node)


def parse_file(path: str, text: str) -> FileFacts:
    facts = FileFacts(path, text)
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if Path(path).name in CONFIG_NAMES or Path(path).suffix.lower() in CONFIG_EXTS:
            if stripped and re.search(r"(entry_points|console_scripts|scripts|dependencies|pytest|mypy|flake8|ruff|tool\.|extras_require|env|DJANGO|FLASK|FASTAPI|settings|DATABASE|INSTALLED_APPS)", stripped):
                facts.configs.append((line_no, short(stripped, 260), Path(path).name))
    if not path.endswith(PY_EXT):
        return facts
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return facts
    visitor = Visitor(path)
    visitor.visit(tree)
    facts.symbols = visitor.symbols
    facts.imports = visitor.imports
    facts.calls = visitor.calls
    facts.tests = visitor.tests
    facts.controls = visitor.controls
    facts.env_vars = visitor.env_vars
    facts.exports = visitor.exports
    facts.route_decorators = visitor.route_decorators
    facts.cli_args = visitor.cli_args
    return facts


def resolve_import(current_path: str, raw_module: str, known_paths: set[str]) -> str:
    level = len(raw_module) - len(raw_module.lstrip("."))
    module = raw_module.lstrip(".")
    current_parts = module_name(current_path).split(".")[:-1]
    if level:
        keep = max(0, len(current_parts) - level + 1)
        parts = current_parts[:keep] + ([p for p in module.split(".") if p] if module else [])
    else:
        parts = [p for p in module.split(".") if p]
    if not parts:
        return ""
    stem = "/".join(parts)
    for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
        if candidate in known_paths:
            return candidate
    return ""


def shuffled(items: Iterable[Any], rng: random.Random) -> list[Any]:
    out = list(items)
    rng.shuffle(out)
    return out


def static_rows_for_snapshot(snapshot: dict[str, Any], repo_root: Path, max_files: int, max_file_bytes: int, max_rows: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    repo_id = str(snapshot.get("repo_id") or "")
    commit = str(snapshot.get("base_commit") or snapshot.get("commit_sha") or "")
    repo = repo_path(repo_root, repo_id)
    if not repo.exists() or not commit:
        return [], {"missing_repo": 1}
    try:
        metadata = json.loads(str(snapshot.get("embedding_metadata_json") or "{}"))
    except json.JSONDecodeError:
        metadata = {}
    files = [str(p) for p in metadata.get("files_used") or [] if isinstance(p, str)]
    if not files:
        proc = git(repo, ["ls-tree", "-r", "--name-only", commit], timeout=180)
        files = [line.strip() for line in proc.stdout.splitlines() if line.strip()] if proc.returncode == 0 else []
    files = [p for p in files if p.endswith(PY_EXT) or Path(p).name in CONFIG_NAMES or Path(p).suffix.lower() in CONFIG_EXTS]
    rng = random.Random(int(stable_id(repo_id, commit, seed)[:8], 16))
    files = shuffled(sorted(dict.fromkeys(files)), rng)[:max_files]

    facts_by_path: dict[str, FileFacts] = {}
    for path in files:
        text = read_file(repo, commit, path, max_file_bytes)
        if text is not None:
            facts_by_path[path] = parse_file(path, text)
    known_paths = set(facts_by_path)
    symbols = [sym for facts in facts_by_path.values() for sym in facts.symbols]
    calls = [call for facts in facts_by_path.values() for call in facts.calls]
    tests = [test for facts in facts_by_path.values() for test in facts.tests]

    symbols_by_name: dict[str, list[Symbol]] = defaultdict(list)
    symbols_by_qn: dict[str, Symbol] = {}
    for sym in symbols:
        symbols_by_name[sym.name].append(sym)
        symbols_by_qn[sym.qualified_name] = sym

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    pool_counts: Counter[str] = Counter()
    pool_limit = max_rows * 8
    per_category_cap = max(64, max_rows)

    def add(row: dict[str, Any]) -> None:
        cat = str(row.get("task_category") or "unknown")
        if len(out) >= pool_limit or pool_counts[cat] >= per_category_cap:
            return
        rid = str(row["id"])
        if rid in seen:
            return
        seen.add(rid)
        pool_counts[cat] += 1
        out.append(row)

    # Multi-hop call chains: call site -> unique local definition -> unique call from that definition.
    for call in shuffled(calls, rng):
        simple = call.name.split(".")[-1]
        candidates = [sym for sym in symbols_by_name.get(simple, []) if sym.path != call.path]
        if len(candidates) != 1:
            continue
        mid = candidates[0]
        inner_calls = [c for c in calls if c.path == mid.path and mid.start_line <= c.line <= mid.end_line]
        rng.shuffle(inner_calls)
        for inner in inner_calls:
            leaf_name = inner.name.split(".")[-1]
            leafs = [sym for sym in symbols_by_name.get(leaf_name, []) if sym.qualified_name != mid.qualified_name]
            if len(leafs) != 1:
                continue
            leaf = leafs[0]
            add(make_row(
                snapshot,
                "multihop_call_chain_to_definition",
                f"In `{call.path}`, the call/name `{call.name}` links to `{mid.qualified_name}`; which repository symbol is then called inside that definition via `{inner.name}`?",
                f"{leaf.qualified_name} in {leaf.path}",
                [ctx(call.path, call.line, call.line, "outer call/name occurrence"), ctx(mid.path, mid.start_line, mid.end_line, "intermediate symbol definition"), ctx(leaf.path, leaf.start_line, leaf.end_line, "second-hop symbol definition")],
                "symbol_location",
            ))
            break

    # Test-as-spec: fixtures, parametrization, mocks, and linked implementation symbols.
    fixture_defs = {sym.name: sym for sym in symbols if any("fixture" in dec for dec in sym.decorators)}
    for test in shuffled(tests, rng):
        fixture_hits = [fixture_defs[arg] for arg in test.args if arg in fixture_defs and fixture_defs[arg].path != test.path]
        if fixture_hits:
            answer = "\n".join(f"{sym.qualified_name} in {sym.path}" for sym in fixture_hits[:5])
            add(make_row(
                snapshot,
                "test_fixture_dependency",
                f"Which fixture definitions does test `{test.name}` in `{test.path}` depend on?",
                answer,
                [ctx(test.path, test.line, test.line, "test function using fixture argument")] + [ctx(sym.path, sym.start_line, sym.end_line, "fixture definition") for sym in fixture_hits[:3]],
                "list",
            ))
        param_decs = [dec for dec in test.decorators if "parametrize" in dec]
        if param_decs:
            add(make_row(snapshot, "test_parametrize_cases", f"Which parametrization decorators shape test `{test.name}` in `{test.path}`?", "\n".join(param_decs[:4]), [ctx(test.path, test.line, test.line, "parametrized test definition")], "list"))
        if test.mocks:
            add(make_row(snapshot, "test_mock_or_monkeypatch_behavior", f"What mock or monkeypatch behavior appears in test `{test.name}` in `{test.path}`?", "\n".join(test.mocks[:5]), [ctx(test.path, test.line, test.line, "test mock or monkeypatch evidence")], "list"))
        linked = []
        for cname in test.calls:
            linked.extend([sym for sym in symbols_by_name.get(cname.split(".")[-1], []) if not is_test_path(sym.path)])
        unique = []
        seen_qn = set()
        for sym in linked:
            if sym.qualified_name not in seen_qn:
                unique.append(sym)
                seen_qn.add(sym.qualified_name)
        if unique and test.assertions:
            add(make_row(
                snapshot,
                "test_assertion_to_implementation_surface",
                f"Which implementation surfaces are tested by the assertions in `{test.name}` from `{test.path}`?",
                "\n".join(f"{sym.qualified_name} in {sym.path}" for sym in unique[:6]),
                [ctx(test.path, test.line, test.line, "asserting test function")] + [ctx(sym.path, sym.start_line, sym.end_line, "implementation symbol referenced by test") for sym in unique[:3]],
                "list",
            ))

    # Runtime, config, decorators, exports, and control-flow microfacts.
    for facts in shuffled(facts_by_path.values(), rng):
        for line, name, rendered in facts.env_vars[:6]:
            add(make_row(snapshot, "runtime_environment_variable_usage", f"Where does `{repo_id}` read an environment variable or environment mapping?", f"{facts.path}:{line} -> {rendered}", [ctx(facts.path, line, line, "environment-variable access")], "location"))
        for line, _name, rendered in facts.cli_args[:6]:
            add(make_row(snapshot, "runtime_cli_argument_surface", f"Which CLI argument registration appears in `{facts.path}`?", rendered, [ctx(facts.path, line, line, "CLI argument registration")], "code"))
        for line, func, dec in facts.route_decorators[:6]:
            add(make_row(snapshot, "runtime_decorator_registration", f"Which decorator registers or marks `{func}` in `{facts.path}`?", dec, [ctx(facts.path, line, line, "decorated callable")], "code"))
        for line, config, fname in facts.configs[:8]:
            add(make_row(snapshot, "runtime_config_setting_surface", f"What runtime/tooling configuration from `{fname}` is present in `{facts.path}`?", config, [ctx(facts.path, line, line, "configuration evidence")], "config"))
        for line, names in facts.exports[:4]:
            resolved = [sym for name in names for sym in symbols_by_name.get(name, []) if sym.path == facts.path]
            if resolved:
                add(make_row(
                    snapshot,
                    "api_export_to_definition",
                    f"Which definitions in `{facts.path}` are exported through `__all__`?",
                    "\n".join(f"{sym.qualified_name} at {sym.path}:{sym.start_line}" for sym in resolved[:12]),
                    [ctx(facts.path, line, line, "__all__ export list")] + [ctx(sym.path, sym.start_line, sym.end_line, "exported symbol definition") for sym in resolved[:3]],
                    "list",
                ))
        for control in facts.controls[:8]:
            category = "control_flow_condition_to_exception" if control.kind == "raise" else "control_flow_condition_to_return"
            add(make_row(
                snapshot,
                category,
                f"In `{control.path}`, what {control.kind} behavior is guarded by condition `{control.condition}`?",
                f"{control.consequence} in {control.scope}",
                [ctx(control.path, control.line, control.line, "guarding condition and consequence")],
                "code",
            ))

    # Subclass/override relationships.
    methods_by_parent: dict[str, list[Symbol]] = defaultdict(list)
    classes = {sym.name: sym for sym in symbols if sym.kind == "class"}
    for sym in symbols:
        if sym.kind == "method" and sym.parent:
            methods_by_parent[sym.parent].append(sym)
    for cls in shuffled([s for s in symbols if s.kind == "class" and s.bases], rng):
        for base in cls.bases:
            base_name = base.split(".")[-1]
            if base_name not in classes:
                continue
            base_methods = {m.name: m for m in methods_by_parent.get(base_name, [])}
            overrides = [m for m in methods_by_parent.get(cls.name, []) if m.name in base_methods]
            if overrides:
                answer = "\n".join(f"{m.qualified_name} overrides {base_methods[m.name].qualified_name}" for m in overrides[:8])
                add(make_row(
                    snapshot,
                    "class_override_relationship",
                    f"Which methods in `{cls.qualified_name}` override methods from base class `{base_name}`?",
                    answer,
                    [ctx(cls.path, cls.start_line, cls.end_line, "subclass definition"), ctx(classes[base_name].path, classes[base_name].start_line, classes[base_name].end_line, "base class definition")],
                    "list",
                ))

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in out:
        by_category[str(row.get("task_category") or "unknown")].append(row)
    for rows in by_category.values():
        rng.shuffle(rows)
    active = [cat for cat, rows in sorted(by_category.items()) if rows]
    rng.shuffle(active)
    selected: list[dict[str, Any]] = []
    while active and len(selected) < max_rows:
        next_active = []
        for cat in active:
            rows = by_category[cat]
            if rows:
                selected.append(rows.pop())
                if len(selected) >= max_rows:
                    break
            if rows:
                next_active.append(cat)
        active = next_active
    meta = {"files": len(files), "parsed_files": len(facts_by_path), "symbols": len(symbols), "calls": len(calls), "tests": len(tests), "candidate_pool": len(out), "generated": len(selected)}
    return selected, meta


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
            current_hunk = Hunk(int(hm.group("old_start")), int(hm.group("old_len") or "1"), int(hm.group("new_start")), int(hm.group("new_len") or "1"), hm.group("header").strip(), [])
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


def compact_lines(lines: Iterable[str], limit: int = 10) -> str:
    out = []
    for line in lines:
        text = line.strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return "\n".join(out)


def changed_paths(files: list[FileDiff], tests: bool | None = None) -> list[str]:
    out = []
    for fd in files:
        path = fd.new_path
        if tests is True and not is_test_path(path):
            continue
        if tests is False and is_test_path(path):
            continue
        out.append(path)
    return sorted(dict.fromkeys(out))


def issue_title(row: dict[str, Any]) -> str:
    lines = [line.strip() for line in str(row.get("problem_statement") or "").splitlines() if line.strip()]
    return short(lines[0] if lines else row.get("instance_id") or "this issue", 180)


def patch_rows_for_issue(row: dict[str, Any], max_rows: int) -> list[dict[str, Any]]:
    issue = issue_title(row)
    patch_files = parse_diff(str(row.get("target_patch") or row.get("patch") or ""))
    test_files = parse_diff(str(row.get("test_patch") or ""))
    all_files = patch_files + test_files
    source_paths = changed_paths(patch_files, tests=False)
    test_paths = changed_paths(test_files, tests=True)
    out: list[dict[str, Any]] = []

    def add(category: str, question: str, answer: str, contexts: list[dict[str, Any]], style: str = "concise", extra: dict[str, Any] | None = None) -> None:
        if answer and len(out) < max_rows:
            out.append(make_row(row, category, question, answer, contexts, style, extra))

    if source_paths and test_paths:
        add(
            "issue_test_to_source_edit_link",
            f"For `{issue}`, which source files and regression-test files are linked by the fix?",
            "Source files:\n" + "\n".join(source_paths[:10]) + "\nTest files:\n" + "\n".join(test_paths[:10]),
            [ctx(path, 1, 1, "source file changed by patch") for path in source_paths[:4]] + [ctx(path, 1, 1, "test file changed by patch") for path in test_paths[:4]],
            "list",
        )
    for fd in all_files:
        for hunk in fd.hunks:
            plus = added_lines(hunk)
            minus = removed_lines(hunk)
            added = compact_lines(plus)
            removed = compact_lines(minus)
            if not added and not removed:
                continue
            hctx = [ctx(fd.new_path, hunk.new_start, hunk.new_start + max(0, hunk.new_len - 1), f"patch hunk {hunk.header or fd.new_path}", added or removed)]
            syms = [line.strip() for line in plus + minus if SYMBOL_LINE_RE.match(line)]
            if syms:
                add("issue_changed_symbol_surface", f"For `{issue}`, which function/class surfaces are touched in `{fd.new_path}`?", "\n".join(syms[:10]), hctx, "list")
            conditions = [line.strip() for line in plus if CONDITION_LINE_RE.match(line)]
            if conditions:
                add("issue_added_guard_condition", f"For `{issue}`, what new guard or branch condition is added in `{fd.new_path}`?", "\n".join(conditions[:8]), hctx, "list")
            returns = [line.strip() for line in plus if RETURN_LINE_RE.match(line)]
            if returns:
                add("issue_added_return_behavior", f"For `{issue}`, what return behavior is added in `{fd.new_path}`?", "\n".join(returns[:8]), hctx, "list")
            exceptions = [line.strip() for line in plus if EXCEPTION_LINE_RE.search(line)]
            if exceptions:
                add("issue_added_exception_behavior", f"For `{issue}`, what exception-handling behavior is added in `{fd.new_path}`?", "\n".join(exceptions[:8]), hctx, "list")
            imports = [line.strip() for line in plus if IMPORT_LINE_RE.match(line)]
            if imports:
                add("issue_added_import_dependency", f"For `{issue}`, which import dependencies are added in `{fd.new_path}`?", "\n".join(imports[:8]), hctx, "list")
            assertions = [line.strip() for line in plus if ASSERT_LINE_RE.search(line)]
            if assertions:
                add("issue_regression_assertion_surface", f"For `{issue}`, what regression assertion or expected failure is introduced in `{fd.new_path}`?", "\n".join(assertions[:8]), hctx, "list")
            if plus and minus:
                old_control = [line.strip() for line in minus if CONDITION_LINE_RE.match(line) or RETURN_LINE_RE.match(line) or EXCEPTION_LINE_RE.search(line)]
                new_control = [line.strip() for line in plus if CONDITION_LINE_RE.match(line) or RETURN_LINE_RE.match(line) or EXCEPTION_LINE_RE.search(line)]
                if old_control or new_control:
                    add(
                        "issue_before_after_control_flow_change",
                        f"For `{issue}`, what control-flow behavior changes in `{fd.new_path}` around `{hunk.header or fd.new_path}`?",
                        "Before:\n" + compact_lines(old_control or minus, 8) + "\nAfter:\n" + compact_lines(new_control or plus, 8),
                        hctx,
                        "diff_summary",
                    )
            if added:
                add("issue_fix_hunk_edit_location", f"Where is a concrete edit made for `{issue}`?", f"{fd.new_path}:{hunk.new_start}-{hunk.new_start + max(0, hunk.new_len - 1)}\n{added}", hctx, "location_code")
            if len(out) >= max_rows:
                break
        if len(out) >= max_rows:
            break
    return out[:max_rows]


def read_static_rows(path: Path) -> list[dict[str, Any]]:
    columns = ["repo_id", "commit_sha", "base_commit", "instance_id", "source_dataset", "embedding_metadata_json"]
    return pq.read_table(path, columns=columns).to_pylist()


def iter_patch_rows(path: Path) -> Iterable[dict[str, Any]]:
    columns = ["source_dataset", "row_index", "instance_id", "repo_id", "base_commit", "commit_sha", "problem_statement", "patch", "target_patch", "test_patch", "usable_for_train"]
    table = pq.read_table(path, columns=columns)
    yield from table.to_pylist()


def load_seen(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
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
    parser.add_argument("--mode", choices=["static", "patch"], required=True)
    parser.add_argument("--static-shards-root", type=Path, default=Path("/path/to/ad-local/storage/issuefix_swefixer/static_tables/shards"))
    parser.add_argument("--input-metadata", type=Path, default=Path("/path/to/ad-local/storage/issuefix_swefixer/metadata/swefixer_full.parquet"))
    parser.add_argument("--repo-root", type=Path, default=Path("/path/to/ad-local/storage/issuefix_swefixer/repos"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=24)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--target-rows", type=int, default=0)
    parser.add_argument("--rows-per-snapshot", type=int, default=192)
    parser.add_argument("--rows-per-issue", type=int, default=40)
    parser.add_argument("--max-files-per-snapshot", type=int, default=128)
    parser.add_argument("--max-file-bytes", type=int, default=524288)
    parser.add_argument("--seed", type=int, default=8117)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(args.output_jsonl) if args.resume else set()
    generated = len(seen)
    mode = "a" if args.resume else "w"
    counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    meta_totals: Counter[str] = Counter()
    started = time.time()

    with args.output_jsonl.open(mode, encoding="utf-8") as out_fh:
        if args.mode == "static":
            static_paths = sorted(args.static_shards_root.glob("static_full.shard_*_of_8.parquet"))
            if not static_paths:
                raise SystemExit("no static shard parquet files found")
            static_path = static_paths[args.shard_index % len(static_paths)]
            subshard_index = args.shard_index // len(static_paths)
            subshard_count = max(1, (args.num_shards + len(static_paths) - 1) // len(static_paths))
            rows = read_static_rows(static_path)
            for idx, snapshot in enumerate(rows):
                if idx % subshard_count != subshard_index:
                    continue
                if args.target_rows and generated >= args.target_rows:
                    break
                counts["snapshots_seen"] += 1
                try:
                    items, meta = static_rows_for_snapshot(snapshot, args.repo_root, args.max_files_per_snapshot, args.max_file_bytes, args.rows_per_snapshot, args.seed + args.shard_index)
                except Exception as exc:
                    counts["snapshot_errors"] += 1
                    counts[f"error_{type(exc).__name__}"] += 1
                    continue
                meta_totals.update(meta)
                for item in items:
                    rid = item["id"]
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
                if args.progress_every and counts["snapshots_seen"] % args.progress_every == 0:
                    print(json.dumps({"snapshots_seen": counts["snapshots_seen"], "generated": generated, "elapsed_sec": round(time.time() - started, 1)}, sort_keys=True), flush=True)
        else:
            for row in iter_patch_rows(args.input_metadata):
                idx = int(row.get("row_index") or 0)
                if args.num_shards > 1 and idx % args.num_shards != args.shard_index:
                    continue
                if row.get("usable_for_train") is False:
                    counts["unusable_skipped"] += 1
                    continue
                if args.target_rows and generated >= args.target_rows:
                    break
                counts["issues_seen"] += 1
                for item in patch_rows_for_issue(row, args.rows_per_issue):
                    rid = item["id"]
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
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "elapsed_sec": round(time.time() - started, 3),
        "generated_total_in_output": generated,
        "counts": dict(counts),
        "meta_totals": dict(meta_totals),
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
