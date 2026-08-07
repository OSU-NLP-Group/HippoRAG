#!/usr/bin/env python3
"""Generate diverse deterministic repo-coverage QA from repository snapshots.

This CPU-only generator scans repository files directly and creates fresh facts:

* cross-file import/call/test links
* docstring/comment/API facts
* config and exception/logging facts
* deterministic contrastive/disambiguation facts

Unlike the fast expansion wave, these rows are not just alternate views of
existing QA rows; they come from parsing source files in each snapshot.
"""

import argparse
import ast
import hashlib
import json
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq


GENERATOR = "deterministic_diverse_repo_coverage_v1"
PY_EXT = ".py"
CONFIG_NAMES = {"pyproject.toml", "setup.cfg", "setup.py", "tox.ini", "pytest.ini", "mypy.ini"}
CONFIG_EXTS = {".toml", ".cfg", ".ini", ".yaml", ".yml", ".json"}
TEST_RE = re.compile(r"(^|/)(tests?|testing|test)/|(^|/|_)test_[^/]*\.py$|(^|/)test[^/]*\.py$")
COMMENT_RE = re.compile(r"^\s*#\s*(TODO|FIXME|NOTE|WARNING|HACK|XXX)\b:?\s*(.+)$", re.I)


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
    docstring: str = ""
    bases: List[str] = field(default_factory=list)


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
    assertions: List[str]
    calls: List[str]


@dataclass
class FileFacts:
    path: str
    text: str
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[ImportFact] = field(default_factory=list)
    calls: List[CallFact] = field(default_factory=list)
    tests: List[TestFact] = field(default_factory=list)
    raises: List[Tuple[int, str, str]] = field(default_factory=list)
    logs: List[Tuple[int, str]] = field(default_factory=list)
    comments: List[Tuple[int, str, str]] = field(default_factory=list)
    config_lines: List[Tuple[int, str]] = field(default_factory=list)
    exports: List[Tuple[int, List[str]]] = field(default_factory=list)


def stable_id(*parts: Any) -> str:
    raw = "\0".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()


def repo_path(repo_root: Path, repo_id: str) -> Path:
    direct = repo_root / repo_id
    if direct.exists():
        return direct
    return repo_root / repo_id.replace("/", "__")


def git(repo: Path, args: List[str], text: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text, timeout=timeout, check=False)


def module_name(path: str) -> str:
    p = path[:-3] if path.endswith(".py") else path
    if p.endswith("/__init__"):
        p = p[: -len("/__init__")]
    return p.replace("/", ".")


def source_signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        args = ast.unparse(node.args)
        returns = " -> %s" % ast.unparse(node.returns) if node.returns is not None else ""
        return "%s %s(%s)%s" % (prefix, node.name, args, returns)
    return ""


def is_test_path(path: str) -> bool:
    return bool(TEST_RE.search(path))


def short(text: Any, limit: int = 320) -> str:
    return " ".join(str(text or "").split())[:limit].rstrip()


def read_file(repo: Path, commit: str, path: str, max_file_bytes: int) -> Optional[str]:
    proc = git(repo, ["show", "%s:%s" % (commit, path)], text=False, timeout=120)
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
        return "%s.%s" % (prefix, node.attr) if prefix else node.attr
    return ""


def literal_string_list(node: ast.AST) -> List[str]:
    try:
        value = ast.literal_eval(node)
    except Exception:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if isinstance(item, str)]
    return []


class Visitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.stack = []  # type: List[str]
        self.symbols = []  # type: List[Symbol]
        self.imports = []  # type: List[ImportFact]
        self.calls = []  # type: List[CallFact]
        self.tests = []  # type: List[TestFact]
        self.raises = []  # type: List[Tuple[int, str, str]]
        self.logs = []  # type: List[Tuple[int, str]]
        self.exports = []  # type: List[Tuple[int, List[str]]]

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

    def _visit_function(self, node: Any) -> None:
        mod = module_name(self.path)
        parent = self.stack[-1] if self.stack else ""
        qn = "%s.%s.%s" % (mod, parent, node.name) if parent else "%s.%s" % (mod, node.name)
        kind = "method" if parent else "function"
        self.symbols.append(
            Symbol(kind, node.name, qn, self.path, node.lineno, getattr(node, "end_lineno", node.lineno),
                   source_signature(node), parent, short(ast.get_docstring(node) or "", 500))
        )
        test_asserts = []  # type: List[str]
        test_calls = []  # type: List[str]
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                test_asserts.append(short(ast.unparse(child.test), 220))
            elif isinstance(child, ast.Call):
                cname = call_name(child.func)
                if cname:
                    test_calls.append(cname)
                    if cname.startswith("pytest.raises") or ".assert" in cname or cname.startswith("assert"):
                        test_asserts.append(short(ast.unparse(child), 220))
        if node.name.startswith("test_") or is_test_path(self.path):
            if test_asserts or test_calls:
                self.tests.append((TestFact(self.path, node.lineno, node.name, test_asserts[:6], test_calls[:12])))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        mod = module_name(self.path)
        bases = [ast.unparse(base) for base in node.bases]
        self.symbols.append(
            Symbol("class", node.name, "%s.%s" % (mod, node.name), self.path, node.lineno,
                   getattr(node, "end_lineno", node.lineno), "", "", short(ast.get_docstring(node) or "", 500), bases)
        )
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Assign(self, node: ast.Assign) -> Any:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                values = literal_string_list(node.value)
                if values:
                    self.exports.append((node.lineno, values[:20]))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        name = call_name(node.func)
        if name:
            self.calls.append(CallFact(self.path, node.lineno, name, self.scope(), short(ast.unparse(node), 280)))
            if name.endswith((".debug", ".info", ".warning", ".error", ".exception")) or name == "warnings.warn":
                self.logs.append((node.lineno, short(ast.unparse(node), 280)))
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> Any:
        self.raises.append((node.lineno, self.scope(), short(ast.unparse(node), 280)))
        self.generic_visit(node)


def parse_file(path: str, text: str) -> FileFacts:
    facts = FileFacts(path, text)
    for line_no, line in enumerate(text.splitlines(), start=1):
        cm = COMMENT_RE.match(line)
        if cm:
            facts.comments.append((line_no, cm.group(1).upper(), short(cm.group(2), 260)))
        if Path(path).name in CONFIG_NAMES or Path(path).suffix.lower() in CONFIG_EXTS:
            stripped = line.strip()
            if stripped and re.search(r"(entry_points|console_scripts|scripts|dependencies|pytest|mypy|flake8|ruff|tool\.|extras_require)", stripped):
                facts.config_lines.append((line_no, short(stripped, 260)))
    if path.endswith(PY_EXT):
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
        facts.raises = visitor.raises
        facts.logs = visitor.logs
        facts.exports = visitor.exports
    return facts


def resolve_import(current_path: str, raw_module: str, known_paths: set) -> str:
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
    for candidate in ("%s.py" % stem, "%s/__init__.py" % stem):
        if candidate in known_paths:
            return candidate
    return ""


def ctx(path: str, start: int, end: int, reason: str) -> Dict[str, Any]:
    return {"path": path, "start_line": max(1, int(start)), "end_line": max(1, int(end)), "reason": reason}


def make_row(snapshot: Dict[str, Any], category: str, question: str, answer: str, contexts: List[Dict[str, Any]],
             style: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "id": stable_id(GENERATOR, snapshot.get("repo_id", ""), snapshot.get("base_commit", ""), category, question, answer),
        "repo_id": snapshot.get("repo_id", ""),
        "base_commit": snapshot.get("base_commit", ""),
        "commit_sha": snapshot.get("commit_sha", snapshot.get("base_commit", "")),
        "source_instance_id": snapshot.get("instance_id", ""),
        "source_dataset": snapshot.get("source_dataset", ""),
        "qa_source": "diverse_repo_coverage_scan",
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


def shuffled(items: List[Any], rng: random.Random) -> List[Any]:
    out = list(items)
    rng.shuffle(out)
    return out


def rows_for_snapshot(snapshot: Dict[str, Any], repo_root: Path, max_files: int, max_file_bytes: int,
                      max_rows: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    repo_id = str(snapshot.get("repo_id") or "")
    commit = str(snapshot.get("base_commit") or snapshot.get("commit_sha") or "")
    repo = repo_path(repo_root, repo_id)
    if not repo.exists() or not commit:
        return [], {"missing_repo": 1}
    metadata = {}
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

    facts_by_path = {}  # type: Dict[str, FileFacts]
    for path in files:
        text = read_file(repo, commit, path, max_file_bytes)
        if text is None:
            continue
        facts_by_path[path] = parse_file(path, text)
    known_paths = set(facts_by_path)
    symbols = [sym for facts in facts_by_path.values() for sym in facts.symbols]
    imports = [imp for facts in facts_by_path.values() for imp in facts.imports]
    calls = [call for facts in facts_by_path.values() for call in facts.calls]
    tests = [test for facts in facts_by_path.values() for test in facts.tests]

    symbols_by_name = defaultdict(list)  # type: Dict[str, List[Symbol]]
    symbols_by_qn = {}  # type: Dict[str, Symbol]
    for sym in symbols:
        symbols_by_name[sym.name].append(sym)
        symbols_by_qn[sym.qualified_name] = sym

    out = []  # type: List[Dict[str, Any]]
    seen = set()
    pool_limit = max_rows * 6
    per_category_cap = max(48, max_rows)
    pool_counts = Counter()

    def add(row: Dict[str, Any]) -> None:
        category = str(row.get("task_category") or "unknown")
        if len(out) >= pool_limit or pool_counts[category] >= per_category_cap:
            return
        rid = row["id"]
        if rid in seen:
            return
        seen.add(rid)
        pool_counts[category] += 1
        out.append(row)

    # Cross-file imports, including test-to-code imports.
    for imp in shuffled(imports, rng):
        if len(out) >= pool_limit:
            break
        target_path = resolve_import(imp.path, imp.module, known_paths)
        if not target_path or target_path == imp.path:
            continue
        cat = "cross_file_test_import_link" if is_test_path(imp.path) and not is_test_path(target_path) else "cross_file_import_dependency"
        add(make_row(
            snapshot,
            cat,
            "In `%s`, what repository file does `%s` connect to?" % (imp.path, short(imp.statement, 220)),
            target_path,
            [ctx(imp.path, imp.line, imp.line, "import statement"), ctx(target_path, 1, 1, "resolved repository import target")],
            "path",
            {"source_path": imp.path, "target_path": target_path, "import_statement": imp.statement},
        ))

    # Cross-file call/name links: a call/use in one file matches a unique symbol in another file.
    for call in shuffled(calls, rng):
        if len(out) >= pool_limit:
            break
        simple = call.name.split(".")[-1]
        candidates = [sym for sym in symbols_by_name.get(simple, []) if sym.path != call.path]
        if len(candidates) != 1:
            continue
        sym = candidates[0]
        add(make_row(
            snapshot,
            "cross_file_call_to_symbol",
            "Which repository symbol is the call/name `%s` in `%s` most directly linked to?" % (call.name, call.path),
            "%s in %s" % (sym.qualified_name, sym.path),
            [ctx(call.path, call.line, call.line, "call/name occurrence"), ctx(sym.path, sym.start_line, sym.end_line, "unique matching symbol definition")],
            "symbol_location",
            {"source_path": call.path, "target_path": sym.path, "symbol_name": sym.name, "qualified_name": sym.qualified_name},
        ))

    # Tests to implementation via imports/calls.
    for test in shuffled(tests, rng):
        if len(out) >= pool_limit:
            break
        linked = []  # type: List[Symbol]
        for cname in test.calls:
            simple = cname.split(".")[-1]
            linked.extend([sym for sym in symbols_by_name.get(simple, []) if not is_test_path(sym.path)])
        unique = []
        seen_sym = set()
        for sym in linked:
            if sym.qualified_name not in seen_sym:
                unique.append(sym)
                seen_sym.add(sym.qualified_name)
        if unique:
            answer = "\n".join("%s in %s" % (sym.qualified_name, sym.path) for sym in unique[:6])
            contexts = [ctx(test.path, test.line, test.line, "test function")] + [
                ctx(sym.path, sym.start_line, sym.end_line, "implementation symbol referenced by test") for sym in unique[:3]
            ]
            add(make_row(
                snapshot,
                "test_to_implementation_symbol_links",
                "Which implementation symbols are linked to test `%s` in `%s`?" % (test.name, test.path),
                answer,
                contexts,
                "list",
                {"source_path": test.path},
            ))
        if test.assertions:
            add(make_row(
                snapshot,
                "test_assertion_behavior",
                "What assertions or expected behaviors are recorded in test `%s` in `%s`?" % (test.name, test.path),
                "\n".join(test.assertions[:6]),
                [ctx(test.path, test.line, test.line, "test assertion evidence")],
                "list",
                {"source_path": test.path},
            ))

    # Docstrings, exports, exceptions, logging, comments, config.
    for facts in shuffled(list(facts_by_path.values()), rng):
        if len(out) >= pool_limit:
            break
        for sym in shuffled([s for s in facts.symbols if s.docstring], rng)[:8]:
            add(make_row(
                snapshot,
                "docstring_symbol_behavior",
                "What behavior is documented for `%s`?" % sym.qualified_name,
                sym.docstring,
                [ctx(sym.path, sym.start_line, sym.end_line, "symbol docstring")],
                "docstring",
                {"source_path": sym.path, "qualified_name": sym.qualified_name},
            ))
            add(make_row(
                snapshot,
                "docstring_to_symbol",
                "Which symbol is documented as: `%s`?" % short(sym.docstring, 180),
                sym.qualified_name,
                [ctx(sym.path, sym.start_line, sym.end_line, "symbol docstring")],
                "symbol",
                {"source_path": sym.path, "qualified_name": sym.qualified_name},
            ))
        for line, names in facts.exports[:3]:
            add(make_row(
                snapshot,
                "api_export_members",
                "Which public API names are exported from `%s` through `__all__`?" % facts.path,
                "\n".join(names),
                [ctx(facts.path, line, line, "__all__ assignment")],
                "list",
                {"source_path": facts.path},
            ))
        for line, scope, statement in facts.raises[:4]:
            add(make_row(
                snapshot,
                "repo_exception_behavior",
                "Where is exception behavior `%s` defined in `%s`?" % (short(statement, 180), repo_id),
                "%s:%s in %s" % (facts.path, line, scope),
                [ctx(facts.path, line, line, "raise statement")],
                "location",
                {"source_path": facts.path},
            ))
        for line, statement in facts.logs[:4]:
            add(make_row(
                snapshot,
                "repo_logging_behavior",
                "Where does `%s` log or warn with `%s`?" % (repo_id, short(statement, 180)),
                "%s:%s" % (facts.path, line),
                [ctx(facts.path, line, line, "logging/warning call")],
                "location",
                {"source_path": facts.path},
            ))
        for line, tag, text in facts.comments[:4]:
            add(make_row(
                snapshot,
                "repo_maintenance_comment",
                "Where is the `%s` maintenance note `%s` recorded?" % (tag, short(text, 180)),
                "%s:%s" % (facts.path, line),
                [ctx(facts.path, line, line, "maintenance comment")],
                "location",
                {"source_path": facts.path},
            ))
        for line, config in facts.config_lines[:5]:
            add(make_row(
                snapshot,
                "repo_config_behavior",
                "What configuration line appears in `%s`?" % facts.path,
                config,
                [ctx(facts.path, line, line, "configuration line")],
                "config",
                {"source_path": facts.path},
            ))

    # Contrastive/disambiguation: same simple symbol in multiple files.
    ambiguous = [(name, syms) for name, syms in symbols_by_name.items() if 1 < len({s.path for s in syms}) <= 8]
    for name, syms in shuffled(ambiguous, rng):
        if len(out) >= pool_limit:
            break
        syms = shuffled(syms, rng)
        target = syms[0]
        distractors = [s for s in syms[1:4] if s.path != target.path]
        if not distractors:
            continue
        choices = [target.path] + [s.path for s in distractors]
        add(make_row(
            snapshot,
            "contrastive_symbol_path_disambiguation",
            "Several files define a symbol named `%s`. Which listed path contains `%s`?\nChoices:\n%s" % (
                name, target.qualified_name, "\n".join(choices)
            ),
            target.path,
            [ctx(target.path, target.start_line, target.end_line, "target symbol definition")] + [
                ctx(s.path, s.start_line, s.end_line, "same-name distractor symbol") for s in distractors[:2]
            ],
            "path",
            {"symbol_name": name, "qualified_name": target.qualified_name, "choices": choices},
        ))

    meta = {
        "files": len(files),
        "parsed_files": len(facts_by_path),
        "symbols": len(symbols),
        "imports": len(imports),
        "calls": len(calls),
        "tests": len(tests),
        "generated": min(len(out), max_rows),
        "candidate_pool": len(out),
    }
    by_category = defaultdict(list)  # type: Dict[str, List[Dict[str, Any]]]
    for item in out:
        by_category[str(item.get("task_category") or "unknown")].append(item)
    for values in by_category.values():
        rng.shuffle(values)
    active = [category for category, values in sorted(by_category.items()) if values]
    rng.shuffle(active)
    selected = []  # type: List[Dict[str, Any]]
    while active and len(selected) < max_rows:
        next_active = []
        for category in active:
            values = by_category[category]
            if values:
                selected.append(values.pop())
                if len(selected) >= max_rows:
                    break
            if values:
                next_active.append(category)
        active = next_active
    return selected, meta


def read_static_rows(path: Path) -> List[Dict[str, Any]]:
    columns = ["repo_id", "commit_sha", "base_commit", "instance_id", "source_dataset", "embedding_metadata_json"]
    table = pq.read_table(path, columns=columns)
    return table.to_pylist()


def load_seen(path: Path) -> set:
    seen = set()
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
    parser.add_argument("--static-shards-root", type=Path, default=Path("/path/to/ad-local/storage/issuefix_swefixer/static_tables/shards"))
    parser.add_argument("--repo-root", type=Path, default=Path("/path/to/ad-local/storage/issuefix_swefixer/repos"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=24)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--target-rows", type=int, default=420000)
    parser.add_argument("--max-files-per-snapshot", type=int, default=96)
    parser.add_argument("--max-file-bytes", type=int, default=524288)
    parser.add_argument("--rows-per-snapshot", type=int, default=256)
    parser.add_argument("--seed", type=int, default=9109)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    static_paths = sorted(args.static_shards_root.glob("static_full.shard_*_of_8.parquet"))
    if not static_paths:
        raise SystemExit("no static shard parquet files found")
    static_path = static_paths[args.shard_index % len(static_paths)]
    subshard_index = args.shard_index // len(static_paths)
    subshard_count = max(1, (args.num_shards + len(static_paths) - 1) // len(static_paths))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(args.output_jsonl) if args.resume else set()
    generated = len(seen)
    category_counts = Counter()
    if args.resume and args.output_jsonl.exists():
        for row in load_seen(args.output_jsonl):
            pass
    mode = "a" if args.resume else "w"
    rows = read_static_rows(static_path)
    started = time.time()
    counts = Counter()
    meta_totals = Counter()

    with args.output_jsonl.open(mode, encoding="utf-8") as out:
        for idx, snapshot in enumerate(rows):
            if idx % subshard_count != subshard_index:
                continue
            if args.target_rows and generated >= args.target_rows:
                break
            counts["snapshots_seen"] += 1
            try:
                items, meta = rows_for_snapshot(
                    snapshot,
                    args.repo_root,
                    max_files=args.max_files_per_snapshot,
                    max_file_bytes=args.max_file_bytes,
                    max_rows=args.rows_per_snapshot,
                    seed=args.seed + args.shard_index,
                )
            except Exception as exc:  # noqa: BLE001
                counts["snapshot_errors"] += 1
                counts["error_%s" % type(exc).__name__] += 1
                continue
            meta_totals.update(meta)
            for item in items:
                rid = item["id"]
                if rid in seen:
                    counts["duplicate_skipped"] += 1
                    continue
                seen.add(rid)
                out.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                generated += 1
                counts["generated"] += 1
                category_counts[str(item.get("task_category") or "unknown")] += 1
                if args.target_rows and generated >= args.target_rows:
                    break
            if args.progress_every and counts["snapshots_seen"] % args.progress_every == 0:
                print(json.dumps({"snapshots_seen": counts["snapshots_seen"], "generated": generated}, sort_keys=True), flush=True)

    audit = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "static_path": str(static_path),
        "subshard_index": subshard_index,
        "subshard_count": subshard_count,
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
