#!/usr/bin/env python3
"""Generate deterministic AST QA at repository-snapshot scale.

This writes compact JSONL QA facts rather than trainer parquets.  The intended
use is over-generation followed by later source-mix selection/conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_code2lora_context_distill_candidates import (  # noqa: E402
    Symbol,
    eligible_source_path,
    line_snippet,
    parse_symbols,
    resolve_import_path,
)
from prepare_code2lora_repo_ntp_dataset import repo_path, stable_seed  # noqa: E402


TEXT_EXTENSIONS = {".py"}
GENERATOR = "deterministic_ast_snapshot_v1"
warnings.filterwarnings("ignore", category=SyntaxWarning)


def git(repo: Path, args: list[str], *, text: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=text,
        timeout=timeout,
        check=False,
    )


def list_python_files(repo: Path, commit: str, max_file_bytes: int) -> list[tuple[str, int]]:
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
        if (
            ftype == "blob"
            and 0 < size <= max_file_bytes
            and Path(path).suffix.lower() in TEXT_EXTENSIONS
            and eligible_source_path(path)
        ):
            out.append((path, size))
    return out


def read_file(repo: Path, commit: str, path: str, max_file_bytes: int) -> str | None:
    proc = git(repo, ["show", f"{commit}:{path}"], text=False, timeout=180)
    if proc.returncode != 0 or len(proc.stdout) > max_file_bytes:
        return None
    try:
        return proc.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def read_static_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["repo_id", "commit_sha", "base_commit", "instance_id"])
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for idx in range(table.num_rows):
        repo_id = str(cols["repo_id"][idx])
        base_commit = str(cols["base_commit"][idx] or cols["commit_sha"][idx])
        key = (repo_id, base_commit)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "repo_id": repo_id,
                "commit_sha": str(cols["commit_sha"][idx] or base_commit),
                "base_commit": base_commit,
                "source_instance_id": str(cols["instance_id"][idx] or ""),
            }
        )
    return rows


def qa_id(repo_id: str, base_commit: str, category: str, question: str, answer: str) -> str:
    raw = "\0".join([repo_id, base_commit, category, question, answer])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def row(
    snapshot: dict[str, str],
    category: str,
    question: str,
    answer: str,
    *,
    source_path: str,
    start_line: int,
    end_line: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": qa_id(snapshot["repo_id"], snapshot["base_commit"], category, question, answer),
        "repo_id": snapshot["repo_id"],
        "commit_sha": snapshot["commit_sha"],
        "base_commit": snapshot["base_commit"],
        "qa_source": "ast_generated_snapshot",
        "task_category": category,
        "question": question,
        "answer": answer,
        "answer_style": "symbolic",
        "source_path": source_path,
        "required_context": [{"path": source_path, "start_line": start_line, "end_line": end_line}],
        "source_instance_id": snapshot.get("source_instance_id", ""),
        "generator": GENERATOR,
    }
    if extra:
        payload.update(extra)
    return payload


def candidates_for_snapshot(
    snapshot: dict[str, str],
    repo: Path,
    *,
    max_file_bytes: int,
    max_files_per_snapshot: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = list_python_files(repo, snapshot["base_commit"], max_file_bytes)
    if not files:
        return [], {"files": 0, "texts": 0, "symbols": 0, "imports": 0}

    rng = random.Random(stable_seed(snapshot["repo_id"], snapshot["base_commit"], seed, "ast-files"))
    files = sorted(files)
    rng.shuffle(files)
    if max_files_per_snapshot:
        files = files[:max_files_per_snapshot]

    texts: dict[str, str] = {}
    symbols: list[Symbol] = []
    imports: list[dict[str, str]] = []
    known_paths = {path for path, _size in files}
    for path, _size in files:
        text = read_file(repo, snapshot["base_commit"], path, max_file_bytes)
        if text is None:
            continue
        syms, imps, _ids = parse_symbols(path, text)
        texts[path] = text
        symbols.extend(syms)
        imports.extend(imps)

    out: list[dict[str, Any]] = []
    methods_by_name: dict[str, list[Symbol]] = defaultdict(list)
    for sym in symbols:
        if sym.kind == "method":
            methods_by_name[sym.name].append(sym)

    def add(category: str, question: str, answer: str, sym: Symbol, extra: dict[str, Any] | None = None) -> None:
        if not answer or sym.path not in texts:
            return
        context_hint = line_snippet(texts[sym.path], sym.start_line, sym.end_line)
        payload = row(
            snapshot,
            category,
            question,
            answer,
            source_path=sym.path,
            start_line=sym.start_line,
            end_line=sym.end_line,
            extra={"context_hint": context_hint, **(extra or {})},
        )
        out.append(payload)

    for sym in symbols:
        if sym.kind in {"function", "class", "method"}:
            add(
                "qa_symbol_path",
                f"What file defines `{sym.qualified_name}`?",
                sym.path,
                sym,
                {"symbol_kind": sym.kind, "symbol_name": sym.name, "qualified_name": sym.qualified_name},
            )
        if sym.kind in {"function", "method"} and sym.signature:
            add(
                "qa_signature",
                f"What is the signature of `{sym.qualified_name}`?",
                sym.signature,
                sym,
                {"symbol_kind": sym.kind, "symbol_name": sym.name, "qualified_name": sym.qualified_name},
            )
        if sym.kind == "method" and len(methods_by_name[sym.name]) == 1:
            add(
                "qa_method_class",
                f"Which class defines method `{sym.name}`?",
                sym.parent,
                sym,
                {"symbol_kind": sym.kind, "symbol_name": sym.name, "qualified_name": sym.qualified_name},
            )
        if sym.kind == "class" and sym.bases:
            add(
                "qa_class_bases",
                f"What are the base classes of `{sym.qualified_name}`?",
                ", ".join(sym.bases),
                sym,
                {"symbol_kind": sym.kind, "symbol_name": sym.name, "qualified_name": sym.qualified_name},
            )
        if sym.kind == "constant":
            add(
                "qa_constant",
                f"What is the value of constant `{sym.qualified_name}`?",
                sym.value,
                sym,
                {"symbol_kind": sym.kind, "symbol_name": sym.name, "qualified_name": sym.qualified_name},
            )
        if sym.kind == "enum":
            add(
                "qa_enum_members",
                f"What enum members are defined in `{sym.qualified_name}`?",
                sym.value,
                sym,
                {"symbol_kind": sym.kind, "symbol_name": sym.name, "qualified_name": sym.qualified_name},
            )

    symbol_by_path_name = {(s.path, s.name): s for s in symbols}
    for imp in imports:
        resolved = resolve_import_path(imp["path"], imp["module"], known_paths)
        if not resolved:
            continue
        answer = resolved if imp["module"].startswith(".") else imp["module"]
        source = texts.get(imp["path"], "")
        statement = imp["statement"]
        line_no = next((i + 1 for i, line in enumerate(source.splitlines()) if statement in line), 1)
        fake = symbol_by_path_name.get((imp["path"], imp["name"])) or Symbol(
            "import", imp["name"], "", imp["path"], line_no, line_no, "", "", [], "", []
        )
        if imp["module"].startswith("."):
            question = f"In `{imp['path']}`, what repository file does `{statement}` resolve to?"
        else:
            question = f"Where is `{imp['name'] or imp['module']}` imported from in `{imp['path']}`?"
        add(
            "qa_import_resolution",
            question,
            answer,
            fake,
            {"import_statement": statement, "resolved_path": resolved},
        )

    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in out:
        key = (str(item["task_category"]), str(item["question"]), str(item["answer"]))
        if key in seen:
            continue
        seen.add(key)
        if len(str(item["question"])) > 512 or len(str(item["answer"])) > 512:
            continue
        deduped.append(item)

    by_category: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for item in deduped:
        by_category[str(item["task_category"])].append(item)
    for category, values in list(by_category.items()):
        shuffled = list(values)
        rng.shuffle(shuffled)
        by_category[category] = deque(shuffled)
    category_order = [
        "qa_signature",
        "qa_symbol_path",
        "qa_import_resolution",
        "qa_method_class",
        "qa_class_bases",
        "qa_constant",
        "qa_enum_members",
    ]
    stratified: list[dict[str, Any]] = []
    active = deque([category for category in category_order if by_category[category]])
    while active:
        category = active.popleft()
        stratified.append(by_category[category].popleft())
        if by_category[category]:
            active.append(category)

    return stratified, {
        "files": len(files),
        "texts": len(texts),
        "symbols": len(symbols),
        "imports": len(imports),
        "candidates": len(stratified),
    }


def load_completed(path: Path) -> tuple[set[str], set[tuple[str, str]], Counter[str], int]:
    ids: set[str] = set()
    complete_snapshots: set[tuple[str, str]] = set()
    categories: Counter[str] = Counter()
    lines = 0
    if not path.exists():
        return ids, complete_snapshots, categories, lines
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            lines += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            ids.add(str(item.get("id") or ""))
            categories[str(item.get("task_category") or "unknown")] += 1
            complete_snapshots.add((str(item.get("repo_id")), str(item.get("base_commit"))))
    ids.discard("")
    return ids, complete_snapshots, categories, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-static", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("/path/to/ad-local/storage/issuefix_swefixer/repos"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--target-rows", type=int, default=500_000)
    parser.add_argument("--qa-rows-per-snapshot", type=int, default=64)
    parser.add_argument("--skip-rows-per-snapshot", type=int, default=0)
    parser.add_argument("--max-files-per-snapshot", type=int, default=128)
    parser.add_argument("--max-file-bytes", type=int, default=524_288)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    started = time.time()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    done_snapshots: set[tuple[str, str]] = set()
    category_counts: Counter[str] = Counter()
    generated = 0
    mode = "w"
    if args.resume:
        done_ids, done_snapshots, category_counts, generated = load_completed(args.output_jsonl)
        mode = "a"
    if args.target_rows and generated >= args.target_rows:
        print(json.dumps({"status": "already_complete", "generated": generated}), flush=True)
        return 0

    snapshots = read_static_rows(args.input_static)
    random.Random(stable_seed(str(args.input_static), args.seed, "snapshot-order")).shuffle(snapshots)
    audit: dict[str, Any] = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "input_snapshots": len(snapshots),
        "started_unix": started,
        "generator": GENERATOR,
        "repos": {},
        "errors": Counter(),
    }
    processed = skipped = missing_repo = 0
    with args.output_jsonl.open(mode, encoding="utf-8") as out:
        for index, snapshot in enumerate(snapshots, 1):
            key = (snapshot["repo_id"], snapshot["base_commit"])
            if key in done_snapshots:
                skipped += 1
                continue
            repo = repo_path(args.repo_root, snapshot["repo_id"])
            if not (repo / ".git").exists():
                missing_repo += 1
                audit["errors"]["missing_repo"] += 1
                continue
            try:
                candidates, meta = candidates_for_snapshot(
                    snapshot,
                    repo,
                    max_file_bytes=args.max_file_bytes,
                    max_files_per_snapshot=args.max_files_per_snapshot,
                    seed=args.seed,
                )
            except Exception as exc:  # noqa: BLE001
                audit["errors"][type(exc).__name__] += 1
                continue
            emitted_here = 0
            start = max(0, args.skip_rows_per_snapshot)
            end = start + max(0, args.qa_rows_per_snapshot)
            for item in candidates[start:end]:
                if item["id"] in done_ids:
                    continue
                out.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
                done_ids.add(item["id"])
                category_counts[str(item["task_category"])] += 1
                generated += 1
                emitted_here += 1
                if args.target_rows and generated >= args.target_rows:
                    break
            processed += 1
            repo_audit = audit["repos"].setdefault(snapshot["repo_id"], {"snapshots": 0, "generated": 0})
            repo_audit["snapshots"] += 1
            repo_audit["generated"] += emitted_here
            if emitted_here or meta.get("candidates"):
                for field in ("files", "texts", "symbols", "imports", "candidates"):
                    repo_audit[field] = repo_audit.get(field, 0) + int(meta.get(field) or 0)
            if args.progress_every and (processed % args.progress_every == 0 or generated >= args.target_rows):
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "input_snapshots": len(snapshots),
                            "generated": generated,
                            "target_rows": args.target_rows,
                            "categories": dict(category_counts),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if args.target_rows and generated >= args.target_rows:
                break
    audit.update(
        {
            "elapsed_sec": round(time.time() - started, 3),
            "processed_snapshots": processed,
            "skipped_completed_snapshots": skipped,
            "missing_repo_snapshots": missing_repo,
            "generated": generated,
            "category_counts": dict(category_counts),
            "errors": dict(audit["errors"]),
            "output_jsonl": str(args.output_jsonl),
        }
    )
    tmp = args.audit_output.with_suffix(args.audit_output.suffix + ".tmp")
    tmp.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.audit_output)
    print(json.dumps({"generated": generated, "audit": str(args.audit_output)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
