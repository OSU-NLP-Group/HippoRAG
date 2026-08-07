#!/usr/bin/env python3
"""Build auditable context packs for future enriched Code2LoRA LLM QA generation.

This is intentionally CPU/data prep only.  It does not choose a serving stack or
call an LLM.  The output JSONL is the input a later, user-approved Gemma 31B
generation job can consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
import sys
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_code2lora_context_distill_candidates import eligible_source_path, parse_symbols  # noqa: E402
from prepare_code2lora_jina_dense_dataset import changed_files_from_patch  # noqa: E402
from prepare_code2lora_repo_ntp_dataset import repo_path, stable_seed  # noqa: E402

DEFAULT_REPO_ROOT = Path("/path/to/ad-local/storage/issuefix_swefixer/repos")
DEFAULT_SWEFIXER_METADATA = Path("/path/to/ad-local/storage/issuefix_swefixer/metadata/swefixer_full.parquet")
TEXT_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".sh", ".bash", ".zsh", ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".md", ".rst", ".txt",
}
CONFIG_HINTS = ("config", "settings", "registry", "plugin", "hook", "callback", "schema", "route")
TEST_HINTS = ("/test", "test_", "_test.", "tests/")


def read_table(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    return [{name: cols[name][i] for name in table.column_names} for i in range(table.num_rows)]


def read_snapshot_table(path: Path) -> list[dict[str, Any]]:
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    columns = [name for name in ("repo_id", "repo", "base_commit", "commit_sha", "commit") if name in names]
    if not columns:
        raise SystemExit(f"snapshot table has no repo/commit columns: {path}")
    table = pq.read_table(path, columns=columns)
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    return [{name: cols[name][i] for name in table.column_names} for i in range(table.num_rows)]


def unique_snapshots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        repo_id = str(row.get("repo_id") or row.get("repo") or "")
        commit = str(row.get("base_commit") or row.get("commit_sha") or row.get("commit") or "")
        if not repo_id or not commit:
            continue
        key = (repo_id, commit)
        if key in seen:
            continue
        seen.add(key)
        normalized = dict(row)
        normalized["repo_id"] = repo_id
        normalized["base_commit"] = commit
        normalized["commit_sha"] = commit
        snapshots.append(normalized)
    return snapshots


def git(repo: Path, args: list[str], *, text: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text,
                          timeout=timeout, check=False)


def list_files(repo: Path, commit: str, max_file_bytes: int) -> list[tuple[str, int]]:
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
        if ftype != "blob" or size <= 0 or size > max_file_bytes:
            continue
        if Path(path).suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if eligible_source_path(path):
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


def line_excerpt(text: str, max_lines: int) -> tuple[str, int, int]:
    lines = text.splitlines()
    excerpt = "\n".join(lines[:max_lines])
    return excerpt, 1, min(len(lines), max_lines)


def context_span(path: str, text: str, *, reason: str, max_lines: int) -> dict[str, Any]:
    excerpt, start, end = line_excerpt(text, max_lines)
    return {"path": path, "start_line": start, "end_line": end, "reason": reason, "text": excerpt}


def pack_prompt(category: str, repo_id: str, contexts: list[dict[str, Any]]) -> str:
    context_blocks = []
    for item in contexts:
        context_blocks.append(
            f"# path: {item['path']}\n# lines: {item['start_line']}-{item['end_line']}\n"
            f"# reason: {item['reason']}\n{item['text']}"
        )
    return (
        "You are a perceptive QA generator that creates context-dependent QA pairs exposing a wide "
        "surface of repository-specific knowledge for a Code2LoRA dataset.\n"
        "The best questions should require reading and synthesizing the supplied repository context. "
        "Avoid generic documentation questions, shallow summaries, trivia, and narrow lookup questions "
        "already covered by AST templates, such as file-defines-symbol, exact signature, or simple "
        "import-location questions.\n"
        "Prefer questions that teach a model how this repository works. Good QA types include:\n"
        "- cross-file control/data flow: how two modules cooperate, where an object/value originates, "
        "or why a helper is wired into another subsystem;\n"
        "- issue and patch reasoning: what behavior is being fixed, which files/functions are implicated, "
        "what minimal edit or regression test would be relevant, or why the changed area matters;\n"
        "- test-to-implementation reasoning: what behavior a test is asserting and how the implementation "
        "satisfies or violates it;\n"
        "- configuration/runtime behavior: how settings, registries, callbacks, hooks, schemas, or routes "
        "change execution at runtime;\n"
        "- lifecycle and invariant reasoning: initialization/order-of-operations, resource ownership, "
        "error handling, compatibility constraints, or assumptions that must remain true across files.\n"
        "Write answers that are specific, grounded, and explanatory. The answer should synthesize the "
        "relevant context rather than merely restating symbol names or file contents.\n"
        "Return JSON only: a list of objects with question, answer, answer_style, task_category, "
        "and required_context.\n\n"
        f"Repository: {repo_id}\n"
        f"Pack category: {category}\n\n"
        + "\n\n---\n\n".join(context_blocks)
    )


def stable_pack_id(repo_id: str, commit: str, category: str, paths: list[str], ordinal: int) -> str:
    raw = "\0".join([repo_id, commit, category, str(ordinal), *paths])
    return hashlib.sha1(raw.encode()).hexdigest()


def issue_rows_by_snapshot(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        repo = str(row.get("repo_id") or row.get("repo") or "")
        commit = str(row.get("base_commit") or row.get("commit_sha") or "")
        if repo and commit:
            out[(repo, commit)].append(row)
    return out


def choose_test_impl_pairs(paths: list[str], limit: int) -> list[list[str]]:
    tests = [path for path in paths if any(hint in path.lower() for hint in TEST_HINTS)]
    impls = [path for path in paths if path not in tests]
    pairs = []
    for test_path in tests:
        stem = Path(test_path).stem.removeprefix("test_").removesuffix("_test").lower()
        match = next((path for path in impls if stem and stem in Path(path).stem.lower()), None)
        if match:
            pairs.append([test_path, match])
        if len(pairs) >= limit:
            break
    return pairs


def choose_config_packs(paths: list[str], limit: int, width: int) -> list[list[str]]:
    hinted = [path for path in paths if any(hint in path.lower() for hint in CONFIG_HINTS)]
    packs = []
    for start in range(0, len(hinted), width):
        pack = hinted[start:start + width]
        if len(pack) >= 2:
            packs.append(pack)
        if len(packs) >= limit:
            break
    return packs


def choose_symbol_packs(texts: dict[str, str], limit: int, width: int, seed: int) -> list[list[str]]:
    symbol_paths = []
    for path, text in texts.items():
        try:
            symbols, _imports, _identifiers = parse_symbols(path, text)
        except (MemoryError, RecursionError, SyntaxError, ValueError):
            continue
        if any(sym.kind in {"class", "function", "method"} for sym in symbols):
            symbol_paths.append(path)
    random.Random(seed).shuffle(symbol_paths)
    packs = []
    for start in range(0, len(symbol_paths), width):
        pack = symbol_paths[start:start + width]
        if len(pack) >= 2:
            packs.append(pack)
        if len(packs) >= limit:
            break
    return packs


def emit_pack(out_fh, snapshot: dict[str, Any], category: str, paths: list[str], texts: dict[str, str],
              ordinal: int, *, max_lines_per_file: int, issue: dict[str, Any] | None = None) -> dict[str, Any]:
    contexts = []
    for path in paths:
        text = texts.get(path)
        if text:
            contexts.append(context_span(path, text, reason=f"{category} context", max_lines=max_lines_per_file))
    if not contexts:
        return {}
    repo_id = str(snapshot["repo_id"])
    commit = str(snapshot["base_commit"])
    pack = {
        "pack_id": stable_pack_id(repo_id, commit, category, [c["path"] for c in contexts], ordinal),
        "repo_id": repo_id,
        "base_commit": commit,
        "category": category,
        "contexts": contexts,
        "prompt_version": "repoqa_aug_context_pack_v2",
        "generator_instructions": pack_prompt(category, repo_id, contexts),
    }
    if issue:
        patch = str(issue.get("target_patch") or issue.get("patch") or "")
        pack.update({
            "issue_id": issue.get("instance_id", ""),
            "issue_text": str(issue.get("problem_statement") or issue.get("prompt") or "")[:4000],
            "gold_changed_files": changed_files_from_patch(patch),
            "gold_patch_hash": hashlib.sha256(patch.encode("utf-8", errors="ignore")).hexdigest() if patch else "",
        })
    out_fh.write(json.dumps(pack, ensure_ascii=False, sort_keys=True) + "\n")
    return pack


def build_packs_for_snapshot(snapshot: dict[str, Any], repo_root: Path, issues: list[dict[str, Any]],
                             args: argparse.Namespace, out_fh) -> Counter[str]:
    repo = repo_path(repo_root, str(snapshot["repo_id"]))
    paths = [path for path, _size in list_files(repo, str(snapshot["base_commit"]), args.max_file_bytes)]
    if args.max_files_per_repo:
        paths = paths[:args.max_files_per_repo]
    texts: dict[str, str] = {}
    for path in paths:
        text = read_file(repo, str(snapshot["base_commit"]), path, args.max_file_bytes)
        if text is not None:
            texts[path] = text
    counts: Counter[str] = Counter()
    ordinal = 0

    for issue in issues[:args.issue_packs_per_repo]:
        patch = str(issue.get("target_patch") or issue.get("patch") or "")
        issue_paths = [path for path in changed_files_from_patch(patch) if path in texts]
        pack = emit_pack(out_fh, snapshot, "issue_related", issue_paths[:args.pack_width], texts, ordinal,
                         max_lines_per_file=args.max_lines_per_file, issue=issue)
        if pack:
            counts["issue_related"] += 1
            ordinal += 1

    for pair in choose_test_impl_pairs(list(texts), args.test_impl_packs_per_repo):
        pack = emit_pack(out_fh, snapshot, "test_to_implementation", pair, texts, ordinal,
                         max_lines_per_file=args.max_lines_per_file)
        if pack:
            counts["test_to_implementation"] += 1
            ordinal += 1

    for pack_paths in choose_config_packs(list(texts), args.config_packs_per_repo, args.pack_width):
        pack = emit_pack(out_fh, snapshot, "config_runtime_behavior", pack_paths, texts, ordinal,
                         max_lines_per_file=args.max_lines_per_file)
        if pack:
            counts["config_runtime_behavior"] += 1
            ordinal += 1

    seed = stable_seed(snapshot["repo_id"], snapshot["base_commit"], args.seed, "llm-qa-packs")
    for pack_paths in choose_symbol_packs(texts, args.symbol_packs_per_repo, args.pack_width, seed):
        pack = emit_pack(out_fh, snapshot, "cross_file_semantic", pack_paths, texts, ordinal,
                         max_lines_per_file=args.max_lines_per_file)
        if pack:
            counts["cross_file_semantic"] += 1
            ordinal += 1
    return counts


def pack_shard_key(snapshot: dict[str, Any]) -> int:
    return stable_seed(snapshot["repo_id"], snapshot.get("base_commit") or snapshot.get("commit_sha", ""),
                       "llm-qa-pack")


def pack_subshard_key(snapshot: dict[str, Any], args: argparse.Namespace) -> int:
    return stable_seed(snapshot["repo_id"], snapshot.get("base_commit") or snapshot.get("commit_sha", ""),
                       "llm-qa-pack-subshard", args.parent_num_shards, args.parent_shard_index)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-embeddings", type=Path,
                   help="Deprecated alias for --snapshot-table.")
    p.add_argument("--snapshot-table", type=Path,
                   help="Parquet table with repo_id/base_commit or repo/commit_sha columns.")
    p.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    p.add_argument("--swefixer-metadata", type=Path, default=DEFAULT_SWEFIXER_METADATA)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--audit-output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--parent-num-shards", type=int, default=0,
                   help="If set, first select this original parent shard, then subshard within it.")
    p.add_argument("--parent-shard-index", type=int, default=0)
    p.add_argument("--limit-repos", type=int, default=0)
    p.add_argument("--max-files-per-repo", type=int, default=0)
    p.add_argument("--max-file-bytes", type=int, default=524288)
    p.add_argument("--max-lines-per-file", type=int, default=160)
    p.add_argument("--pack-width", type=int, default=4)
    p.add_argument("--issue-packs-per-repo", type=int, default=4)
    p.add_argument("--test-impl-packs-per-repo", type=int, default=2)
    p.add_argument("--config-packs-per-repo", type=int, default=2)
    p.add_argument("--symbol-packs-per-repo", type=int, default=4)
    args = p.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard config")
    if args.parent_num_shards:
        if args.parent_num_shards < 1 or not 0 <= args.parent_shard_index < args.parent_num_shards:
            raise SystemExit("invalid parent shard config")
    snapshot_table = args.snapshot_table or args.repo_embeddings
    if snapshot_table is None:
        raise SystemExit("set --snapshot-table or --repo-embeddings")
    snapshots = unique_snapshots(read_snapshot_table(snapshot_table))
    if args.parent_num_shards:
        snapshots = [row for row in snapshots
                     if pack_shard_key(row) % args.parent_num_shards == args.parent_shard_index]
        snapshots = [row for row in snapshots
                     if pack_subshard_key(row, args) % args.num_shards == args.shard_index]
    else:
        snapshots = [row for row in snapshots if pack_shard_key(row) % args.num_shards == args.shard_index]
    if args.limit_repos:
        snapshots = snapshots[:args.limit_repos]
    issues = issue_rows_by_snapshot(read_table(args.swefixer_metadata))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total_counts: Counter[str] = Counter()
    repos_with_packs = 0
    with args.output_jsonl.open("w", encoding="utf-8") as out_fh:
        for index, snapshot in enumerate(snapshots, 1):
            key = (str(snapshot["repo_id"]), str(snapshot.get("base_commit") or snapshot.get("commit_sha", "")))
            counts = build_packs_for_snapshot(snapshot, args.repo_root, issues.get(key, []), args, out_fh)
            total_counts.update(counts)
            repos_with_packs += int(sum(counts.values()) > 0)
            if index % 10 == 0 or index == len(snapshots):
                print(json.dumps({"repos": index, "total_repos": len(snapshots),
                                  "packs": sum(total_counts.values())}), flush=True)
    audit = {
        "rows": sum(total_counts.values()),
        "repos": len(snapshots),
        "repos_with_packs": repos_with_packs,
        "snapshot_table": str(snapshot_table),
        "pack_counts": dict(total_counts),
        "output_jsonl": str(args.output_jsonl),
        "elapsed_sec": round(time.time() - started, 3),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "generation_stack": "not_selected_by_this_script",
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
