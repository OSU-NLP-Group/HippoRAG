#!/usr/bin/env python3
# Copyright © 2026, Oracle and/or its affiliates.
"""Build deterministic whole-repository chunks for Doc-to-LoRA training.

The chunking policy is intentionally small and auditable:

* enumerate every eligible text file at a repository commit;
* sort files by repository-relative path;
* greedily pack whole files under a token budget;
* split only a file that cannot fit in an empty chunk.

Output is resumable per repository.  Chunk payloads are content-addressed and
deduplicated across commits of the same repository; snapshot rows retain their
ordered chunk IDs and exact context token counts.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


EXCLUDED_DIR_PARTS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "site-packages",
    ".eggs",
    "vendor",
    "vendors",
    "vendored",
    "_vendor",
    "third_party",
    "thirdparty",
    "external",
    "dist",
    "build",
    "target",
    ".next",
    ".cache",
    "coverage",
    "htmlcov",
}

EXCLUDED_BASENAMES = {
    "package-lock.json",
    "npm-shrinkwrap.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pdm.lock",
    "cargo.lock",
}

BINARY_SUFFIXES = {
    ".7z", ".a", ".avi", ".bin", ".bmp", ".bz2", ".class", ".db",
    ".dll", ".dylib", ".egg", ".eot", ".exe", ".feather", ".gif",
    ".gz", ".ico", ".jar", ".jpeg", ".jpg", ".lockb", ".mp3",
    ".mp4", ".npy", ".npz", ".o", ".otf", ".parquet", ".pdf",
    ".pickle", ".pkl", ".png", ".pyc", ".safetensors", ".so", ".sqlite",
    ".sqlite3", ".tar", ".tif", ".tiff", ".ttf", ".wav", ".webm",
    ".webp", ".whl", ".woff", ".woff2", ".xz", ".zip", ".zst",
}

GENERATED_SUFFIXES = {".map", ".min.css", ".min.js", ".ipynb"}

CHUNK_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("repo_id", pa.string()),
        ("payload_token_count", pa.int32()),
        ("payload_text", pa.large_string()),
        ("file_entries_json", pa.large_string()),
        ("payload_sha256", pa.string()),
    ]
)

SNAPSHOT_SCHEMA = pa.schema(
    [
        ("repo_id", pa.string()),
        ("commit_sha", pa.string()),
        ("base_commit", pa.string()),
        ("snapshot_id", pa.string()),
        ("chunk_index", pa.int32()),
        ("num_chunks", pa.int32()),
        ("chunk_id", pa.string()),
        ("payload_token_count", pa.int32()),
        ("context_token_count", pa.int32()),
        ("num_file_parts", pa.int32()),
    ]
)


@dataclass(frozen=True)
class TreeEntry:
    path: str
    blob_sha: str
    size_bytes: int
    mode: str


@dataclass(frozen=True)
class FilePart:
    path: str
    blob_sha: str
    start_line: int
    end_line: int
    text: str
    rendered: str
    token_count: int
    split_kind: str

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "blob_sha": self.blob_sha,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "token_count": self.token_count,
            "split_kind": self.split_kind,
        }


@dataclass(frozen=True)
class PackedChunk:
    payload_text: str
    payload_token_count: int
    parts: tuple[FilePart, ...]
    chunk_id: str


def stable_int(*parts: Any) -> int:
    raw = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def repo_path(repo_root: Path, repo_id: str) -> Path:
    owner, name = repo_id.split("/", 1)
    return repo_root / owner / name


def safe_repo_name(repo_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_id).strip("_")
    return f"{slug}__{hashlib.sha1(repo_id.encode()).hexdigest()[:10]}"


def excluded_path(path: str) -> str | None:
    pure = PurePosixPath(path)
    lowered_parts = {part.lower() for part in pure.parts[:-1]}
    basename = pure.name.lower()
    lower_path = path.lower()
    if lowered_parts & EXCLUDED_DIR_PARTS:
        return "excluded_directory"
    if basename in EXCLUDED_BASENAMES:
        return "dependency_lockfile"
    if any(lower_path.endswith(suffix) for suffix in GENERATED_SUFFIXES):
        return "generated_or_notebook"
    if pure.suffix.lower() in BINARY_SUFFIXES:
        return "known_binary_extension"
    if basename.startswith(".") and basename in {".coverage", ".ds_store"}:
        return "generated_metadata"
    return None


def text_from_blob(data: bytes) -> tuple[str | None, str | None]:
    sample = data[:65536]
    if b"\x00" in sample:
        return None, "nul_binary"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        replacement_ratio = text.count("\ufffd") / max(1, len(text))
        if replacement_ratio > 0.01:
            return None, "non_utf8"
    if sample:
        printable = sum(
            chr(byte).isprintable() or byte in (9, 10, 12, 13)
            for byte in sample
        )
        if printable / len(sample) < 0.80:
            return None, "low_printable_ratio"
    return text.replace("\r\n", "\n").replace("\r", "\n"), None


def token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return len(ids)


def render_file_part(
    path: str,
    text: str,
    *,
    start_line: int,
    end_line: int,
    is_split: bool,
) -> str:
    quoted_path = json.dumps(path, ensure_ascii=False)
    span = f" lines={start_line}-{end_line}" if is_split else ""
    body = text
    if body and not body.endswith("\n"):
        body += "\n"
    return f"<<<FILE path={quoted_path}{span}>>>\n{body}<<<END FILE>>>\n"


def _make_part(
    tokenizer: Any,
    path: str,
    blob_sha: str,
    text: str,
    start_line: int,
    end_line: int,
    split_kind: str,
) -> FilePart:
    rendered = render_file_part(
        path,
        text,
        start_line=start_line,
        end_line=end_line,
        is_split=split_kind != "whole_file",
    )
    return FilePart(
        path=path,
        blob_sha=blob_sha,
        start_line=start_line,
        end_line=end_line,
        text=text,
        rendered=rendered,
        token_count=token_count(tokenizer, rendered),
        split_kind=split_kind,
    )


def _max_fitting_line_end(
    tokenizer: Any,
    path: str,
    lines: list[str],
    start: int,
    end_limit: int,
    line_offset: int,
    hard_tokens: int,
) -> int:
    lo, hi = start + 1, end_limit
    best = start
    while lo <= hi:
        mid = (lo + hi) // 2
        rendered = render_file_part(
            path,
            "".join(lines[start:mid]),
            start_line=line_offset + start + 1,
            end_line=line_offset + mid,
            is_split=True,
        )
        if token_count(tokenizer, rendered) <= hard_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _split_giant_line(
    tokenizer: Any,
    path: str,
    blob_sha: str,
    line: str,
    line_number: int,
    hard_tokens: int,
) -> list[FilePart]:
    parts: list[FilePart] = []
    start = 0
    while start < len(line):
        lo, hi, best = start + 1, len(line), start
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = _make_part(
                tokenizer, path, blob_sha, line[start:mid], line_number, line_number,
                "character_fallback",
            )
            if candidate.token_count <= hard_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best == start:
            raise ValueError(f"Cannot fit even one character from {path}:{line_number}")
        parts.append(
            _make_part(
                tokenizer, path, blob_sha, line[start:best], line_number, line_number,
                "character_fallback",
            )
        )
        start = best
    return parts


def split_lines_to_parts(
    tokenizer: Any,
    path: str,
    blob_sha: str,
    lines: list[str],
    *,
    line_offset: int,
    hard_tokens: int,
    split_kind: str,
) -> list[FilePart]:
    out: list[FilePart] = []
    start = 0
    while start < len(lines):
        end = _max_fitting_line_end(
            tokenizer, path, lines, start, len(lines), line_offset, hard_tokens
        )
        if end == start:
            out.extend(
                _split_giant_line(
                    tokenizer,
                    path,
                    blob_sha,
                    lines[start],
                    line_offset + start + 1,
                    hard_tokens,
                )
            )
            start += 1
            continue
        # Prefer a nearby blank-line boundary without sacrificing more than 10%.
        minimum = start + max(1, int((end - start) * 0.90))
        for candidate in range(end - 1, minimum - 1, -1):
            if not lines[candidate - 1].strip():
                end = candidate
                break
        text = "".join(lines[start:end])
        out.append(
            _make_part(
                tokenizer,
                path,
                blob_sha,
                text,
                line_offset + start + 1,
                line_offset + end,
                split_kind,
            )
        )
        start = end
    return out


def split_oversized_file(
    tokenizer: Any,
    path: str,
    blob_sha: str,
    text: str,
    hard_tokens: int,
) -> list[FilePart]:
    lines = text.splitlines(keepends=True)
    if not lines:
        lines = [""]

    if path.lower().endswith(".py"):
        try:
            tree = ast.parse(text)
            boundaries = {0, len(lines)}
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    boundaries.add(max(0, node.lineno - 1))
            ordered = sorted(boundaries)
            semantic_parts: list[FilePart] = []
            for start, end in zip(ordered, ordered[1:]):
                segment = "".join(lines[start:end])
                candidate = _make_part(
                    tokenizer,
                    path,
                    blob_sha,
                    segment,
                    start + 1,
                    end,
                    "python_top_level",
                )
                if candidate.token_count <= hard_tokens:
                    semantic_parts.append(candidate)
                else:
                    semantic_parts.extend(
                        split_lines_to_parts(
                            tokenizer,
                            path,
                            blob_sha,
                            lines[start:end],
                            line_offset=start,
                            hard_tokens=hard_tokens,
                            split_kind="line_fallback",
                        )
                    )
            if semantic_parts and all(part.token_count <= hard_tokens for part in semantic_parts):
                return semantic_parts
        except (SyntaxError, ValueError, UnicodeError):
            pass

    return split_lines_to_parts(
        tokenizer,
        path,
        blob_sha,
        lines,
        line_offset=0,
        hard_tokens=hard_tokens,
        split_kind="line_fallback",
    )


def parts_for_file(
    tokenizer: Any,
    path: str,
    blob_sha: str,
    text: str,
    hard_tokens: int,
) -> list[FilePart]:
    line_count = max(1, len(text.splitlines()))
    whole = _make_part(
        tokenizer, path, blob_sha, text, 1, line_count, "whole_file"
    )
    if whole.token_count <= hard_tokens:
        return [whole]
    return split_oversized_file(tokenizer, path, blob_sha, text, hard_tokens)


def directory_key(path: str) -> str:
    parent = PurePosixPath(path).parent.as_posix()
    return parent if parent != "." else ""


def _packed_chunk(tokenizer: Any, parts: list[FilePart], hard_tokens: int) -> PackedChunk:
    payload = "\n".join(part.rendered for part in parts)
    count = token_count(tokenizer, payload)
    if count > hard_tokens:
        raise ValueError(f"Packed chunk exceeds hard limit: {count} > {hard_tokens}")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return PackedChunk(payload, count, tuple(parts), digest)


def pack_file_parts(
    tokenizer: Any,
    parts: Iterable[FilePart],
    *,
    soft_tokens: int,
    hard_tokens: int,
    packing_safety_tokens: int = 1024,
) -> list[PackedChunk]:
    """Greedily pack path-ordered file parts without splitting fitting files."""
    ordered = list(parts)
    if any(a.path > b.path for a, b in zip(ordered, ordered[1:])):
        raise ValueError("File parts must be in repository-path order")
    safe_hard = hard_tokens - packing_safety_tokens
    if safe_hard <= soft_tokens:
        raise ValueError("packing safety margin leaves no room above soft target")

    queue: deque[FilePart] = deque(ordered)
    current: list[FilePart] = []
    approximate = 0
    chunks: list[PackedChunk] = []

    def emit() -> None:
        nonlocal current, approximate
        if not current:
            return
        # The safety margin should make this unnecessary, but retain order and
        # requeue tail items if tokenizer boundary effects exceed the estimate.
        carry: list[FilePart] = []
        while current:
            try:
                chunks.append(_packed_chunk(tokenizer, current, hard_tokens))
                break
            except ValueError:
                if len(current) == 1:
                    raise
                carry.append(current.pop())
        for item in carry:
            queue.appendleft(item)
        current = []
        approximate = 0

    while queue:
        part = queue.popleft()
        if part.token_count > hard_tokens:
            raise ValueError(f"Oversized part was not split: {part.path}")
        entering_new_directory = bool(current) and (
            directory_key(part.path) != directory_key(current[-1].path)
        )
        if current and approximate >= soft_tokens and entering_new_directory:
            queue.appendleft(part)
            emit()
            continue
        separator_cost = 1 if current else 0
        if current and approximate + separator_cost + part.token_count > safe_hard:
            queue.appendleft(part)
            emit()
            continue
        current.append(part)
        approximate += separator_cost + part.token_count
    emit()
    return chunks


def render_repository_header(repo_id: str, commit: str, index: int, total: int) -> str:
    return (
        f"<<<REPOSITORY name={json.dumps(repo_id, ensure_ascii=False)} "
        f"commit={commit} chunk={index + 1}/{total}>>>\n"
    )


def parse_ls_tree(data: bytes) -> list[TreeEntry]:
    entries: list[TreeEntry] = []
    for raw in data.split(b"\0"):
        if not raw:
            continue
        metadata, sep, path_bytes = raw.partition(b"\t")
        if not sep:
            continue
        fields = metadata.decode("ascii", errors="replace").split()
        if len(fields) != 4:
            continue
        mode, obj_type, sha, size_raw = fields
        if obj_type != "blob" or size_raw == "-":
            continue
        path = path_bytes.decode("utf-8", errors="replace")
        entries.append(TreeEntry(path, sha, int(size_raw), mode))
    return sorted(entries, key=lambda entry: entry.path)


def list_tree(repo: Path, commit: str, timeout: int) -> tuple[list[TreeEntry], str | None]:
    proc = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "-z", "-l", commit],
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        return [], proc.stderr.decode("utf-8", errors="replace")[-2000:]
    return parse_ls_tree(proc.stdout), None


class GitBatchReader:
    def __init__(self, repo: Path):
        self.proc = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def read(self, sha: str) -> bytes:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(sha.encode("ascii") + b"\n")
        self.proc.stdin.flush()
        header = self.proc.stdout.readline()
        fields = header.rstrip(b"\n").split()
        if len(fields) == 2 and fields[1] == b"missing":
            raise KeyError(sha)
        if len(fields) != 3:
            raise RuntimeError(f"Unexpected git cat-file header: {header!r}")
        size = int(fields[2])
        data = self.proc.stdout.read(size)
        newline = self.proc.stdout.read(1)
        if len(data) != size or newline != b"\n":
            raise RuntimeError(f"Truncated git cat-file response for {sha}")
        return data

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=5)

    def __enter__(self) -> "GitBatchReader":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class FilePartCache:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.items: OrderedDict[tuple[str, str], tuple[list[FilePart], int]] = OrderedDict()

    def get(self, key: tuple[str, str]) -> list[FilePart] | None:
        found = self.items.get(key)
        if found is None:
            return None
        self.items.move_to_end(key)
        return found[0]

    def put(self, key: tuple[str, str], value: list[FilePart]) -> None:
        size = sum(len(part.rendered.encode("utf-8")) for part in value)
        if size > self.max_bytes:
            return
        previous = self.items.pop(key, None)
        if previous is not None:
            self.current_bytes -= previous[1]
        while self.items and self.current_bytes + size > self.max_bytes:
            _, (_, evicted_size) = self.items.popitem(last=False)
            self.current_bytes -= evicted_size
        self.items[key] = (value, size)
        self.current_bytes += size


class BufferedParquetWriter:
    def __init__(self, path: Path, schema: pa.Schema, batch_rows: int = 128):
        self.path = path
        self.schema = schema
        self.batch_rows = batch_rows
        self.buffer: list[dict[str, Any]] = []
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")

    def write(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()

    def __enter__(self) -> "BufferedParquetWriter":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def load_snapshots(path: Path) -> dict[str, list[dict[str, str]]]:
    names = set(pq.ParquetFile(path).schema.names)
    columns = [name for name in ("repo_id", "repo", "commit_sha", "base_commit") if name in names]
    table = pq.read_table(path, columns=columns, memory_map=True)
    rows = table.to_pylist()
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        rid = str(row.get("repo_id") or row.get("repo") or "")
        commit = str(row.get("commit_sha") or row.get("base_commit") or "")
        base = str(row.get("base_commit") or commit)
        if rid and commit:
            grouped[rid].setdefault(
                commit,
                {"repo_id": rid, "commit_sha": commit, "base_commit": base},
            )
    return {
        rid: sorted(commits.values(), key=lambda row: row["commit_sha"])
        for rid, commits in grouped.items()
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def process_repository(
    repo_id: str,
    snapshots: list[dict[str, str]],
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    destination = args.output_root / "repositories" / safe_repo_name(repo_id)
    destination.mkdir(parents=True, exist_ok=True)
    audit_path = destination / "audit.json"
    if args.resume and audit_path.exists():
        try:
            old = json.loads(audit_path.read_text(encoding="utf-8"))
            if old.get("status") in {"complete", "complete_with_errors"}:
                return {"repo_id": repo_id, "status": "resumed_skip", "destination": str(destination)}
        except (OSError, json.JSONDecodeError):
            pass

    repo = repo_path(args.repo_root, repo_id)
    if not (repo / ".git").exists():
        result = {"repo_id": repo_id, "status": "missing_repo", "repo_path": str(repo)}
        atomic_write_json(audit_path, result)
        return result

    if args.limit_snapshots_per_repo:
        snapshots = snapshots[: args.limit_snapshots_per_repo]

    chunks_final = destination / "chunks.parquet"
    snapshots_final = destination / "snapshots.parquet"
    chunks_tmp = destination / f"chunks.parquet.tmp.{os.getpid()}"
    snapshots_tmp = destination / f"snapshots.parquet.tmp.{os.getpid()}"
    for temp in (chunks_tmp, snapshots_tmp):
        temp.unlink(missing_ok=True)

    counters: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    seen_chunk_ids: set[str] = set()
    cache = FilePartCache(args.file_cache_mb * 1024 * 1024)

    try:
        with (
            GitBatchReader(repo) as blobs,
            BufferedParquetWriter(chunks_tmp, CHUNK_SCHEMA, args.parquet_batch_rows) as chunk_writer,
            BufferedParquetWriter(snapshots_tmp, SNAPSHOT_SCHEMA, args.parquet_batch_rows) as snapshot_writer,
        ):
            for snapshot_number, snapshot in enumerate(snapshots, 1):
                commit = snapshot["commit_sha"]
                entries, tree_error = list_tree(repo, commit, args.git_timeout)
                if tree_error:
                    counters["failed_snapshots"] += 1
                    errors.append({"commit": commit, "error": tree_error})
                    continue

                file_parts: list[FilePart] = []
                per_snapshot_exclusions: Counter[str] = Counter()
                for entry in entries:
                    counters["tree_entries"] += 1
                    exclusion = excluded_path(entry.path)
                    if exclusion:
                        counters[f"excluded_{exclusion}"] += 1
                        per_snapshot_exclusions[exclusion] += 1
                        continue
                    if entry.size_bytes > args.max_file_bytes:
                        counters["excluded_file_safety_limit"] += 1
                        per_snapshot_exclusions["file_safety_limit"] += 1
                        continue
                    cache_key = (entry.path, entry.blob_sha)
                    parts = cache.get(cache_key)
                    if parts is None:
                        try:
                            data = blobs.read(entry.blob_sha)
                        except (KeyError, RuntimeError) as exc:
                            counters["excluded_blob_read_error"] += 1
                            per_snapshot_exclusions["blob_read_error"] += 1
                            if len(errors) < args.max_recorded_errors:
                                errors.append({"commit": commit, "path": entry.path, "error": repr(exc)})
                            continue
                        text, decode_error = text_from_blob(data)
                        if text is None:
                            counters[f"excluded_{decode_error}"] += 1
                            per_snapshot_exclusions[str(decode_error)] += 1
                            continue
                        parts = parts_for_file(
                            tokenizer, entry.path, entry.blob_sha, text, args.hard_tokens
                        )
                        cache.put(cache_key, parts)
                        counters["tokenized_unique_path_blobs"] += 1
                    else:
                        counters["file_cache_hits"] += 1
                    file_parts.extend(parts)

                file_parts.sort(key=lambda part: (part.path, part.start_line))
                if not file_parts:
                    counters["empty_snapshots"] += 1
                    errors.append({"commit": commit, "error": "no_eligible_text_files"})
                    continue

                chunks = pack_file_parts(
                    tokenizer,
                    file_parts,
                    soft_tokens=args.soft_tokens,
                    hard_tokens=args.hard_tokens,
                    packing_safety_tokens=args.packing_safety_tokens,
                )
                total = len(chunks)
                snapshot_id = f"{repo_id}@{commit}"
                for index, chunk in enumerate(chunks):
                    if chunk.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk.chunk_id)
                        chunk_writer.write(
                            {
                                "chunk_id": chunk.chunk_id,
                                "repo_id": repo_id,
                                "payload_token_count": chunk.payload_token_count,
                                "payload_text": chunk.payload_text,
                                "file_entries_json": json.dumps(
                                    [part.manifest_entry() for part in chunk.parts],
                                    sort_keys=True,
                                    ensure_ascii=False,
                                ),
                                "payload_sha256": chunk.chunk_id,
                            }
                        )
                        counters["unique_chunks"] += 1
                    else:
                        counters["deduplicated_chunk_references"] += 1

                    header = render_repository_header(repo_id, commit, index, total)
                    context_count = token_count(tokenizer, header + chunk.payload_text)
                    if context_count + args.doc_to_lora_affix_tokens > args.model_max_tokens:
                        raise ValueError(
                            f"Final context exceeds model limit for {snapshot_id} chunk {index}: "
                            f"{context_count}+{args.doc_to_lora_affix_tokens}>{args.model_max_tokens}"
                        )
                    snapshot_writer.write(
                        {
                            "repo_id": repo_id,
                            "commit_sha": commit,
                            "base_commit": snapshot["base_commit"],
                            "snapshot_id": snapshot_id,
                            "chunk_index": index,
                            "num_chunks": total,
                            "chunk_id": chunk.chunk_id,
                            "payload_token_count": chunk.payload_token_count,
                            "context_token_count": context_count + args.doc_to_lora_affix_tokens,
                            "num_file_parts": len(chunk.parts),
                        }
                    )
                    counters["snapshot_chunk_references"] += 1
                    counters["context_tokens"] += context_count + args.doc_to_lora_affix_tokens

                counters["processed_snapshots"] += 1
                counters["eligible_file_parts"] += len(file_parts)
                counters["source_files"] += len({part.path for part in file_parts})
                counters["split_file_parts"] += sum(part.split_kind != "whole_file" for part in file_parts)
                if args.progress_every and snapshot_number % args.progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "repo_id": repo_id,
                                "snapshot": snapshot_number,
                                "snapshots": len(snapshots),
                                "unique_chunks": counters["unique_chunks"],
                                "chunk_references": counters["snapshot_chunk_references"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

        chunks_tmp.replace(chunks_final)
        snapshots_tmp.replace(snapshots_final)
    except Exception as exc:
        for temp in (chunks_tmp, snapshots_tmp):
            temp.unlink(missing_ok=True)
        result = {
            "repo_id": repo_id,
            "status": "failed",
            "error": repr(exc),
            "elapsed_sec": round(time.time() - started, 3),
            "counters": dict(counters),
            "errors": errors[: args.max_recorded_errors],
        }
        atomic_write_json(audit_path, result)
        return result

    status = "complete_with_errors" if counters["failed_snapshots"] else "complete"
    result = {
        "repo_id": repo_id,
        "status": status,
        "repo_path": str(repo),
        "snapshots_requested": len(snapshots),
        "elapsed_sec": round(time.time() - started, 3),
        "counters": dict(counters),
        "errors": errors[: args.max_recorded_errors],
        "chunks_parquet": str(chunks_final),
        "snapshots_parquet": str(snapshots_final),
        "config": {
            "model_name": args.model_name,
            "soft_tokens": args.soft_tokens,
            "hard_tokens": args.hard_tokens,
            "model_max_tokens": args.model_max_tokens,
            "doc_to_lora_affix_tokens": args.doc_to_lora_affix_tokens,
            "max_file_bytes": args.max_file_bytes,
        },
    }
    atomic_write_json(audit_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-static", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-name", default="google/gemma-4-E2B-it")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--soft-tokens", type=int, default=120_000)
    parser.add_argument("--hard-tokens", type=int, default=130_000)
    parser.add_argument("--model-max-tokens", type=int, default=131_072)
    parser.add_argument("--doc-to-lora-affix-tokens", type=int, default=9)
    parser.add_argument("--packing-safety-tokens", type=int, default=1_024)
    parser.add_argument("--max-file-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--file-cache-mb", type=int, default=512)
    parser.add_argument("--git-timeout", type=int, default=600)
    parser.add_argument("--repo-num-shards", type=int, default=1)
    parser.add_argument("--repo-shard-index", type=int, default=0)
    parser.add_argument("--limit-repos", type=int, default=0)
    parser.add_argument(
        "--exclude-repo",
        action="append",
        default=[],
        help="Exact repository ID to omit; repeat for multiple audited exclusions.",
    )
    parser.add_argument("--limit-snapshots-per-repo", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--parquet-batch-rows", type=int, default=128)
    parser.add_argument("--max-recorded-errors", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.repo_shard_index < args.repo_num_shards:
        parser.error("repo-shard-index must be in [0, repo-num-shards)")
    if not 0 < args.soft_tokens < args.hard_tokens < args.model_max_tokens:
        parser.error("require 0 < soft-tokens < hard-tokens < model-max-tokens")
    return args


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    grouped = load_snapshots(args.input_static)
    selected = [
        repo_id
        for repo_id in sorted(grouped)
        if (
            stable_int(repo_id, "doc-to-lora-chunks") % args.repo_num_shards
            == args.repo_shard_index
            and repo_id not in set(args.exclude_repo)
        )
    ]
    if args.limit_repos:
        selected = selected[: args.limit_repos]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    summary = Counter()
    results: list[dict[str, Any]] = []
    for index, repo_id in enumerate(selected, 1):
        result = process_repository(repo_id, grouped[repo_id], tokenizer, args)
        results.append(result)
        summary[str(result["status"])] += 1
        print(
            json.dumps(
                {
                    "repo": index,
                    "repos": len(selected),
                    "repo_id": repo_id,
                    "status": result["status"],
                    "elapsed_sec": result.get("elapsed_sec"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    source_tag = args.input_static.stem
    summary_path = (
        args.output_root
        / "task_summaries"
        / f"{source_tag}.repo_subshard_{args.repo_shard_index}_of_{args.repo_num_shards}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "input_static": str(args.input_static),
        "repo_root": str(args.repo_root),
        "output_root": str(args.output_root),
        "repo_shard_index": args.repo_shard_index,
        "repo_num_shards": args.repo_num_shards,
        "repos_selected": len(selected),
        "excluded_repositories": sorted(set(args.exclude_repo)),
        "status_counts": dict(summary),
        "elapsed_sec": round(time.time() - started, 3),
        "results": results,
    }
    atomic_write_json(summary_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 1 if summary.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
