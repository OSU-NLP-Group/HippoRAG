#!/usr/bin/env python3
"""Build static issue-fix training rows with BGE-M3 repo-state embeddings."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
CHUNK_TOKENS = 2048
CHUNK_OVERLAP = 256
MAX_FILE_BYTES = 2_000_000
MIN_WINDOW_TOKENS = 8


def run(cmd: List[str], *, cwd: Path, timeout: int = 300) -> Tuple[int, bytes, bytes]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def load_table(path: Path) -> List[Dict[str, Any]]:
    table = pq.read_table(path)
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    return [{name: cols[name][i] for name in table.column_names} for i in range(table.num_rows)]


def write_table(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def processed_instance_ids(rows: List[Dict[str, Any]]) -> set[str]:
    return {str(row["instance_id"]) for row in rows if row.get("instance_id") is not None}


def safe_repo(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def stable_shard(value: str, num_shards: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def ls_tree_py(repo_path: Path, commit: str) -> List[Tuple[str, str, int]]:
    code, out, _ = run(["git", "ls-tree", "-r", "-l", commit], cwd=repo_path, timeout=300)
    if code != 0:
        return []
    files = []
    for raw in out.decode("utf-8", errors="ignore").splitlines():
        try:
            meta, path = raw.split("\t", 1)
            _mode, ftype, blob, size_s = meta.split()
        except ValueError:
            continue
        if ftype != "blob" or not path.endswith(".py"):
            continue
        try:
            size = int(size_s)
        except ValueError:
            continue
        if 0 < size <= MAX_FILE_BYTES:
            files.append((blob, path, size))
    return files


def cat_blob(repo_path: Path, blob: str) -> str:
    code, out, _ = run(["git", "cat-file", "blob", blob], cwd=repo_path, timeout=300)
    return out.decode("utf-8", errors="ignore") if code == 0 else ""


def chunk_ids(ids: List[int], *, chunk_tokens: int, overlap: int) -> List[List[int]]:
    step = chunk_tokens - overlap
    windows = []
    for start in range(0, len(ids), step):
        end = min(len(ids), start + chunk_tokens)
        window = ids[start:end]
        if len(window) >= MIN_WINDOW_TOKENS:
            windows.append(window)
        if end >= len(ids):
            break
    return windows


@torch.inference_mode()
def embed_texts(texts: List[str], tokenizer, model, *, device: str, batch_size: int, max_length: int) -> torch.Tensor:
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {key: value.to(device) for key, value in enc.items()}
        out = model(**enc)
        last = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        outputs.append(pooled.detach().to(torch.float32).cpu())
    hidden = int(getattr(model.config, "hidden_size", 1024))
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, hidden), dtype=torch.float32)


def embed_blob_text(text: str, tokenizer, model, *, device: str, batch_size: int) -> np.ndarray | None:
    ids = tokenizer.encode(text or "", add_special_tokens=False)
    windows = chunk_ids(ids, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP)
    if not windows:
        return None
    chunks = [tokenizer.decode(window, skip_special_tokens=True) for window in windows]
    chunk_embs = embed_texts(chunks, tokenizer, model, device=device, batch_size=batch_size, max_length=CHUNK_TOKENS)
    if chunk_embs.numel() == 0:
        return None
    return chunk_embs.mean(dim=0).numpy().astype(np.float32)


def load_blob_cache(cache_dir: Path) -> Tuple[Dict[str, int], np.ndarray | None]:
    shas = cache_dir / "blob_shas.txt"
    arr_path = cache_dir / "blob_embeddings.f16.npy"
    if not shas.exists() or not arr_path.exists():
        return {}, None
    idx = {}
    with shas.open() as fh:
        for i, line in enumerate(fh):
            idx[line.strip()] = i
    try:
        arr = np.load(arr_path)
    except (EOFError, ValueError, OSError):
        return {}, None
    if arr.shape[0] != len(idx):
        return {}, None
    return idx, arr


def save_blob_cache(cache_dir: Path, shas: List[str], arr: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    arr_tmp = cache_dir / f"blob_embeddings.f16.npy.{pid}.tmp"
    shas_tmp = cache_dir / f"blob_shas.txt.{pid}.tmp"
    with arr_tmp.open("wb") as fh:
        np.save(fh, arr.astype(np.float16))
    shas_tmp.write_text("\n".join(shas) + "\n")
    arr_tmp.replace(cache_dir / "blob_embeddings.f16.npy")
    shas_tmp.replace(cache_dir / "blob_shas.txt")


@contextlib.contextmanager
def repo_cache_lock(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".cache.lock"
    with lock_path.open("w") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def repo_vec_for_commit(repo_path: Path, repo_id: str, commit: str, cache_root: Path, tokenizer, model, *, device: str, batch_size: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    files = ls_tree_py(repo_path, commit)
    hidden = int(getattr(model.config, "hidden_size", 1024))
    if not files:
        return np.zeros((hidden * 2,), dtype=np.float32), {"num_files_used": 0, "files_used": []}
    cache_dir = cache_root / safe_repo(repo_id)
    with repo_cache_lock(cache_dir):
        cached_idx, cached_arr = load_blob_cache(cache_dir)
        all_shas = list(cached_idx.keys())
        all_vecs = [] if cached_arr is None else [cached_arr[i].astype(np.float32) for i in range(cached_arr.shape[0])]
        sha_to_idx = dict(cached_idx)
        missing = []
        for blob, _path, _size in files:
            if blob not in sha_to_idx and blob not in missing:
                missing.append(blob)
        for i, blob in enumerate(missing, 1):
            vec = embed_blob_text(cat_blob(repo_path, blob), tokenizer, model, device=device, batch_size=batch_size)
            if vec is None:
                vec = np.zeros((hidden,), dtype=np.float32)
            sha_to_idx[blob] = len(all_vecs)
            all_shas.append(blob)
            all_vecs.append(vec)
            if i % 100 == 0 or i == len(missing):
                print(f"  {repo_id}@{commit[:8]} embedded missing blobs {i}/{len(missing)}", flush=True)
        if missing:
            save_blob_cache(cache_dir, all_shas, np.stack(all_vecs, axis=0))
    idxs = [sha_to_idx[blob] for blob, _path, _size in files if blob in sha_to_idx]
    if not idxs:
        return np.zeros((hidden * 2,), dtype=np.float32), {"num_files_used": 0, "files_used": []}
    mat = np.stack([all_vecs[i].astype(np.float32) for i in idxs], axis=0)
    repo_vec = np.concatenate([mat.mean(axis=0), mat.max(axis=0)], axis=0)
    repo_vec = repo_vec / (np.linalg.norm(repo_vec) + 1e-12)
    return repo_vec.astype(np.float32), {
        "num_files_used": len(idxs),
        "files_used": [path for _blob, path, _size in files[:200]],
        "missing_blobs_embedded": len(missing),
    }


def torch_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--repo-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-cache-root", required=True)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--shard-key",
        default="repo_id",
        help="Row key used for stable sharding. Training defaults to repo_id; eval jobs may prefer instance_id.",
    )
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    rows = load_table(Path(args.rows))
    manifest = load_table(Path(args.repo_manifest))
    manifest_by_key = {(str(r["repo_id"]), str(r["base_commit"])): r for r in manifest}
    if args.limit:
        rows = rows[:args.limit]
    if args.num_shards > 1:
        before = len(rows)
        rows = [
            row
            for row in rows
            if stable_shard(str(row.get(args.shard_key) or row["repo_id"]), args.num_shards) == args.shard_index
        ]
        print(
            f"repo-shard {args.shard_index}/{args.num_shards}: selected rows={len(rows)} from total={before}",
            flush=True,
        )
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model, trust_remote_code=True, use_fast=True)
    model = AutoModel.from_pretrained(args.embed_model, trust_remote_code=True, torch_dtype=torch_dtype(args.dtype)).to(device)
    model.eval()

    output_path = Path(args.output)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if output_path.exists():
        out_rows = load_table(output_path)
        print(f"resuming from existing output rows={len(out_rows)} path={output_path}", flush=True)
    elif partial_path.exists():
        out_rows = load_table(partial_path)
        print(f"resuming from partial rows={len(out_rows)} path={partial_path}", flush=True)
    else:
        out_rows = []
    done_instance_ids = processed_instance_ids(out_rows)
    started = time.time()
    for idx, row in enumerate(rows, 1):
        instance_id = str(row["instance_id"])
        if instance_id in done_instance_ids:
            if idx % 1000 == 0:
                print(f"skipped already-written rows through {idx}/{len(rows)} wrote={len(out_rows)}", flush=True)
            continue
        key = (str(row["repo_id"]), str(row["base_commit"]))
        m = manifest_by_key.get(key)
        if not m or not m.get("usable_repo_snapshot"):
            continue
        vec, meta = repo_vec_for_commit(
            Path(str(m["repo_path"])),
            str(row["repo_id"]),
            str(row["base_commit"]),
            Path(args.embedding_cache_root),
            tokenizer,
            model,
            device=device,
            batch_size=args.batch_size,
        )
        out_row = {
            "repo_id": row["repo_id"],
            "repo": row.get("repo", row["repo_id"]),
            "commit_sha": row.get("commit_sha", row["base_commit"]),
            "base_commit": row["base_commit"],
            "instance_id": row["instance_id"],
            "source_dataset": row.get("source_dataset", "swe-fixer-train-110k"),
            "prompt": row.get("prompt", ""),
            "target_patch": row.get("target_patch", ""),
            "problem_statement": row.get("problem_statement", ""),
            "repo_state_embedding": vec.astype(np.float16).tolist(),
            "embedding_model": args.embed_model,
            "embedding_metadata_json": json.dumps(meta, sort_keys=True),
        }
        out_rows.append(out_row)
        done_instance_ids.add(instance_id)
        if idx % 10 == 0 or idx == len(rows):
            print(f"embedded rows {idx}/{len(rows)} wrote={len(out_rows)} elapsed_min={(time.time()-started)/60:.2f}", flush=True)
        if args.checkpoint_every and len(out_rows) % args.checkpoint_every == 0:
            write_table(out_rows, partial_path)
            print(f"checkpoint rows={len(out_rows)} path={partial_path}", flush=True)
    write_table(out_rows, output_path)
    if partial_path.exists():
        partial_path.unlink()
    print(json.dumps({"rows": len(out_rows), "output": args.output, "elapsed_sec": round(time.time() - started, 3)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
