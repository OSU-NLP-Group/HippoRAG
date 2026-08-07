#!/usr/bin/env python3
"""Clone/fetch repositories needed by normalized SWE-Fixer rows."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def run(cmd: List[str], *, cwd: Path | None = None, timeout: int = 900) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def load_rows(path: Path, *, limit: int = 0) -> List[Dict[str, Any]]:
    table = pq.read_table(path)
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    out = []
    n = table.num_rows if not limit else min(limit, table.num_rows)
    for i in range(n):
        row = {name: cols[name][i] for name in table.column_names}
        if row.get("usable_for_train", True):
            out.append(row)
    return out


def repo_path(root: Path, repo_id: str) -> Path:
    owner, name = repo_id.split("/", 1)
    return root / owner / name


def ensure_repo(root: Path, repo_id: str) -> Dict[str, Any]:
    path = repo_path(root, repo_id)
    url = f"https://github.com/{repo_id}.git"
    path.parent.mkdir(parents=True, exist_ok=True)
    if (path / ".git").exists():
        code, out, err = run(["git", "remote", "set-url", "origin", url], cwd=path, timeout=60)
        return {"repo_id": repo_id, "repo_path": str(path), "clone_ok": True, "clone_action": "exists", "clone_error": err[-1000:]}
    code, out, err = run(["git", "clone", "--no-checkout", url, str(path)], timeout=1800)
    return {
        "repo_id": repo_id,
        "repo_path": str(path),
        "clone_ok": code == 0,
        "clone_action": "cloned",
        "clone_error": err[-2000:],
    }


def ensure_commit(path: Path, commit: str) -> Tuple[bool, str]:
    code, _, _ = run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=path, timeout=60)
    if code == 0:
        return True, ""
    code, _, err = run(["git", "fetch", "--depth", "1", "origin", commit], cwd=path, timeout=900)
    if code != 0:
        # Fallback can be slower but handles servers that reject sha-only fetch.
        code2, _, err2 = run(["git", "fetch", "--depth", "1", "origin"], cwd=path, timeout=900)
        err = (err + "\n" + err2)[-2000:]
        if code2 != 0:
            return False, err
    code, _, err3 = run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=path, timeout=60)
    return code == 0, err3[-2000:]


def write_rows(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    tmp = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    rows = load_rows(Path(args.input), limit=args.limit)
    needed = sorted({(str(r["repo_id"]), str(r["base_commit"])) for r in rows if r.get("repo_id") and r.get("base_commit")})
    root = Path(args.repo_root)
    output_rows: List[Dict[str, Any]] = []
    repo_status: Dict[str, Dict[str, Any]] = {}
    for idx, (repo_id, commit) in enumerate(needed, 1):
        if repo_id not in repo_status:
            repo_status[repo_id] = ensure_repo(root, repo_id)
        status = dict(repo_status[repo_id])
        commit_ok = False
        commit_error = ""
        if status["clone_ok"]:
            commit_ok, commit_error = ensure_commit(Path(status["repo_path"]), commit)
        output_rows.append({
            "repo_id": repo_id,
            "base_commit": commit,
            "commit_sha": commit,
            "repo_path": status["repo_path"],
            "clone_ok": bool(status["clone_ok"]),
            "commit_ok": bool(commit_ok),
            "usable_repo_snapshot": bool(status["clone_ok"] and commit_ok),
            "error": "" if status["clone_ok"] and commit_ok else (status.get("clone_error", "") + "\n" + commit_error)[-3000:],
        })
        if idx % 10 == 0 or idx == len(needed):
            print(f"repo snapshot {idx}/{len(needed)}", flush=True)
    write_rows(output_rows, Path(args.output))
    print(json.dumps({
        "input_rows": len(rows),
        "unique_repo_commits": len(needed),
        "usable": sum(1 for r in output_rows if r["usable_repo_snapshot"]),
        "output": args.output,
        "elapsed_sec": round(time.time() - started, 3),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
