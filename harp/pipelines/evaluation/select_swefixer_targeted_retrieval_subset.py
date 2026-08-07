#!/usr/bin/env python3
"""Select four hard static questions per held-out SWE-fixer repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FAMILIES = (
    "cross_file_relationships",
    "behavior_test_spec",
    "imports_exports_entrypoints",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--per-repository", type=int, default=4)
    parser.add_argument(
        "--seed", default="swefixer-targeted-retrieval-step500-v1"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def key(seed: str, *parts: Any) -> str:
    return hashlib.sha256(
        "\0".join([seed, *(str(part) for part in parts)]).encode("utf-8")
    ).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    for path in (args.output_jsonl, args.manifest_json):
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
    rows = [
        json.loads(line)
        for line in args.input_jsonl.open(encoding="utf-8")
        if line.strip()
    ]
    by_repo_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    repositories = sorted({str(row["repo_id"]) for row in rows})
    for row in rows:
        if bool(row.get("negative")) or row.get("family") not in FAMILIES:
            continue
        by_repo_family[(str(row["repo_id"]), str(row["family"]))].append(row)

    selected: list[dict[str, Any]] = []
    repo_counts: Counter[str] = Counter()
    for repo_id in repositories:
        ordered: dict[str, list[dict[str, Any]]] = {}
        for family in FAMILIES:
            ordered[family] = sorted(
                by_repo_family[(repo_id, family)],
                key=lambda row: key(args.seed, repo_id, family, row["fact_id"]),
            )
        cursor = Counter()
        while repo_counts[repo_id] < args.per_repository:
            made_progress = False
            for family in FAMILIES:
                index = cursor[family]
                if index >= len(ordered[family]):
                    continue
                selected.append(ordered[family][index])
                cursor[family] += 1
                repo_counts[repo_id] += 1
                made_progress = True
                if repo_counts[repo_id] == args.per_repository:
                    break
            if not made_progress:
                raise RuntimeError(
                    f"{repo_id} has only {repo_counts[repo_id]} eligible hard facts"
                )

    selected.sort(key=lambda row: (row["repo_id"], row["family"], row["fact_id"]))
    atomic_text(
        args.output_jsonl,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected
        ),
    )
    manifest = {
        "format": "swefixer_targeted_retrieval_subset_v1",
        "seed": args.seed,
        "input_jsonl": str(args.input_jsonl.resolve()),
        "input_sha256": sha256(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "output_sha256": sha256(args.output_jsonl),
        "repositories": len(repositories),
        "facts": len(selected),
        "per_repository": dict(sorted(repo_counts.items())),
        "family_counts": dict(
            sorted(Counter(row["family"] for row in selected).items())
        ),
    }
    atomic_text(
        args.manifest_json,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
