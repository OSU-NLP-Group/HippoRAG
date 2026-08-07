#!/usr/bin/env python3
"""Atomically remove repeated content-addressed chunk IDs from RepoQA indexes."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    return parser.parse_args()


def unique(values: list[str] | None) -> list[str]:
    return list(dict.fromkeys(str(value) for value in (values or [])))


def main() -> int:
    args = parse_args()
    paths = sorted(
        Path(path)
        for path in glob.glob(str(args.data_root / "stage*" / "*" / "*.repoqa.parquet"))
    )
    if not paths:
        raise ValueError(f"no RepoQA Parquets under {args.data_root}")
    totals = {"files": len(paths), "rewritten_files": 0, "rows": 0, "changed_rows": 0}
    examples: list[dict[str, object]] = []
    for path in paths:
        table = pq.read_table(path, memory_map=True)
        rows = table.to_pylist()
        changed = 0
        for row in rows:
            old_chunks = list(row.get("chunk_ids") or [])
            old_bm25 = list(row.get("bm25_chunk_ids") or [])
            chunks = unique(old_chunks)
            bm25 = unique(old_bm25)
            if chunks != old_chunks or bm25 != old_bm25:
                changed += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "id": row.get("id"),
                            "file": str(path),
                            "chunks_before": len(old_chunks),
                            "chunks_after": len(chunks),
                            "bm25_before": len(old_bm25),
                            "bm25_after": len(bm25),
                        }
                    )
            row["chunk_ids"] = chunks
            row["bm25_chunk_ids"] = bm25
            if not bool(row.get("use_all_chunks")):
                row["num_selected_chunks"] = len(chunks)
                row["num_evidence_chunks"] = min(
                    int(row["num_evidence_chunks"]), len(chunks)
                )
        totals["rows"] += len(rows)
        totals["changed_rows"] += changed
        if not changed:
            continue
        temporary = path.with_suffix(path.suffix + f".canonical.tmp.{os.getpid()}")
        pq.write_table(
            pa.Table.from_pylist(rows, schema=table.schema),
            temporary,
            compression="zstd",
        )
        temporary.replace(path)
        totals["rewritten_files"] += 1
    result = {**totals, "examples": examples}
    audit = args.data_root / "chunk_id_canonicalization.json"
    audit.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
