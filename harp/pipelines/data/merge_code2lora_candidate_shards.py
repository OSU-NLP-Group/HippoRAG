#!/usr/bin/env python3
"""Stream candidate parquet shards into one table without a row cap."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-inputs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = sorted({Path(match) for pattern in args.inputs for match in glob.glob(pattern)})
    if args.expected_inputs and len(paths) != args.expected_inputs:
        raise RuntimeError(f"expected {args.expected_inputs} shards, found {len(paths)}: {paths}")
    if not paths:
        raise FileNotFoundError("no candidate shards matched")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    schema = pq.ParquetFile(paths[0]).schema_arrow
    rows = 0
    repos: set[str] = set()
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    try:
        for path in paths:
            parquet = pq.ParquetFile(path)
            if parquet.schema_arrow != schema:
                raise RuntimeError(f"schema mismatch: {path}")
            for batch in parquet.iter_batches(batch_size=16384):
                table = pa.Table.from_batches([batch])
                rows += table.num_rows
                repos.update(str(value) for value in table.column("repo_id").to_pylist())
                writer.write_table(table)
    finally:
        writer.close()
    tmp.replace(args.output)
    audit = {"inputs": [str(path) for path in paths], "output": str(args.output),
             "rows": rows, "repos": len(repos), "row_cap": None}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
