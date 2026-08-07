#!/usr/bin/env python3
"""Merge issue-fix static parquet shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-glob", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(Path().glob(args.shard_glob) if not args.shard_glob.startswith("/") else Path("/").glob(args.shard_glob[1:]))
    if not paths:
        raise RuntimeError(f"No shards matched: {args.shard_glob}")

    tables = []
    rows_by_shard = {}
    for path in paths:
        table = pq.read_table(path)
        rows_by_shard[path.name] = table.num_rows
        if table.num_rows == 0 and not table.column_names:
            continue
        tables.append(table)
    if not tables:
        raise RuntimeError(f"All matched shards were empty: {args.shard_glob}")

    merged = pa.concat_tables(tables, promote_options="default")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(merged, tmp, compression="zstd")
    tmp.replace(output)
    print(json.dumps({
        "output": str(output),
        "shards": len(paths),
        "rows": merged.num_rows,
        "rows_by_shard": rows_by_shard,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
