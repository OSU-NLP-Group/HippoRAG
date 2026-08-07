#!/usr/bin/env python3
"""Split a static issue-fix table into train/val parquet files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_subset(table: pa.Table, indices: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = table.take(pa.array(indices, type=pa.int64()))
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(subset, tmp, compression="zstd")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    table = pq.read_table(args.input)
    indices = list(range(table.num_rows))
    random.Random(args.seed).shuffle(indices)
    n_val = max(1, int(round(table.num_rows * args.val_fraction))) if table.num_rows > 1 else 0
    val_idx = sorted(indices[:n_val])
    train_idx = sorted(indices[n_val:])
    write_subset(table, train_idx, Path(args.train_output))
    write_subset(table, val_idx, Path(args.val_output))
    print(json.dumps({
        "input": args.input,
        "rows": table.num_rows,
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "train_output": args.train_output,
        "val_output": args.val_output,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
