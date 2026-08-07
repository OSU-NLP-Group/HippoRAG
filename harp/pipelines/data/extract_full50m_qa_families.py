#!/usr/bin/env python3
"""Split the full Doc-to-LoRA QA parquet corpus by source family.

This intentionally copies only the QA rows and their columns.  It does not
copy repository chunks, group manifests, training schedules, or checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


AST_FAMILIES = {"deterministic_original", "deterministic_cross_file_aug"}
LLM_FAMILIES = {"llm_generated", "llm_rewrite"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ast-output", type=Path, required=True)
    parser.add_argument("--llm-output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=131_072)
    return parser.parse_args()


def output_path(source: Path, source_root: Path, output_root: Path) -> Path:
    relative = source.relative_to(source_root)
    return output_root / relative


def write_subset(
    source: Path,
    source_root: Path,
    output_root: Path,
    families: set[str],
    batch_rows: int,
) -> tuple[int, Counter[str]]:
    destination = output_path(source, source_root, output_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()

    reader = pq.ParquetFile(source)
    writer: pq.ParquetWriter | None = None
    count = 0
    family_counts: Counter[str] = Counter()
    try:
        for batch in reader.iter_batches(batch_size=batch_rows):
            table = pa.Table.from_batches([batch])
            families_column = table["source_family"]
            mask = pc.is_in(families_column, value_set=pa.array(sorted(families)))
            selected = table.filter(mask)
            if not selected.num_rows:
                continue
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    selected.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.write_table(selected, row_group_size=batch_rows)
            count += selected.num_rows
            family_counts.update(selected["source_family"].to_pylist())
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        if temporary.exists():
            temporary.unlink()
        return 0, family_counts
    os.replace(temporary, destination)
    return count, family_counts


def main() -> int:
    args = parse_args()
    ready_path = args.source / "READY.json"
    ready = json.loads(ready_path.read_text())
    sources = sorted(
        Path(path)
        for partition in ready["partitions"].values()
        for path in partition.get("qa_files", [])
    )
    if not sources:
        raise FileNotFoundError(f"No QA shards declared by {ready_path}")

    totals: dict[str, object] = {"source": str(args.source), "outputs": {}}
    for name, output_root, families in (
        ("ast", args.ast_output, AST_FAMILIES),
        ("llm", args.llm_output, LLM_FAMILIES),
    ):
        output_root.mkdir(parents=True, exist_ok=True)
        rows = 0
        counts: Counter[str] = Counter()
        files = 0
        for index, source in enumerate(sources, start=1):
            selected, selected_counts = write_subset(
                source, args.source, output_root, families, args.batch_rows
            )
            rows += selected
            counts.update(selected_counts)
            files += int(selected > 0)
            print(f"{name} {index}/{len(sources)} {source.name}: {selected} rows", flush=True)
        totals["outputs"][name] = {
            "rows": rows,
            "files": files,
            "source_families": dict(sorted(counts.items())),
            "included_columns": pq.ParquetFile(sources[0]).schema_arrow.names,
        }

    for output_root in (args.ast_output, args.llm_output):
        (output_root / "MANIFEST.json").write_text(
            json.dumps(totals, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(totals, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
