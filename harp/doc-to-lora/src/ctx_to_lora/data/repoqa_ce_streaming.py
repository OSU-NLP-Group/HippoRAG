"""Production, pretokenized RepoQA CE streaming dataset.

Unlike :mod:`repoqa_lazy`, this loader never builds a Hugging Face Dataset or
a Python tuple for every source QA.  It permutes immutable context groups,
reads only the referenced Parquet row group, and packs pretokenized logical QAs
that share the exact same repository adapter.
"""

from __future__ import annotations

import bisect
import glob
import hashlib
import json
import math
import os
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ctx_to_lora.data.definitions import IGNORE_INDEX
from ctx_to_lora.data.repoqa_lazy import ByteLRU, render_repository_header


READY_FORMAT = "doc_to_lora_repoqa_ce_full_v2"
SNAPSHOT_MEMORY_READY_FORMAT = "doc_to_lora_repoqa_snapshot_memory_v1"
READY_FORMATS = {READY_FORMAT, SNAPSHOT_MEMORY_READY_FORMAT}
BASE_REQUIRED_GROUP_COLUMNS = {
    "context_group_id",
    "stage",
    "repo_id",
    "commit_sha",
    "repo_dir",
    "selected_chunk_ids",
    "selected_context_hash",
    "qa_file",
    "qa_row_group",
    "qa_start",
    "qa_count",
}
SNAPSHOT_MEMORY_GROUP_COLUMNS = {
    "qa_pack_starts",
    "qa_pack_counts",
    "qa_pack_token_counts",
}
REQUIRED_QA_COLUMNS = {
    "logical_example_id",
    "source_qa_id",
    "input_ids",
    "response_start",
    "response_end",
    "duplicate_multiplicity",
    "source_family",
    "task_category",
}
COST_BUCKETED_RANK_ASSIGNMENT = "cost_bucketed_ddp_rounds_v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(*parts: object) -> str:
    value = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def canonical_context_hash(
    repo_id: str,
    commit_sha: str,
    all_chunk_ids: list[str],
    selected_chunk_ids: list[str],
    payload_hashes: list[str],
) -> str:
    positions = {chunk_id: index for index, chunk_id in enumerate(all_chunk_ids)}
    if len(payload_hashes) != len(selected_chunk_ids):
        raise ValueError("Selected chunks and payload hashes differ in length")
    if any(chunk_id not in positions for chunk_id in selected_chunk_ids):
        raise ValueError("Selected chunk is absent from its exact snapshot")
    descriptors = [
        f"{positions[chunk_id] + 1}/{len(all_chunk_ids)}:{payload_hash}"
        for chunk_id, payload_hash in zip(selected_chunk_ids, payload_hashes)
    ]
    return _digest(
        "canonical-rendered-context-v1", repo_id, commit_sha, *descriptors
    )


@dataclass(frozen=True)
class GroupShard:
    path: str
    rows: int
    row_group_cumulative: tuple[int, ...]


@dataclass(frozen=True)
class GroupBlock:
    path: str
    row_group: int
    rows: int


class FrozenChunkStore:
    """Bounded metadata/payload/token cache with exact-freeze assertions."""

    def __init__(self, tokenizer: Any, max_context_tokens: int, cache_mb: int):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.tokens = ByteLRU(cache_mb * 1024 * 1024)
        self.metadata: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.payload_groups: OrderedDict[tuple[str, int], dict[str, str]] = (
            OrderedDict()
        )

    def _metadata(self, repo_dir: str) -> dict[str, Any]:
        cached = self.metadata.get(repo_dir)
        if cached is not None:
            self.metadata.move_to_end(repo_dir)
            return cached
        chunks_path = Path(repo_dir) / "chunks.parquet"
        parquet = pq.ParquetFile(chunks_path, memory_map=True)
        columns = set(parquet.schema_arrow.names)
        wanted = ["chunk_id"]
        if "payload_sha256" in columns:
            wanted.append("payload_sha256")
        chunk_rows = parquet.read(columns=wanted).to_pylist()
        locations: dict[str, tuple[int, int]] = {}
        offset = 0
        for group_index in range(parquet.metadata.num_row_groups):
            count = parquet.metadata.row_group(group_index).num_rows
            for row_index in range(count):
                locations[str(chunk_rows[offset + row_index]["chunk_id"])] = (
                    group_index,
                    row_index,
                )
            offset += count
        payload_hashes = {
            str(row["chunk_id"]): str(row["payload_sha256"])
            for row in chunk_rows
            if row.get("payload_sha256")
        }
        snapshots: dict[str, list[tuple[int, str]]] = {}
        for row in pq.read_table(
            Path(repo_dir) / "snapshots.parquet",
            columns=["commit_sha", "chunk_index", "chunk_id"],
            memory_map=True,
        ).to_pylist():
            snapshots.setdefault(str(row["commit_sha"]), []).append(
                (int(row["chunk_index"]), str(row["chunk_id"]))
            )
        value = {
            "parquet": parquet,
            "locations": locations,
            "payload_hashes": payload_hashes,
            "snapshots": {
                commit: list(dict.fromkeys(chunk for _index, chunk in sorted(rows)))
                for commit, rows in snapshots.items()
            },
        }
        self.metadata[repo_dir] = value
        while len(self.metadata) > 16:
            self.metadata.popitem(last=False)
        return value

    def _payload(self, repo_dir: str, chunk_id: str) -> str:
        metadata = self._metadata(repo_dir)
        group_index, _row_index = metadata["locations"][chunk_id]
        key = (repo_dir, group_index)
        payloads = self.payload_groups.get(key)
        if payloads is None:
            table = metadata["parquet"].read_row_group(
                group_index, columns=["chunk_id", "payload_text"]
            )
            payloads = {
                str(chunk): str(payload)
                for chunk, payload in zip(
                    table.column("chunk_id").to_pylist(),
                    table.column("payload_text").to_pylist(),
                )
            }
            self.payload_groups[key] = payloads
            while len(self.payload_groups) > 8:
                self.payload_groups.popitem(last=False)
        else:
            self.payload_groups.move_to_end(key)
        return payloads[chunk_id]

    def hydrate(self, group: dict[str, Any]) -> list[list[int]]:
        repo_dir = str(group["repo_dir"])
        repo_id = str(group["repo_id"])
        commit = str(group["commit_sha"])
        selected = [str(value) for value in group["selected_chunk_ids"]]
        metadata = self._metadata(repo_dir)
        all_chunks = metadata["snapshots"].get(commit)
        if not all_chunks:
            raise KeyError(f"Exact snapshot {commit} is missing from {repo_dir}")
        if not selected:
            raise ValueError("A production context group selected zero chunks")
        payload_hashes = []
        payloads = []
        for chunk_id in selected:
            payload = self._payload(repo_dir, chunk_id)
            payloads.append(payload)
            payload_hashes.append(
                metadata["payload_hashes"].get(chunk_id)
                or hashlib.sha256(payload.encode("utf-8")).hexdigest()
            )
        actual_hash = canonical_context_hash(
            repo_id, commit, all_chunks, selected, payload_hashes
        )
        if actual_hash != str(group["selected_context_hash"]):
            raise ValueError(
                f"Context hash mismatch for {group['context_group_id']}: "
                f"{actual_hash} != {group['selected_context_hash']}"
            )
        positions = {chunk_id: index for index, chunk_id in enumerate(all_chunks)}
        contexts: list[list[int]] = []
        for chunk_id, payload in zip(selected, payloads):
            cache_key = (f"{repo_dir}\0{commit}", chunk_id)
            ids = self.tokens.get(cache_key)
            if ids is None:
                text = render_repository_header(
                    repo_id, commit, positions[chunk_id], len(all_chunks)
                ) + payload
                encoded = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": text.strip()},
                    ],
                    tokenize=True,
                    add_special_tokens=False,
                    add_generation_prompt=True,
                    return_attention_mask=False,
                    padding=False,
                    truncation=False,
                )
                ids = list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)
                if len(ids) > self.max_context_tokens:
                    raise ValueError(
                        f"Canonical chunk {chunk_id} has {len(ids)} tokens; "
                        f"limit is {self.max_context_tokens}"
                    )
                self.tokens.put(cache_key, ids)
            contexts.append(ids)
        return contexts


class RepoQACEStreamingDataset(IterableDataset):

    def __init__(
        self,
        ready_path: str | Path,
        stage: str,
        split: str,
        ctx_tokenizer: Any,
        max_context_tokens: int,
        qa_token_budget: int = 8192,
        cache_mb: int = 256,
        seed: int = 42,
        validation_panel: str = "fast",
    ) -> None:
        super().__init__()
        self.ready_path = Path(ready_path).resolve()
        self.ready = json.loads(self.ready_path.read_text(encoding="utf-8"))
        self.ready_format = self.ready.get("format")
        if self.ready_format not in READY_FORMATS:
            raise ValueError(
                f"Expected one of {sorted(READY_FORMATS)}, "
                f"got {self.ready_format!r}"
            )
        self.snapshot_memory = self.ready_format == SNAPSHOT_MEMORY_READY_FORMAT
        self.required_group_columns = set(BASE_REQUIRED_GROUP_COLUMNS)
        self.required_qa_columns = set(REQUIRED_QA_COLUMNS)
        if self.snapshot_memory:
            self.required_group_columns.update(SNAPSHOT_MEMORY_GROUP_COLUMNS)
        raw_loss_weights = self.ready.get("qa_loss_weights", {})
        if raw_loss_weights:
            unknown = set(raw_loss_weights) - {"ast", "llm", "default"}
            if unknown:
                raise ValueError(f"Unknown QA loss-weight families: {sorted(unknown)}")
            self.qa_loss_weights = {
                str(family): float(weight)
                for family, weight in raw_loss_weights.items()
            }
            if any(weight <= 0 for weight in self.qa_loss_weights.values()):
                raise ValueError("QA loss weights must all be positive")
            self.required_qa_columns.add("qa_family")
        else:
            self.qa_loss_weights = {"default": 1.0}
        if self.ready.get("loss") != "answer_token_ce":
            raise ValueError("Production CE loader rejects teacher-target datasets")
        expected_tokenizer = self.ready.get("tokenizer")
        actual_tokenizer = getattr(ctx_tokenizer, "name_or_path", None)
        if expected_tokenizer and actual_tokenizer and expected_tokenizer != actual_tokenizer:
            raise ValueError(
                f"READY tokenizer {expected_tokenizer!r} != loaded {actual_tokenizer!r}"
            )
        expected_revision = self.ready.get("tokenizer_revision")
        actual_revision = getattr(ctx_tokenizer, "init_kwargs", {}).get("_commit_hash")
        if expected_revision and actual_revision and expected_revision != actual_revision:
            raise ValueError("READY tokenizer revision does not match loaded tokenizer")
        self.stage = stage
        self.split = split
        # This iterator assigns disjoint rows using RANK/WORLD_SIZE itself.
        # Trainer must not let Accelerate apply a second IterableDataset shard.
        self.rank_strided_assignment = True
        self.requires_exact_resume = split == "train"
        self.qa_token_budget = int(qa_token_budget)
        if self.qa_token_budget <= 0:
            raise ValueError("qa_token_budget must be positive")
        frozen_budget = int(self.ready.get("qa_token_budget", self.qa_token_budget))
        if frozen_budget != self.qa_token_budget:
            raise ValueError(
                f"READY QA token budget {frozen_budget} != configured "
                f"{self.qa_token_budget}"
            )
        self.seed = int(seed)
        self.pass_number = 0
        self.permutation_position = 0
        self.within_group_qa_offset = 0
        self.logical_qas_consumed = 0
        self.unique_logical_qas_consumed = 0
        self.supervised_tokens_consumed = 0
        self.answer_side_tokens_consumed = 0
        self.context_tokens_consumed = 0
        self.physical_packs_consumed = 0
        self.qas_per_adapter_histogram: Counter[int] = Counter()
        self.consumed_by_source_family: Counter[str] = Counter()
        self.consumed_by_task_category: Counter[str] = Counter()
        self.consumed_by_repo_bucket: Counter[str] = Counter()
        self.exhausted = False
        self.ready_sha256 = file_sha256(self.ready_path)
        key = f"{stage}/{split}"
        entry = self.ready.get("partitions", {}).get(key)
        if split == "val" and validation_panel:
            entry = (
                self.ready.get("validation_panels", {})
                .get(stage, {})
                .get(validation_panel)
            )
            key = f"{stage}/validation_panel/{validation_panel}"
        if not entry:
            raise KeyError(f"READY has no partition {key}")
        self.rank_assignment = str(
            entry.get("rank_assignment", "row-group-local-affine-permutation-stride-v2")
        )
        if self.rank_assignment not in {
            "row-group-local-affine-permutation-stride-v2",
            COST_BUCKETED_RANK_ASSIGNMENT,
        }:
            raise ValueError(f"Unsupported RepoQA rank assignment {self.rank_assignment!r}")
        raw_files = entry.get("group_manifests") or []
        if not raw_files:
            raise ValueError(f"READY partition {key} has no group manifests")
        self.shards = self._load_shards(raw_files)
        self.cumulative = []
        total = 0
        for shard in self.shards:
            total += shard.rows
            self.cumulative.append(total)
        self.total_groups = total
        self.padding_group_ordinal = int(entry.get("padding_group_ordinal", 0))
        if not 0 <= self.padding_group_ordinal < max(1, self.total_groups):
            raise ValueError("Invalid padding_group_ordinal in READY partition")
        self.blocks: list[GroupBlock] = []
        for shard in self.shards:
            previous = 0
            for row_group, cumulative in enumerate(shard.row_group_cumulative):
                self.blocks.append(
                    GroupBlock(shard.path, row_group, cumulative - previous)
                )
                previous = cumulative
        self._configure_permutation()
        self.expected_logical_qas = int(entry["logical_qas"])
        self.expected_unique_logical_qas = int(
            entry.get("physical_qa_rows", self.expected_logical_qas)
        )
        self.manifest_hashes = {
            shard.path: file_sha256(shard.path) for shard in self.shards
        }
        expected_hashes = entry.get("group_manifest_sha256", {})
        for path, actual in self.manifest_hashes.items():
            expected = expected_hashes.get(path) or expected_hashes.get(Path(path).name)
            if expected and expected != actual:
                raise ValueError(f"Group manifest hash mismatch for {path}")
        self.store = FrozenChunkStore(ctx_tokenizer, max_context_tokens, cache_mb)
        self._parquet_files: OrderedDict[str, pq.ParquetFile] = OrderedDict()
        self._group_row_groups: OrderedDict[
            tuple[str, int], list[dict[str, Any]]
        ] = OrderedDict()
        self._qa_row_groups: OrderedDict[tuple[str, int], Any] = OrderedDict()

    def _parquet(self, path: str | Path) -> pq.ParquetFile:
        key = str(path)
        parquet = self._parquet_files.get(key)
        if parquet is None:
            parquet = pq.ParquetFile(key, memory_map=True)
            self._parquet_files[key] = parquet
            while len(self._parquet_files) > 32:
                self._parquet_files.popitem(last=False)
        else:
            self._parquet_files.move_to_end(key)
        return parquet

    def _load_shards(self, patterns: list[str]) -> list[GroupShard]:
        paths: list[str] = []
        for pattern in patterns:
            candidate = Path(pattern)
            if not candidate.is_absolute():
                pattern = str(self.ready_path.parent / candidate)
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"No group manifests match {pattern!r}")
            paths.extend(matches)
        shards = []
        for path in paths:
            parquet = pq.ParquetFile(path, memory_map=True)
            missing = self.required_group_columns - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{path} is missing group columns {sorted(missing)}")
            cumulative = []
            total = 0
            for index in range(parquet.metadata.num_row_groups):
                total += parquet.metadata.row_group(index).num_rows
                cumulative.append(total)
            shards.append(
                GroupShard(str(Path(path).resolve()), total, tuple(cumulative))
            )
        return shards

    def __len__(self) -> int:
        # A safe upper bound; max_steps is only a ceiling and exhaustion is the
        # authoritative production stopping condition.
        return self.expected_logical_qas

    def _affine_parameters(self, n: int, *salt: object) -> tuple[int, int]:
        if n <= 1:
            return 0, 0
        raw = int(
            _digest(self.seed, self.pass_number, *salt, "affine-a")[:16], 16
        )
        multiplier = (raw % n) | 1
        while math.gcd(multiplier, n) != 1:
            multiplier = (multiplier + 2) % n or 1
        offset = int(
            _digest(self.seed, self.pass_number, *salt, "affine-b")[:16], 16
        ) % n
        return multiplier, offset

    def _configure_permutation(self) -> None:
        if self.rank_assignment == COST_BUCKETED_RANK_ASSIGNMENT:
            self._permuted_blocks = list(self.blocks)
            self._permuted_block_cumulative = []
            total = 0
            for block in self._permuted_blocks:
                total += block.rows
                self._permuted_block_cumulative.append(total)
            return
        multiplier, offset = self._affine_parameters(len(self.blocks), "blocks")
        if len(self.blocks) <= 1:
            order = list(range(len(self.blocks)))
        else:
            order = [
                (multiplier * position + offset) % len(self.blocks)
                for position in range(len(self.blocks))
            ]
        self._permuted_blocks = [self.blocks[index] for index in order]
        self._permuted_block_cumulative = []
        total = 0
        for block in self._permuted_blocks:
            total += block.rows
            self._permuted_block_cumulative.append(total)

    def _group(self, ordinal: int) -> dict[str, Any]:
        block_position = bisect.bisect_right(
            self._permuted_block_cumulative, ordinal
        )
        previous = (
            self._permuted_block_cumulative[block_position - 1]
            if block_position
            else 0
        )
        block = self._permuted_blocks[block_position]
        within_position = ordinal - previous
        if self.rank_assignment == COST_BUCKETED_RANK_ASSIGNMENT:
            within = within_position
        else:
            multiplier, offset = self._affine_parameters(
                block.rows, block.path, block.row_group
            )
            within = (
                0
                if block.rows <= 1
                else (multiplier * within_position + offset) % block.rows
            )
        key = (block.path, block.row_group)
        rows = self._group_row_groups.get(key)
        if rows is None:
            rows = self._parquet(block.path).read_row_group(
                block.row_group, columns=sorted(self.required_group_columns)
            ).to_pylist()
            self._group_row_groups[key] = rows
            while len(self._group_row_groups) > 4:
                self._group_row_groups.popitem(last=False)
        else:
            self._group_row_groups.move_to_end(key)
        return rows[within]

    def _qa_rows(self, group: dict[str, Any]) -> list[dict[str, Any]]:
        qa_path = Path(str(group["qa_file"]))
        if not qa_path.is_absolute():
            qa_path = self.ready_path.parent / qa_path
        parquet = self._parquet(qa_path)
        missing = self.required_qa_columns - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{qa_path} is missing QA columns {sorted(missing)}")
        row_group = int(group["qa_row_group"])
        cache_key = (str(qa_path), row_group)
        row_group_table = self._qa_row_groups.get(cache_key)
        if row_group_table is None:
            row_group_table = parquet.read_row_group(
                row_group, columns=sorted(self.required_qa_columns)
            )
            self._qa_row_groups[cache_key] = row_group_table
            while len(self._qa_row_groups) > 16:
                self._qa_row_groups.popitem(last=False)
        else:
            self._qa_row_groups.move_to_end(cache_key)
        table = row_group_table.slice(
            int(group["qa_start"]), int(group["qa_count"])
        )
        rows = table.to_pylist()
        if len(rows) != int(group["qa_count"]):
            raise ValueError(f"Truncated QA range for {group['context_group_id']}")
        return rows

    def _pack(
        self,
        group: dict[str, Any],
        contexts: list[list[int]],
        rows: list[dict[str, Any]],
        qa_offset: int,
    ) -> tuple[dict[str, Any], int]:
        selected: list[dict[str, Any]] = []
        tokens = 0
        for row in rows[qa_offset:]:
            ids = [int(value) for value in row["input_ids"]]
            if (
                not self.snapshot_memory
                and selected
                and tokens + len(ids) > self.qa_token_budget
            ):
                break
            selected.append(row)
            tokens += len(ids)
            if not self.snapshot_memory and tokens >= self.qa_token_budget:
                break
        if not selected:
            raise RuntimeError("Token-budget packer made no progress")
        input_ids: list[int] = []
        labels: list[int] = []
        position_ids: list[int] = []
        sequence_ids: list[int] = []
        boundaries: list[list[int]] = []
        logical_ids: list[str] = []
        source_ids: list[str] = []
        qa_weights: list[int] = []
        qa_loss_weights: list[float] = []
        source_family_counts: Counter[str] = Counter()
        task_category_counts: Counter[str] = Counter()
        for sequence, row in enumerate(selected):
            ids = [int(value) for value in row["input_ids"]]
            start = int(row["response_start"])
            end = int(row["response_end"])
            if not 0 < start < end <= len(ids):
                raise ValueError(
                    f"Invalid response span [{start}, {end}) for "
                    f"{row['logical_example_id']} with {len(ids)} tokens"
                )
            offset = len(input_ids)
            input_ids.extend(ids)
            labels.extend(
                token if start <= index < end else IGNORE_INDEX
                for index, token in enumerate(ids)
            )
            position_ids.extend(range(len(ids)))
            sequence_ids.extend([sequence] * len(ids))
            boundaries.append([offset, offset + len(ids), offset + start, offset + end])
            logical_ids.append(str(row["logical_example_id"]))
            source_ids.append(str(row["source_qa_id"]))
            multiplicity = int(row["duplicate_multiplicity"])
            if multiplicity <= 0:
                raise ValueError("duplicate_multiplicity must be positive")
            qa_weights.append(multiplicity)
            qa_family = str(row.get("qa_family") or "default")
            family_weight = self.qa_loss_weights.get(
                qa_family, self.qa_loss_weights.get("default")
            )
            if family_weight is None:
                raise ValueError(f"No loss weight for QA family {qa_family!r}")
            qa_loss_weights.append(multiplicity * family_weight)
            source_family_counts[str(row.get("source_family") or "unknown")] += multiplicity
            task_category_counts[str(row.get("task_category") or "unknown")] += multiplicity
        flat_context = [token for chunk in contexts for token in chunk]
        physical_pack_count = 1
        if self.snapshot_memory:
            starts = [int(value) for value in group["qa_pack_starts"]]
            counts = [int(value) for value in group["qa_pack_counts"]]
            token_counts = [
                int(value) for value in group["qa_pack_token_counts"]
            ]
            expected_starts = []
            offset = 0
            for count in counts:
                expected_starts.append(offset)
                offset += count
            if (
                starts != expected_starts
                or offset != len(rows)
                or len(token_counts) != len(counts)
                or any(value > self.qa_token_budget for value in token_counts)
            ):
                raise ValueError(
                    f"Invalid frozen QA packs for {group['context_group_id']}"
                )
            actual_token_counts = [
                sum(
                    len(rows[index]["input_ids"])
                    for index in range(start, start + count)
                )
                for start, count in zip(starts, counts)
            ]
            if actual_token_counts != token_counts:
                raise ValueError(
                    f"Frozen QA pack tokens changed for {group['context_group_id']}"
                )
            physical_pack_count = len(counts)
        return (
            {
                "ctx_ids": np.asarray(flat_context, dtype=np.int64),
                "ctx_position_ids": np.asarray(
                    [position for chunk in contexts for position in range(len(chunk))],
                    dtype=np.int64,
                ),
                "input_ids": np.asarray(input_ids, dtype=np.int64),
                "position_ids": np.asarray(position_ids, dtype=np.int64),
                "labels": np.asarray(labels, dtype=np.int64),
                "n_queries": [len(selected)],
                "logical_qa_count": [sum(qa_weights)],
                "qa_weights": np.asarray(qa_weights, dtype=np.int32),
                "logical_qa_loss_weight": [sum(qa_loss_weights)],
                "qa_loss_weights": np.asarray(qa_loss_weights, dtype=np.float32),
                "n_ctx_chunks": [len(contexts)],
                "packed_sequence_ids": np.asarray(sequence_ids, dtype=np.int32),
                "qa_boundaries": np.asarray(boundaries, dtype=np.int64),
                "logical_example_ids": logical_ids,
                "source_qa_ids": source_ids,
                "context_group_id": str(group["context_group_id"]),
                "repo_key": [
                    int(
                        hashlib.sha256(
                            str(group["repo_id"]).encode("utf-8")
                        ).hexdigest()[:15],
                        16,
                    )
                ],
                "qa_pack_starts": (
                    [int(value) for value in group["qa_pack_starts"]]
                    if self.snapshot_memory
                    else [0]
                ),
                "qa_pack_counts": (
                    [int(value) for value in group["qa_pack_counts"]]
                    if self.snapshot_memory
                    else [len(selected)]
                ),
                "physical_pack_count": physical_pack_count,
                "source_cursor": {
                    "permutation_position": self.permutation_position,
                    "within_group_qa_offset": qa_offset,
                },
                "audit_counts": {
                    "source_family": dict(source_family_counts),
                    "task_category": dict(task_category_counts),
                    "repo_bucket": hashlib.sha256(
                        str(group["repo_id"]).encode("utf-8")
                    ).hexdigest()[:2],
                    "supervised_tokens": sum(
                        (int(row["response_end"]) - int(row["response_start"]))
                        * weight
                        for row, weight in zip(selected, qa_weights)
                    ),
                    "answer_side_tokens": sum(
                        len(row["input_ids"]) * weight
                        for row, weight in zip(selected, qa_weights)
                    ),
                    "context_tokens": len(flat_context),
                },
            },
            qa_offset + len(selected),
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        if worker is not None:
            raise RuntimeError(
                "Exact production resume currently requires "
                "dataloader_num_workers=0; subprocess dataset cursors cannot be "
                "atomically included in a Trainer checkpoint"
            )
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        consumer = rank * workers + worker_id
        consumers = world_size * workers
        padded_total_groups = (
            ((self.total_groups + consumers - 1) // consumers) * consumers
            if self.total_groups
            else 0
        )
        track_state = self.split == "train"
        position = self.permutation_position if track_state else 0
        if position % consumers != consumer:
            position += (consumer - position) % consumers
        while position < padded_total_groups:
            is_padding = position >= self.total_groups
            group_position = self.padding_group_ordinal if is_padding else position
            group = self._group(group_position)
            if str(group["stage"]) != self.stage:
                raise ValueError(f"Group {group['context_group_id']} has wrong stage")
            contexts = self.store.hydrate(group)
            rows = self._qa_rows(group)
            qa_offset = (
                self.within_group_qa_offset
                if track_state and position == self.permutation_position
                else 0
            )
            while qa_offset < len(rows):
                item, next_offset = self._pack(group, contexts, rows, qa_offset)
                if next_offset != len(rows):
                    raise ValueError(
                        "Production group manifests must be pre-split at the "
                        "frozen QA token budget so every DDP rank emits exactly "
                        "one physical pack per permutation round"
                    )
                item["is_ddp_padding"] = is_padding
                if is_padding:
                    item["qa_weights"][:] = 0
                    item["qa_loss_weights"][:] = 0
                    item["logical_qa_count"] = [0]
                    item["logical_qa_loss_weight"] = [0.0]
                    item["audit_counts"] = {
                        "source_family": {},
                        "task_category": {},
                        "repo_bucket": "padding",
                        "supervised_tokens": 0,
                        "answer_side_tokens": 0,
                        "context_tokens": 0,
                    }
                if track_state and not is_padding:
                    audit = item["audit_counts"]
                    self.logical_qas_consumed += int(item["logical_qa_count"][0])
                    self.unique_logical_qas_consumed += int(item["n_queries"][0])
                    self.supervised_tokens_consumed += int(audit["supervised_tokens"])
                    self.answer_side_tokens_consumed += int(audit["answer_side_tokens"])
                    self.context_tokens_consumed += int(audit["context_tokens"])
                    self.physical_packs_consumed += int(
                        item["physical_pack_count"]
                    )
                    self.qas_per_adapter_histogram[int(item["n_queries"][0])] += 1
                    self.consumed_by_source_family.update(audit["source_family"])
                    self.consumed_by_task_category.update(audit["task_category"])
                    self.consumed_by_repo_bucket[audit["repo_bucket"]] += int(
                        item["logical_qa_count"][0]
                    )
                if track_state:
                    self.permutation_position = position + consumers
                    self.within_group_qa_offset = 0
                    # The cursor is authoritative. Mark exact exhaustion before
                    # yielding the final assigned group so a max_steps ceiling
                    # that lands exactly on the corpus boundary does not require
                    # one extra DataLoader read merely to discover StopIteration.
                    self.exhausted = (
                        self.permutation_position >= padded_total_groups
                    )
                yield item
                qa_offset = next_offset
            position += consumers
        if track_state:
            self.exhausted = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "doc_to_lora_repoqa_sampler_v1",
            "ready_path": str(self.ready_path),
            "ready_sha256": self.ready_sha256,
            "manifest_hashes": self.manifest_hashes,
            "stage": self.stage,
            "split": self.split,
            "permutation_seed": self.seed,
            "pass_number": self.pass_number,
            "permutation_position": self.permutation_position,
            "within_group_qa_offset": self.within_group_qa_offset,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "rank": int(os.environ.get("RANK", "0")),
            "rank_assignment": self.rank_assignment,
            "qa_token_budget": self.qa_token_budget,
            "total_groups": self.total_groups,
            "logical_qas_consumed": self.logical_qas_consumed,
            "unique_logical_qas_consumed": self.unique_logical_qas_consumed,
            "supervised_tokens_consumed": self.supervised_tokens_consumed,
            "answer_side_tokens_consumed": self.answer_side_tokens_consumed,
            "context_tokens_consumed": self.context_tokens_consumed,
            "physical_packs_consumed": self.physical_packs_consumed,
            "qas_per_adapter_histogram": dict(self.qas_per_adapter_histogram),
            "consumed_by_source_family": dict(self.consumed_by_source_family),
            "consumed_by_task_category": dict(self.consumed_by_task_category),
            "consumed_by_repo_bucket": dict(self.consumed_by_repo_bucket),
            "expected_logical_qas": self.expected_logical_qas,
            "expected_unique_logical_qas": self.expected_unique_logical_qas,
            "accumulated_ce_numerator": 0.0,
            "accumulated_l1_numerator": 0.0,
            "accumulated_logical_qa_denominator": 0,
            "checkpoint_boundary": "optimizer_step",
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        checks = {
            "format": "doc_to_lora_repoqa_sampler_v1",
            "ready_sha256": self.ready_sha256,
            "manifest_hashes": self.manifest_hashes,
            "stage": self.stage,
            "split": self.split,
            "permutation_seed": self.seed,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "rank": int(os.environ.get("RANK", "0")),
            "rank_assignment": self.rank_assignment,
            "qa_token_budget": self.qa_token_budget,
            "total_groups": self.total_groups,
        }
        mismatches = {
            key: (state.get(key), expected)
            for key, expected in checks.items()
            if state.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"RepoQA sampler resume mismatch: {mismatches}")
        self.pass_number = int(state["pass_number"])
        self._configure_permutation()
        self.permutation_position = int(state["permutation_position"])
        self.within_group_qa_offset = int(state["within_group_qa_offset"])
        self.logical_qas_consumed = int(state["logical_qas_consumed"])
        self.unique_logical_qas_consumed = int(
            state.get("unique_logical_qas_consumed", self.logical_qas_consumed)
        )
        self.supervised_tokens_consumed = int(
            state.get("supervised_tokens_consumed", 0)
        )
        self.answer_side_tokens_consumed = int(
            state.get("answer_side_tokens_consumed", 0)
        )
        self.context_tokens_consumed = int(state.get("context_tokens_consumed", 0))
        self.physical_packs_consumed = int(state.get("physical_packs_consumed", 0))
        self.qas_per_adapter_histogram = Counter(
            {int(key): value for key, value in state.get("qas_per_adapter_histogram", {}).items()}
        )
        self.consumed_by_source_family = Counter(
            state.get("consumed_by_source_family", {})
        )
        self.consumed_by_task_category = Counter(
            state.get("consumed_by_task_category", {})
        )
        self.consumed_by_repo_bucket = Counter(
            state.get("consumed_by_repo_bucket", {})
        )
        self.exhausted = self.permutation_position >= self.total_groups


def load_repoqa_ce_streaming_dataset(
    ready_path: str,
    stage: str,
    split: str,
    ctx_tokenizer: Any,
    max_context_tokens: int,
    qa_token_budget: int,
    cache_mb: int,
    seed: int,
    validation_panel: str = "fast",
) -> RepoQACEStreamingDataset:
    return RepoQACEStreamingDataset(
        ready_path,
        stage,
        split,
        ctx_tokenizer,
        max_context_tokens,
        qa_token_budget,
        cache_mb,
        seed,
        validation_panel,
    )
