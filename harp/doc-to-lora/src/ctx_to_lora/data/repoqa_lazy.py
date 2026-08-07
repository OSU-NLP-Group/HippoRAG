"""Lazy frozen-repository dataset for whole-repository Doc-to-LoRA training."""

from __future__ import annotations

import glob
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from ctx_to_lora.data.definitions import IGNORE_INDEX


def render_repository_header(repo_id: str, commit: str, index: int, total: int) -> str:
    return (
        f"<<<REPOSITORY name={json.dumps(repo_id, ensure_ascii=False)} "
        f"commit={commit} chunk={index + 1}/{total}>>>\n"
    )


def _input_ids(tokenized: Any) -> list[int]:
    values = tokenized["input_ids"] if hasattr(tokenized, "keys") else tokenized
    return list(values)


def _unique_in_order(values: list[str]) -> list[str]:
    # Chunk IDs are content hashes. Repeated IDs within one snapshot are
    # byte-identical payloads (usually repeated pieces of a huge one-line JSON
    # file), so encoding them more than once adds no repository information.
    return list(dict.fromkeys(values))


class ByteLRU:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.values: OrderedDict[tuple[str, str], tuple[list[int], int]] = OrderedDict()

    def get(self, key: tuple[str, str]) -> list[int] | None:
        value = self.values.get(key)
        if value is None:
            return None
        self.values.move_to_end(key)
        return value[0]

    def put(self, key: tuple[str, str], value: list[int]) -> None:
        size = len(value) * 4
        if size > self.max_bytes:
            return
        old = self.values.pop(key, None)
        if old is not None:
            self.current_bytes -= old[1]
        while self.values and self.current_bytes + size > self.max_bytes:
            _old_key, (_old_value, old_size) = self.values.popitem(last=False)
            self.current_bytes -= old_size
        self.values[key] = (value, size)
        self.current_bytes += size


class FrozenRepoQADataset(Dataset):
    """Hydrate canonical chunks only when a sampled QA row is requested."""

    def __init__(
        self,
        rows: datasets.Dataset,
        tokenizer: Any,
        ctx_tokenizer: Any,
        repo_merge_method: str,
        max_qas_len: int,
        max_context_tokens: int,
        chunk_cache_mb: int = 256,
        max_qas_per_sample: int = 1,
        group_indices: list[tuple[int, ...]] | None = None,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.repo_merge_method = repo_merge_method
        self.max_qas_len = max_qas_len
        self.max_context_tokens = max_context_tokens
        self.max_qas_per_sample = max_qas_per_sample
        self.chunk_cache = ByteLRU(chunk_cache_mb * 1024 * 1024)
        self.payload_row_groups: OrderedDict[
            tuple[str, int], tuple[dict[str, str], int]
        ] = OrderedDict()
        self.payload_row_groups_bytes = 0
        self.payload_row_groups_max_bytes = chunk_cache_mb * 1024 * 1024
        self.repo_metadata: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.group_indices = (
            group_indices
            if group_indices is not None
            else self._build_qa_groups(max_qas_per_sample)
        )

    def _context_key(self, row: dict[str, Any]) -> tuple[Any, ...]:
        if self.repo_merge_method == "bm25_topk_ties":
            selected = tuple(str(value) for value in (row.get("bm25_chunk_ids") or []))
        elif bool(row.get("use_all_chunks")):
            selected = ("__all_snapshot_chunks__",)
        else:
            selected = tuple(str(value) for value in (row.get("chunk_ids") or []))
        return (
            str(row["repo_dir"]),
            str(row["commit_sha"]),
            selected,
        )

    def _build_qa_groups(self, maximum: int) -> list[tuple[int, ...]]:
        if maximum <= 1:
            return [(index,) for index in range(len(self.rows))]
        pending: OrderedDict[tuple[Any, ...], list[int]] = OrderedDict()
        groups: list[tuple[int, ...]] = []
        columns = [
            "repo_dir",
            "commit_sha",
            "chunk_ids",
            "use_all_chunks",
        ]
        if self.repo_merge_method == "bm25_topk_ties":
            columns.append("bm25_chunk_ids")
        for batch in self.rows.iter(batch_size=16_384):
            batch_size = len(batch[columns[0]])
            base = sum(len(group) for group in groups) + sum(len(group) for group in pending.values())
            # ``datasets.Dataset.iter`` does not expose source indices. The
            # number of rows already assigned is exactly the next source index
            # because this pass is sequential and every row is assigned once.
            for offset in range(batch_size):
                row = {column: batch[column][offset] for column in columns}
                key = self._context_key(row)
                group = pending.setdefault(key, [])
                group.append(base + offset)
                if len(group) == maximum:
                    groups.append(tuple(group))
                    del pending[key]
        groups.extend(tuple(group) for group in pending.values())
        return groups

    def __len__(self) -> int:
        return len(self.group_indices)

    @property
    def column_names(self) -> list[str]:
        return list(self.rows.column_names)

    def _with_group_indices(
        self, group_indices: list[tuple[int, ...]]
    ) -> "FrozenRepoQADataset":
        return FrozenRepoQADataset(
            self.rows,
            self.tokenizer,
            self.ctx_tokenizer,
            self.repo_merge_method,
            self.max_qas_len,
            self.max_context_tokens,
            max(1, self.chunk_cache.max_bytes // (1024 * 1024)),
            self.max_qas_per_sample,
            group_indices,
        )

    def select(self, indices: Any) -> "FrozenRepoQADataset":
        return self._with_group_indices([self.group_indices[int(index)] for index in indices])

    def take(self, count: int) -> "FrozenRepoQADataset":
        return self.select(range(min(count, len(self))))

    def skip(self, count: int) -> "FrozenRepoQADataset":
        return self.select(range(min(count, len(self)), len(self)))

    def _metadata(self, repo_dir: str) -> dict[str, Any]:
        cached = self.repo_metadata.get(repo_dir)
        if cached is not None:
            self.repo_metadata.move_to_end(repo_dir)
            return cached
        chunks_path = Path(repo_dir) / "chunks.parquet"
        chunks_file = pq.ParquetFile(chunks_path, memory_map=True)
        chunk_ids = chunks_file.read(columns=["chunk_id"]).column("chunk_id").to_pylist()
        row_groups: dict[str, tuple[int, int]] = {}
        offset = 0
        for group_index in range(chunks_file.metadata.num_row_groups):
            group_rows = chunks_file.metadata.row_group(group_index).num_rows
            for row_index in range(group_rows):
                row_groups[str(chunk_ids[offset + row_index])] = (
                    group_index,
                    row_index,
                )
            offset += group_rows

        snapshots = pq.read_table(
            Path(repo_dir) / "snapshots.parquet",
            columns=["commit_sha", "chunk_index", "chunk_id"],
            memory_map=True,
        ).to_pylist()
        snapshot_chunks: dict[str, list[tuple[int, str]]] = {}
        for row in snapshots:
            snapshot_chunks.setdefault(str(row["commit_sha"]), []).append(
                (int(row["chunk_index"]), str(row["chunk_id"]))
            )
        metadata = {
            "chunk_locations": row_groups,
            "snapshot_chunks": {
                commit: _unique_in_order(
                    [chunk_id for _index, chunk_id in sorted(values)]
                )
                for commit, values in snapshot_chunks.items()
            },
        }
        self.repo_metadata[repo_dir] = metadata
        while len(self.repo_metadata) > 16:
            self.repo_metadata.popitem(last=False)
        return metadata

    def _payload(self, repo_dir: str, chunk_id: str) -> str:
        metadata = self._metadata(repo_dir)
        try:
            group_index, row_index = metadata["chunk_locations"][chunk_id]
        except KeyError as exc:
            raise KeyError(f"Missing chunk {chunk_id} in {repo_dir}/chunks.parquet") from exc
        cache_key = (repo_dir, group_index)
        cached = self.payload_row_groups.get(cache_key)
        if cached is None:
            table = pq.ParquetFile(
                Path(repo_dir) / "chunks.parquet", memory_map=True
            ).read_row_group(group_index, columns=["chunk_id", "payload_text"])
            payloads = {
                str(key): str(value)
                for key, value in zip(
                    table.column("chunk_id").to_pylist(),
                    table.column("payload_text").to_pylist(),
                )
            }
            size = sum(len(value.encode("utf-8")) for value in payloads.values())
            while (
                self.payload_row_groups
                and self.payload_row_groups_bytes + size
                > self.payload_row_groups_max_bytes
            ):
                _key, (_payloads, old_size) = self.payload_row_groups.popitem(
                    last=False
                )
                self.payload_row_groups_bytes -= old_size
            if size <= self.payload_row_groups_max_bytes:
                self.payload_row_groups[cache_key] = (payloads, size)
                self.payload_row_groups_bytes += size
        else:
            self.payload_row_groups.move_to_end(cache_key)
            payloads = cached[0]
        if chunk_id not in payloads:
            raise RuntimeError(
                f"Chunk index mismatch for {chunk_id} in row group {group_index}"
            )
        return payloads[chunk_id]

    def _snapshot_chunk_ids(self, repo_dir: str, commit_sha: str) -> list[str]:
        snapshots = self._metadata(repo_dir)["snapshot_chunks"]
        if commit_sha not in snapshots:
            raise KeyError(f"Missing snapshot {commit_sha} in {repo_dir}/snapshots.parquet")
        return snapshots[commit_sha]

    def _tokenize_chunk(
        self,
        repo_id: str,
        commit_sha: str,
        chunk_id: str,
        chunk_index: int,
        num_chunks: int,
        repo_dir: str,
    ) -> list[int]:
        cache_key = (commit_sha, chunk_id)
        cached = self.chunk_cache.get(cache_key)
        if cached is not None:
            return cached
        text = render_repository_header(repo_id, commit_sha, chunk_index, num_chunks)
        text += self._payload(repo_dir, chunk_id)
        tokenized = self.ctx_tokenizer.apply_chat_template(
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
            return_dict=True,
        )
        ids = _input_ids(tokenized)
        if len(ids) > self.max_context_tokens:
            raise ValueError(
                f"Canonical chunk {chunk_id} has {len(ids)} tokens; "
                f"limit is {self.max_context_tokens}"
            )
        self.chunk_cache.put(cache_key, ids)
        return ids

    def _tokenize_qa(self, question: str, answer: str) -> tuple[list[int], list[int]]:
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ]
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            add_generation_prompt=False,
            return_attention_mask=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        ids = _input_ids(tokenized)
        masks = list(tokenized.get("assistant_masks") or [])
        if masks and any(masks):
            labels = [token if mask else IGNORE_INDEX for token, mask in zip(ids, masks)]
        else:
            prompt_ids = _input_ids(
                self.tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=True,
                    add_special_tokens=False,
                    add_generation_prompt=True,
                    return_attention_mask=False,
                    padding=False,
                    truncation=False,
                )
            )
            start = min(len(prompt_ids), len(ids))
            labels = [IGNORE_INDEX] * start + ids[start:]
        if not any(label != IGNORE_INDEX for label in labels):
            raise ValueError("QA example has no supervised assistant tokens")
        if self.max_qas_len > 0 and len(ids) > self.max_qas_len:
            raise ValueError(
                f"QA example has {len(ids)} tokens; max_qas_len={self.max_qas_len}"
            )
        return ids, labels

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_indices = self.group_indices[int(index)]
        qa_rows = [self.rows[source_index] for source_index in source_indices]
        row = qa_rows[0]
        repo_id = str(row["repo_id"])
        commit_sha = str(row["commit_sha"])
        repo_dir = str(row["repo_dir"])
        all_chunk_ids = self._snapshot_chunk_ids(repo_dir, commit_sha)
        chunk_ids = row.get("chunk_ids") or []
        if bool(row.get("use_all_chunks")):
            chunk_ids = all_chunk_ids
        if self.repo_merge_method == "bm25_topk_ties":
            chunk_ids = row.get("bm25_chunk_ids")
            if not chunk_ids:
                raise ValueError("BM25 row is missing preselected bm25_chunk_ids")
        chunk_ids = [str(value) for value in chunk_ids]
        if not chunk_ids:
            raise ValueError("RepoQA row did not select any repository chunks")
        chunk_positions = {chunk_id: i for i, chunk_id in enumerate(all_chunk_ids)}
        if any(chunk_id not in chunk_positions for chunk_id in chunk_ids):
            raise ValueError("Selected chunk IDs are not a subset of all_chunk_ids")
        contexts = [
            self._tokenize_chunk(
                repo_id,
                commit_sha,
                chunk_id,
                chunk_positions[chunk_id],
                len(all_chunk_ids),
                repo_dir,
            )
            for chunk_id in chunk_ids
        ]
        tokenized_qas = [
            self._tokenize_qa(str(qa_row["question"]), str(qa_row["answer"]))
            for qa_row in qa_rows
        ]
        input_ids = [token for ids, _labels in tokenized_qas for token in ids]
        labels = [label for _ids, qa_labels in tokenized_qas for label in qa_labels]
        position_ids = [
            position
            for ids, _labels in tokenized_qas
            for position in range(len(ids))
        ]
        flat_context = np.asarray([token for chunk in contexts for token in chunk], dtype=np.int64)
        context_positions = np.asarray(
            [position for chunk in contexts for position in range(len(chunk))],
            dtype=np.int64,
        )
        return {
            "ctx_ids": flat_context,
            "ctx_position_ids": context_positions,
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "position_ids": np.asarray(position_ids, dtype=np.int64),
            "labels": np.asarray(labels, dtype=np.int64),
            "n_queries": [len(qa_rows)],
            "n_ctx_chunks": [len(contexts)],
        }


def load_frozen_repoqa_dataset(
    patterns: list[str],
    tokenizer: Any,
    ctx_tokenizer: Any,
    repo_merge_method: str,
    max_qas_len: int,
    max_context_tokens: int,
    chunk_cache_mb: int,
    max_qas_per_sample: int = 1,
    *,
    split: str = "train",
    max_repo_chunks: int = 0,
    require_bm25_full_evidence: bool = False,
) -> FrozenRepoQADataset:
    files: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No lazy RepoQA index files match {pattern!r}")
        files.extend(matches)
    rows = datasets.load_dataset("parquet", data_files=files, split="train")
    required = {
        "repo_id",
        "commit_sha",
        "repo_dir",
        "question",
        "answer",
        "chunk_ids",
        "use_all_chunks",
    }
    if max_repo_chunks > 0:
        required.add("num_repo_chunks")
    if require_bm25_full_evidence:
        required.add("bm25_evidence_recall")
    missing = required - set(rows.column_names)
    if missing:
        raise ValueError(f"Lazy RepoQA index is missing columns: {sorted(missing)}")
    original_rows = len(rows)
    if max_repo_chunks > 0:
        rows = rows.filter(
            lambda row: int(row["num_repo_chunks"]) <= max_repo_chunks,
            desc=f"Filtering repositories to <= {max_repo_chunks} chunks",
        )
    if require_bm25_full_evidence:
        if split != "train":
            raise ValueError(
                "BM25 full-evidence filtering is training-only; evaluation "
                "must retain retrieval misses"
            )
        if repo_merge_method != "bm25_topk_ties":
            raise ValueError(
                "repoqa_require_bm25_full_evidence is valid only for "
                "repo_merge_method=bm25_topk_ties"
            )
        rows = rows.filter(
            lambda row: float(row["bm25_evidence_recall"]) == 1.0,
            desc="Filtering BM25 training retrieval misses",
        )
    if not len(rows):
        raise ValueError("Lazy RepoQA filters removed every row")
    if len(rows) != original_rows:
        logging.info(
            "Lazy RepoQA filters retained %d/%d rows",
            len(rows),
            original_rows,
        )
    return FrozenRepoQADataset(
        rows,
        tokenizer,
        ctx_tokenizer,
        repo_merge_method,
        max_qas_len,
        max_context_tokens,
        chunk_cache_mb,
        max_qas_per_sample,
    )
