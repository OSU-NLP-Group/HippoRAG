#!/usr/bin/env python3
"""Dataset helpers for static Code2LoRA prompt-target training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class StaticPromptTargetRow:
    repo_id: str
    commit_sha: str
    instance_id: str
    prompt: str
    target_text: str
    repo_state_embedding: np.ndarray
    metadata: Dict[str, Any]

    @property
    def target_patch(self) -> str:
        """Backward-compatible alias for older issue-fix trainer code."""
        return self.target_text


IssueFixRow = StaticPromptTargetRow


def _embedding_to_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"repo_state_embedding must be 1-D, got shape {arr.shape}")
    return arr


def load_static_prompt_target_rows(
    path: Path,
    *,
    limit: int = 0,
    target_column: str = "target_text",
) -> List[StaticPromptTargetRow]:
    """Load a static prompt-target parquet/jsonl table.

    Expected columns/keys:
    ``repo_id``, ``commit_sha`` or ``base_commit``, ``instance_id``, ``prompt``,
    a target column, and ``repo_state_embedding``. If ``target_column`` is
    missing and ``target_patch`` exists, the legacy patch column is used.
    Extra fields are preserved under ``metadata``.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        rows: List[StaticPromptTargetRow] = []
        with path.open() as fh:
            for line in fh:
                if limit and len(rows) >= limit:
                    break
                item = json.loads(line)
                rows.append(_row_from_mapping(item, target_column=target_column))
        return rows

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    rows = []
    names = set(table.column_names)
    required = {"repo_id", "instance_id", "prompt", "repo_state_embedding"}
    missing = sorted(required - names)
    if "commit_sha" not in names and "base_commit" not in names:
        missing.append("commit_sha_or_base_commit")
    if target_column not in names and "target_patch" not in names:
        missing.append(f"{target_column}_or_target_patch")
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    n = table.num_rows if not limit else min(limit, table.num_rows)
    for i in range(n):
        item = {name: columns[name][i] for name in table.column_names}
        rows.append(_row_from_mapping(item, target_column=target_column))
    return rows


def load_issuefix_rows(path: Path, *, limit: int = 0, target_column: str = "target_text") -> List[IssueFixRow]:
    return load_static_prompt_target_rows(path, limit=limit, target_column=target_column)


def _row_from_mapping(item: Dict[str, Any], *, target_column: str = "target_text") -> StaticPromptTargetRow:
    target_key = target_column if target_column in item else "target_patch"
    excluded = {
        "repo_id",
        "commit_sha",
        "base_commit",
        "instance_id",
        "prompt",
        target_key,
        "repo_state_embedding",
    }
    metadata = {
        key: value
        for key, value in item.items()
        if key not in excluded
    }
    commit = item.get("commit_sha") or item.get("base_commit") or ""
    return StaticPromptTargetRow(
        repo_id=str(item["repo_id"]),
        commit_sha=str(commit),
        instance_id=str(item["instance_id"]),
        prompt=str(item["prompt"]),
        target_text=str(item[target_key]),
        repo_state_embedding=_embedding_to_array(item["repo_state_embedding"]),
        metadata=metadata,
    )


class StaticPromptTargetDataset(Dataset):
    def __init__(self, rows: Iterable[StaticPromptTargetRow], *, seed: int = 3407):
        self.rows = list(rows)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> StaticPromptTargetRow:
        return self.rows[idx]


StaticIssueFixDataset = StaticPromptTargetDataset


def tokenize_prompt_target_batch(
    tokenizer,
    prompts: List[str],
    targets: List[str],
    *,
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    """Tokenize ``prompt || target`` and mask prompt labels.

    Examples are kept only when the full prompt and full target fit. Dataset
    builders should do task-specific truncation before rows reach the trainer.
    """
    eos = tokenizer.eos_token or ""
    input_ids_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    attn_list: List[torch.Tensor] = []
    for prompt, target in zip(prompts, targets):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target + eos, add_special_tokens=False)["input_ids"]
        if not target_ids:
            continue
        if len(prompt_ids) + len(target_ids) > max_seq_len:
            continue
        ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)
        input_ids_list.append(torch.tensor(ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
        attn_list.append(torch.ones(len(ids), dtype=torch.long))
    if not input_ids_list:
        return {}
    length = max(x.size(0) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id or 0

    def _left_pad(x: torch.Tensor, value: int) -> torch.Tensor:
        return F.pad(x, (length - x.size(0), 0), value=value)

    return {
        "input_ids": torch.stack([_left_pad(x, pad_id) for x in input_ids_list]),
        "labels": torch.stack([_left_pad(x, -100) for x in labels_list]),
        "attention_mask": torch.stack([_left_pad(x, 0) for x in attn_list]),
    }


def tokenize_issuefix_batch(tokenizer, prompts: List[str], targets: List[str], *, max_seq_len: int) -> Dict[str, torch.Tensor]:
    return tokenize_prompt_target_batch(tokenizer, prompts, targets, max_seq_len=max_seq_len)
