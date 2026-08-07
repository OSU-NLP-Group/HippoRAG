"""Tokenization helpers for mixed raw-completion and chat QA objectives."""

from __future__ import annotations

import hashlib

from typing import Dict, List

import torch


def score_cache_key(row) -> str:
    """Hash the exact immutable inputs used by context/no-context scoring."""
    fields = ("repo_id", "commit_sha", "task_type", "sample_type", "file_path",
              "student_prompt", "teacher_prompt", "target_text", "target_terminator")
    payload = "\0".join(str(row.get(field, "")) for field in fields)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
import torch.nn.functional as F


def tokenize_prompt_target_batch(
    tokenizer,
    prompts: List[str],
    targets: List[str],
    terminators: List[str],
    *,
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    """Mask prompts and supervise ``target + task-specific terminator``."""
    input_ids_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    attn_list: List[torch.Tensor] = []
    for prompt, target, terminator in zip(prompts, targets, terminators):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target + terminator, add_special_tokens=False)["input_ids"]
        if not target_ids or len(prompt_ids) + len(target_ids) > max_seq_len:
            continue
        ids = prompt_ids + target_ids
        input_ids_list.append(torch.tensor(ids, dtype=torch.long))
        labels_list.append(torch.tensor([-100] * len(prompt_ids) + target_ids, dtype=torch.long))
        attn_list.append(torch.ones(len(ids), dtype=torch.long))
    if not input_ids_list:
        return {}
    length = max(item.size(0) for item in input_ids_list)
    pad_id = tokenizer.pad_token_id or 0

    def left_pad(item: torch.Tensor, value: int) -> torch.Tensor:
        return F.pad(item, (length - item.size(0), 0), value=value)

    return {
        "input_ids": torch.stack([left_pad(item, pad_id) for item in input_ids_list]),
        "labels": torch.stack([left_pad(item, -100) for item in labels_list]),
        "attention_mask": torch.stack([left_pad(item, 0) for item in attn_list]),
    }
