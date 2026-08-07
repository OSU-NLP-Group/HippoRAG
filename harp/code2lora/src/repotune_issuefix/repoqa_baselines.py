"""Shared frozen RepoQA data utilities for non-Doc-to-LoRA baselines.

The production Doc-to-LoRA dataset is organized around repository context
groups.  Conventional SFT and static Code2LoRA need the exact same logical QA
rows, but must not hydrate or tokenize the 64K source chunks.  This module
provides small, resumable readers over the immutable Parquet manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


READY_FORMAT = "doc_to_lora_repoqa_snapshot_memory_v1"
IGNORE_INDEX = -100
QA_COLUMNS = (
    "logical_example_id",
    "repo_id",
    "commit_sha",
    "stage",
    "input_ids",
    "response_start",
    "response_end",
    "qa_family",
    "task_category",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_u64(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_ready(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    ready = json.loads(path.read_text())
    if ready.get("format") != READY_FORMAT:
        raise ValueError(f"Unexpected RepoQA READY format in {path}")
    if int(ready.get("train_logical_qas", 0)) != 10_000_000:
        raise ValueError("The baseline contract requires the frozen 10M corpus")
    if ready.get("qa_loss_weights") != {"ast": 1.0, "llm": 1.8}:
        raise ValueError("Unexpected frozen QA family weights")
    return ready


def group_rows(ready: dict[str, Any], stage: str, split: str) -> list[dict[str, Any]]:
    entry = ready["partitions"][f"{stage}/{split}"]
    rows: list[dict[str, Any]] = []
    for path in entry["group_manifests"]:
        rows.extend(pq.read_table(path, memory_map=True).to_pylist())
    expected = int(entry["context_groups"])
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} {stage}/{split} groups, found {len(rows)}")
    return rows


def read_group_qas(group: dict[str, Any], columns: Sequence[str] = QA_COLUMNS) -> list[dict[str, Any]]:
    parquet = pq.ParquetFile(str(group["qa_file"]), memory_map=True)
    table = parquet.read_row_group(int(group["qa_row_group"]), columns=list(columns))
    start = int(group["qa_start"])
    count = int(group["qa_count"])
    rows = table.slice(start, count).to_pylist()
    if len(rows) != count:
        raise ValueError(f"Short QA read for {group['context_group_id']}")
    expected_repo = str(group["repo_id"])
    expected_commit = str(group["commit_sha"])
    for row in rows:
        if str(row.get("repo_id")) != expected_repo or str(row.get("commit_sha")) != expected_commit:
            raise ValueError(f"QA snapshot mismatch for {group['context_group_id']}")
    return rows


def qa_loss_weight(row: dict[str, Any]) -> float:
    family = str(row.get("qa_family") or "")
    if family == "ast":
        return 1.0
    if family == "llm":
        return 1.8
    raise ValueError(f"Unsupported QA family {family!r}")


def length_bucket(length: int) -> int:
    if length <= 0:
        raise ValueError("QA token length must be positive")
    return int(math.ceil(math.log2(max(16, length))))


def batch_rows(rows: Iterable[dict[str, Any]], token_budget: int) -> Iterator[list[dict[str, Any]]]:
    """Create deterministic padded batches from length-sorted rows."""

    batch: list[dict[str, Any]] = []
    max_length = 0
    for row in rows:
        length = len(row["input_ids"])
        candidate_max = max(max_length, length)
        if batch and candidate_max * (len(batch) + 1) > token_budget:
            yield batch
            batch = []
            max_length = 0
        batch.append(row)
        max_length = max(max_length, length)
    if batch:
        yield batch


@dataclass
class QABatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_start: torch.Tensor
    response_end: torch.Tensor
    loss_weights: torch.Tensor
    logical_example_ids: list[str]

    @property
    def qa_count(self) -> int:
        return len(self.logical_example_ids)


def collate_qa_rows(rows: Sequence[dict[str, Any]], pad_token_id: int, device: torch.device) -> QABatch:
    if not rows:
        raise ValueError("Cannot collate an empty QA batch")
    max_length = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), max_length), int(pad_token_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    starts, ends, weights, ids = [], [], [], []
    for index, row in enumerate(rows):
        values = torch.as_tensor(row["input_ids"], dtype=torch.long, device=device)
        length = values.numel()
        start = int(row["response_start"])
        end = int(row["response_end"])
        if not 0 < start < end <= length:
            raise ValueError(f"Invalid response span for {row['logical_example_id']}")
        input_ids[index, :length] = values
        attention_mask[index, :length] = 1
        starts.append(start)
        ends.append(end)
        weights.append(qa_loss_weight(row))
        ids.append(str(row["logical_example_id"]))
    return QABatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_start=torch.tensor(starts, dtype=torch.long, device=device),
        response_end=torch.tensor(ends, dtype=torch.long, device=device),
        loss_weights=torch.tensor(weights, dtype=torch.float32, device=device),
        logical_example_ids=ids,
    )


def unwrap_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    current = model
    if hasattr(current, "get_base_model"):
        current = current.get_base_model()
    return current


def weighted_answer_ce(
    model: torch.nn.Module,
    batch: QABatch,
    *,
    ce_chunk_tokens: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted per-QA answer CE without prompt-position logits.

    ``ce_chunk_tokens`` bounds the number of answer positions projected over
    the vocabulary at once.  During training, each projection/CE chunk is
    activation-checkpointed so the FP32 vocabulary logits are recomputed
    during backward instead of all being retained simultaneously.  This is
    algebraically identical to the unchunked loss; it only changes the peak
    memory schedule.
    """

    if ce_chunk_tokens < 0:
        raise ValueError("ce_chunk_tokens must be non-negative")

    outer = unwrap_causal_lm(model)
    text_model = outer.model.language_model
    outputs = text_model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
    )
    hidden = outputs.last_hidden_state
    batch_indices: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    qa_indices: list[torch.Tensor] = []
    for row in range(batch.qa_count):
        start = int(batch.response_start[row].item())
        end = int(batch.response_end[row].item())
        count = end - start
        batch_indices.append(torch.full((count,), row, dtype=torch.long, device=hidden.device))
        positions.append(torch.arange(start - 1, end - 1, device=hidden.device))
        targets.append(batch.input_ids[row, start:end])
        qa_indices.append(torch.full((count,), row, dtype=torch.long, device=hidden.device))
    flat_batch = torch.cat(batch_indices)
    flat_positions = torch.cat(positions)
    flat_targets = torch.cat(targets)
    flat_qa = torch.cat(qa_indices)
    selected_hidden = hidden[flat_batch, flat_positions]
    text_config = outer.config.get_text_config()
    softcap = getattr(text_config, "final_logit_softcapping", None)

    def project_and_ce(
        chunk_hidden: torch.Tensor, chunk_targets: torch.Tensor
    ) -> torch.Tensor:
        logits = outer.lm_head(chunk_hidden)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return F.cross_entropy(logits.float(), chunk_targets, reduction="none")

    chunk_size = ce_chunk_tokens or flat_targets.numel()
    token_loss_chunks = []
    for start in range(0, flat_targets.numel(), chunk_size):
        end = min(start + chunk_size, flat_targets.numel())
        chunk_hidden = selected_hidden[start:end]
        chunk_targets = flat_targets[start:end]
        if torch.is_grad_enabled() and chunk_hidden.requires_grad and ce_chunk_tokens:
            chunk_losses = checkpoint(
                project_and_ce,
                chunk_hidden,
                chunk_targets,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            chunk_losses = project_and_ce(chunk_hidden, chunk_targets)
        token_loss_chunks.append(chunk_losses)
    token_losses = torch.cat(token_loss_chunks)
    qa_sums = token_losses.new_zeros(batch.qa_count)
    qa_counts = token_losses.new_zeros(batch.qa_count)
    qa_sums.scatter_add_(0, flat_qa, token_losses)
    qa_counts.scatter_add_(0, flat_qa, torch.ones_like(token_losses))
    qa_means = qa_sums / qa_counts.clamp_min(1)
    denominator = batch.loss_weights.sum().clamp_min(1)
    loss = (qa_means * batch.loss_weights).sum() / denominator
    return loss, {
        "ce_numerator": float((qa_means.detach() * batch.loss_weights).sum().item()),
        "loss_weight": float(denominator.detach().item()),
        "supervised_tokens": float(flat_targets.numel()),
    }


class MaterializedQAReader:
    """Seekable row reader for a selected, length-sorted QA Parquet file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.parquet = pq.ParquetFile(self.path, memory_map=True)
        self.counts = [
            self.parquet.metadata.row_group(i).num_rows
            for i in range(self.parquet.metadata.num_row_groups)
        ]
        self.total_rows = sum(self.counts)

    def iter_rows(self, start: int = 0) -> Iterator[dict[str, Any]]:
        if not 0 <= start <= self.total_rows:
            raise ValueError(f"Invalid selected-QA cursor {start}")
        offset = 0
        for group_index, count in enumerate(self.counts):
            group_end = offset + count
            if group_end <= start:
                offset = group_end
                continue
            table = self.parquet.read_row_group(group_index)
            local_start = max(0, start - offset)
            for row in table.slice(local_start).to_pylist():
                yield row
            offset = group_end

    def read_slice(self, start: int, count: int = 1) -> list[dict[str, Any]]:
        output = []
        for row in self.iter_rows(start):
            output.append(row)
            if len(output) >= count:
                break
        return output

    def longest_offsets(self, count: int = 5) -> list[int]:
        """Return exact global row offsets for the longest frozen QAs."""

        import heapq

        heap: list[tuple[int, int]] = []
        offset = 0
        for group_index, group_count in enumerate(self.counts):
            values = self.parquet.read_row_group(
                group_index, columns=["input_ids"]
            ).column("input_ids")
            for local_index, value in enumerate(values):
                item = (len(value), offset + local_index)
                if len(heap) < count:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            offset += group_count
        return [row_offset for _length, row_offset in sorted(heap, reverse=True)]

def move_optimizer_state_to_parameter_devices(optimizer) -> None:
    """Restore CPU-serialized optimizer tensors beside their parameters.

    PyTorch does not consistently migrate every optimizer-state tensor when a
    state dict is loaded from an explicitly CPU-mapped checkpoint.  Adam's
    foreach implementation then fails when CUDA parameters and CPU moments are
    mixed.  Walk each parameter-keyed state recursively so resumes preserve the
    serialized values while restoring only their storage device.
    """

    def move(value, device):
        if torch.is_tensor(value):
            return value.to(device=device)
        if isinstance(value, dict):
            return {key: move(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item, device) for item in value)
        return value

    for parameter, state in optimizer.state.items():
        optimizer.state[parameter] = move(state, parameter.device)
