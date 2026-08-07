# based on
# https://github.com/MeetKai/functionary/blob/aa3dbdd65f7e388f2386622606bdfeec95c2b863/functionary/train/packing/packed_dataset.py
import json
import logging
import os
import pprint

import numpy as np

from ctx_to_lora.utils import check_is_iterable, concat_list

logger = logging.getLogger()


def block_diagonal_causal_mask(sequence_ids: np.ndarray) -> np.ndarray:
    """Reference mask used by packed-answer isolation parity tests."""

    sequence_ids = np.asarray(sequence_ids)
    indices = np.arange(sequence_ids.size)
    return (sequence_ids[:, None] == sequence_ids[None, :]) & (
        indices[:, None] >= indices[None, :]
    )


def validate_packed_qa_isolation(
    position_ids: np.ndarray, sequence_ids: np.ndarray, labels: np.ndarray
) -> None:
    """Validate the compact FlashAttention/FlexAttention sequence encoding.

    Gemma's supported packed attention paths derive variable-length sequence
    boundaries from reset ``position_ids``. ``sequence_ids`` is retained as an
    independent witness so malformed boundaries fail before model forward.
    """

    positions = np.asarray(position_ids).reshape(-1)
    sequences = np.asarray(sequence_ids).reshape(-1)
    flat_labels = np.asarray(labels).reshape(-1)
    if not (positions.size == sequences.size == flat_labels.size):
        raise ValueError("Packed QA boundary arrays have different lengths")
    if not positions.size or positions[0] != 0 or sequences[0] != 0:
        raise ValueError("Packed QA boundaries must begin with sequence/position zero")
    starts = np.flatnonzero(positions == 0)
    if not np.array_equal(starts, np.flatnonzero(np.r_[True, np.diff(sequences) != 0])):
        raise ValueError("position_ids resets and packed_sequence_ids disagree")
    for sequence, start in enumerate(starts):
        end = starts[sequence + 1] if sequence + 1 < len(starts) else len(positions)
        if not np.array_equal(positions[start:end], np.arange(end - start)):
            raise ValueError("Packed QA position IDs do not reset monotonically")
        if not np.all(sequences[start:end] == sequence):
            raise ValueError("Packed sequence IDs are not contiguous")
        if flat_labels[start] != -100:
            raise ValueError("A QA's first token cannot be supervised across a boundary")


def pack_data_points_by_length(
    lens: list[list[int]],
    ctx_lens: list[list[int]],
    max_packed_inp_len: int,
    max_packed_ctx_len: int,
    max_size: int = -1,
) -> tuple[list[int], list[int]]:
    if not lens:
        return []

    len_arr = np.array([sum(l) for l in lens], dtype=np.long)
    ctx_len_arr = np.array([sum(l) for l in ctx_lens], dtype=np.long)
    n = len(len_arr)
    assert len(ctx_len_arr) == n, "Length of ctx_len_arr must match length of lens"

    if n == 1:
        return (
            [0]
            if len_arr[0] <= max_packed_inp_len and ctx_len_arr[0] <= max_packed_ctx_len
            else []
        )

    # Create cumulative sum arrays for efficient range queries
    cumsum_inp_len = np.cumsum(len_arr)
    cumsum_ctx_len = np.cumsum(ctx_len_arr)

    idx_pairs = []
    i = 0

    while i < n:
        # Find the maximum j such that sum(lens[i:j+1]) <= max_packed_inp_len
        start_sum_inp = cumsum_inp_len[i - 1] if i > 0 else 0
        valid_ends_inp = (cumsum_inp_len[i:] - start_sum_inp) <= max_packed_inp_len

        start_sum_ctx = cumsum_ctx_len[i - 1] if i > 0 else 0
        valid_ends_ctx = (cumsum_ctx_len[i:] - start_sum_ctx) <= max_packed_ctx_len
        valid_ends = valid_ends_inp & valid_ends_ctx

        if not np.any(valid_ends):
            # Single item exceeds max_packed_inp_len, skip it
            logging.debug(
                f"Skipping item {i} with input length {len_arr[i]} and context length {ctx_len_arr[i]}"
            )
            i += 1
            continue

        # Find the last valid index
        max_valid_idx = i + np.where(valid_ends)[0][-1]

        # Apply max_size constraint
        if max_size > 0:
            max_valid_idx = min(max_valid_idx, i + max_size - 1)

        idx_pairs.append((i, max_valid_idx + 1))
        i = max_valid_idx + 1

    return idx_pairs


def pack_data_points_FA(
    batch: dict[str, any],
) -> dict[str, np.ndarray]:
    if not batch:
        raise ValueError("Batch is empty")

    # Pre-allocate lists with known sizes
    total_ctx_len = sum(len(y) for x in batch["ctx_ids"] for y in x)
    total_inp_len = sum(len(y) for x in batch["input_ids"] for y in x)

    ctx_ids = np.empty(total_ctx_len, dtype=np.long)
    ctx_position_ids = np.empty(total_ctx_len, dtype=np.long)
    input_ids = np.empty(total_inp_len, dtype=np.long)
    position_ids = np.empty(total_inp_len, dtype=np.long)
    labels = np.empty(total_inp_len, dtype=np.long)

    has_logprobs = "logprobs_vals" in batch

    if has_logprobs:
        sequences = zip(
            batch["input_ids"],
            batch["labels"],
            batch["logprobs_vals"],
            batch["logprobs_indices"],
        )
        n_labels = sum(len(y) for x in batch["logprobs_vals"] for y in x)
        k = len(batch["logprobs_vals"][0][0][0])  # assuming all have same k
        logprobs_vals = np.empty((n_labels, k), dtype=np.float32)
        logprobs_indices = np.empty((n_labels, k), dtype=np.int32)
        logits_offset = 0
    else:
        sequences = zip(batch["input_ids"], batch["labels"])

    offset = 0
    for sample in sequences:
        input_ids_b, labels_b = sample[:2]
        inp_start = offset

        # compute position_ids for each sub-list in input_ids_b
        local_start = inp_start
        for ids_b in input_ids_b:
            local_end = local_start + len(ids_b)
            position_ids[local_start:local_end] = np.arange(len(ids_b), dtype=np.int32)
            local_start = local_end

        input_ids_b = concat_list(input_ids_b)
        labels_b = concat_list(labels_b)

        inp_len = len(input_ids_b)
        inp_end = offset + inp_len

        input_ids[inp_start:inp_end] = input_ids_b
        labels[inp_start:inp_end] = labels_b
        offset += inp_len

        if has_logprobs:
            logprobs_vals_b, logprobs_indices_b = sample[2:]
            logprobs_vals_b = concat_list(logprobs_vals_b)
            logprobs_indices_b = concat_list(logprobs_indices_b)
            logits_len = len(logprobs_vals_b)
            logprobs_vals[logits_offset : logits_offset + logits_len] = logprobs_vals_b
            logprobs_indices[logits_offset : logits_offset + logits_len] = (
                logprobs_indices_b
            )
            logits_offset += logits_len

    ctx_offset = 0
    for ctx_ids_b in batch["ctx_ids"]:
        local_start = ctx_offset
        for ctx_ids_b_item in ctx_ids_b:
            local_end = local_start + len(ctx_ids_b_item)
            ctx_position_ids[local_start:local_end] = np.arange(
                len(ctx_ids_b_item), dtype=np.int32
            )
            local_start = local_end

        ctx_ids_b = concat_list(ctx_ids_b)
        ctx_len = len(ctx_ids_b)
        ctx_start, ctx_end = ctx_offset, ctx_offset + ctx_len
        ctx_ids[ctx_start:ctx_end] = ctx_ids_b
        ctx_offset += ctx_len

    out = {
        "ctx_ids": ctx_ids,
        "ctx_position_ids": ctx_position_ids,
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
    }
    if has_logprobs:
        out["logprobs_vals"] = logprobs_vals
        out["logprobs_indices"] = logprobs_indices
    return out


def pack_batch(
    batch: dict[str, any],
    max_packed_inp_len: int,
    max_packed_ctx_len: int,
    max_packed_size: int = -1,
    metadata_path: str = "",
) -> dict[str, any]:
    need_flatten = check_is_iterable(batch["input_ids"][0][0])
    assert need_flatten, (
        f"Packing requires the input_ids to be nested "
        f"(allowing multiple QAs per sample), but got {batch['input_ids'][0]}"
    )

    n_queries = [len(x) for x in batch["input_ids"]]
    n_ctx_chunks = [len(x) for x in batch["ctx_ids"]]
    inp_lens = [[len(y) for y in x] for x in batch["input_ids"]]
    inp_count = len(inp_lens)
    if "ctx_ids" not in batch:
        raise ValueError("Batch must contain 'ctx_ids' and 'labels' keys")
    # we do not pad so we can just take the length of the tokens
    ctx_lens = [[len(y) for y in x] for x in batch["ctx_ids"]]

    # Group indices
    idx_pairs = pack_data_points_by_length(
        inp_lens,
        ctx_lens,
        max_packed_inp_len,
        max_packed_ctx_len,
        max_packed_size,
    )

    # Pack groups
    packed_batch = {
        "ctx_ids": [],
        "ctx_position_ids": [],
        "input_ids": [],
        "position_ids": [],
        "labels": [],
        "n_queries": [],
        "n_ctx_chunks": [],
    }
    has_logprobs = "logprobs_vals" in batch
    if has_logprobs:
        packed_batch["logprobs_vals"] = []
        packed_batch["logprobs_indices"] = []

    packing_efficiency_ratios = []
    ctx_packing_efficiency_ratios = []

    for idx_pair in idx_pairs:
        start_idx, end_idx = idx_pair[0], idx_pair[1]
        group_items = {
            "ctx_ids": batch["ctx_ids"][start_idx:end_idx],
            "input_ids": batch["input_ids"][start_idx:end_idx],
            "labels": batch["labels"][start_idx:end_idx],
        }
        if has_logprobs:
            group_items["logprobs_vals"] = batch["logprobs_vals"][start_idx:end_idx]
            group_items["logprobs_indices"] = batch["logprobs_indices"][
                start_idx:end_idx
            ]
        packed_item = pack_data_points_FA(group_items)
        packed_batch["ctx_ids"].append(packed_item["ctx_ids"])
        packed_batch["ctx_position_ids"].append(packed_item["ctx_position_ids"])
        packed_batch["input_ids"].append(packed_item["input_ids"])
        packed_batch["position_ids"].append(packed_item["position_ids"])
        packed_batch["labels"].append(packed_item["labels"])
        packed_batch["n_queries"].append(n_queries[start_idx:end_idx])
        packed_batch["n_ctx_chunks"].append(n_ctx_chunks[start_idx:end_idx])
        if has_logprobs:
            packed_batch["logprobs_vals"].append(packed_item["logprobs_vals"])
            packed_batch["logprobs_indices"].append(packed_item["logprobs_indices"])

        if max_packed_inp_len > 0:
            inp_efficiency = len(packed_item["input_ids"]) / max_packed_inp_len
            packing_efficiency_ratios.append(inp_efficiency)

        if max_packed_ctx_len > 0:
            ctx_efficiency = len(packed_item["ctx_ids"]) / max_packed_ctx_len
            ctx_packing_efficiency_ratios.append(ctx_efficiency)

    # Calculate length statistics
    packed_inp_lens_arr = np.array([len(x) for x in packed_batch["input_ids"]])
    packed_ctx_lens_arr = np.array([len(x) for x in packed_batch["ctx_ids"]])

    # Log performance statistics
    avg_inp_packing_efficiency = (
        np.mean(packing_efficiency_ratios) if packing_efficiency_ratios else 0
    )
    avg_ctx_packing_efficiency = (
        np.mean(ctx_packing_efficiency_ratios) if ctx_packing_efficiency_ratios else 0
    )

    # Create packing statistics dictionary
    packing_stats = {
        "original_samples": inp_count,
        "packed_samples": len(idx_pairs),
        "avg_inp_packing_efficiency": float(avg_inp_packing_efficiency),
        "avg_ctx_packing_efficiency": float(avg_ctx_packing_efficiency),
        "input_ids_length_stats": {
            "avg": float(np.mean(packed_inp_lens_arr)),
            "std": float(np.std(packed_inp_lens_arr)),
            "min": int(np.min(packed_inp_lens_arr)),
            "max": int(np.max(packed_inp_lens_arr)),
        },
        "context_ids_length_stats": {
            "avg": float(np.mean(packed_ctx_lens_arr)),
            "std": float(np.std(packed_ctx_lens_arr)),
            "min": int(np.min(packed_ctx_lens_arr)),
            "max": int(np.max(packed_ctx_lens_arr)),
        },
    }

    # Save to metadata_path if provided
    if metadata_path:
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(packing_stats, f, indent=4)

    logging.debug(f"Packing stats:\n{pprint.pformat(packing_stats, indent=2)}")

    return packed_batch
