#!/usr/bin/env python3
"""Train the historical static Code2LoRA architecture on exact RepoQA CE."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from repotune_issuefix.code2lora_gemma import (
    Code2LoRAHead,
    count_head_parameters,
    discover_module_types_and_dims,
    get_module_specs,
    inject_lora_weights,
    load_gemma4_model,
    replace_with_lora,
)
from repotune_issuefix.repoqa_baselines import (
    batch_rows,
    collate_qa_rows,
    group_rows,
    length_bucket,
    load_ready,
    move_optimizer_state_to_parameter_devices,
    read_group_qas,
    sha256_file,
    stable_u64,
    weighted_answer_ce,
)


MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def cuda_stats() -> dict[str, float]:
    return {
        "cuda_allocated_gb": torch.cuda.memory_allocated() / 2**30,
        "cuda_reserved_gb": torch.cuda.memory_reserved() / 2**30,
        "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
    }


def load_embeddings(path: Path) -> dict[tuple[str, str], torch.Tensor]:
    table = pq.read_table(path, memory_map=True)
    result = {}
    for row in table.to_pylist():
        key = (str(row["repo_id"]), str(row["commit_sha"]))
        value = torch.tensor(row["repo_state_embedding"], dtype=torch.float32)
        if value.numel() != 2048 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid Code2LoRA embedding for {key}")
        result[key] = value
    return result


def set_progress_lr(optimizer, base_lr: float, consumed: int, total: int, warmup_ratio: float) -> float:
    progress = min(1.0, consumed / max(1, total))
    if progress < warmup_ratio:
        multiplier = progress / max(warmup_ratio, 1e-12)
    else:
        decay = (progress - warmup_ratio) / max(1e-12, 1.0 - warmup_ratio)
        multiplier = 0.5 * (1.0 + math.cos(math.pi * decay))
    lr = base_lr * multiplier
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def cpu_optimizer_state(optimizer) -> dict[str, Any]:
    raw = optimizer.state_dict()
    copied_state = {}
    for parameter_id, values in raw["state"].items():
        copied_state[parameter_id] = {
            key: value.detach().cpu() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in values.items()
        }
    return {
        "state": copied_state,
        "param_groups": copy.deepcopy(raw["param_groups"]),
    }


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(output: Path, head, optimizer, state: dict[str, Any]) -> Path:
    checkpoint = output / f"checkpoint-qa{int(state['consumed_qas']):08d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()}, checkpoint / "code2lora_head.pt")
    torch.save(cpu_optimizer_state(optimizer), checkpoint / "optimizer.pt")
    torch.save(rng_state(), checkpoint / "rng_state.pt")
    temporary = checkpoint / "state.json.tmp"
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(checkpoint / "state.json")
    (output / "latest_checkpoint.txt").write_text(str(checkpoint) + "\n")
    return checkpoint


def load_checkpoint(path: Path, head, optimizer) -> dict[str, Any]:
    head.load_state_dict(torch.load(path / "code2lora_head.pt", map_location="cpu", weights_only=True))
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False))
    move_optimizer_state_to_parameter_devices(optimizer)
    rng_path = path / "rng_state.pt"
    if not rng_path.exists():
        raise ValueError(f"Checkpoint is missing exact-resume RNG state: {rng_path}")
    restore_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
    return json.loads((path / "state.json").read_text())


def stage_cuda_optimizer_state_on_cpu(optimizer) -> list[tuple[dict, str, torch.device]]:
    """Temporarily free CUDA optimizer memory without changing Adam values.

    A handful of frozen RepoQA examples are longer than the nominal packed
    token budget.  They must remain single-example batches to preserve the
    immutable 10M-QA corpus, but their attention workspace overlaps with the
    roughly 5.6 GiB of Adam moments for the Code2LoRA head.  Adam state is not
    read until ``optimizer.step()``, so moving it to CPU for forward/backward
    is optimizer-equivalent and gives those outliers enough device headroom.
    """

    moved: list[tuple[dict, str, torch.device]] = []
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if torch.is_tensor(value) and value.device.type == "cuda":
                device = value.device
                state[key] = value.to("cpu")
                moved.append((state, key, device))
    if moved:
        torch.cuda.empty_cache()
    return moved


def restore_staged_optimizer_state(
    moved: list[tuple[dict, str, torch.device]],
) -> None:
    """Restore optimizer tensors staged by ``stage_cuda_optimizer_state_on_cpu``."""

    for state, key, device in moved:
        state[key] = state[key].to(device)


def enable_decoder_checkpointing_in_eval(model) -> list[torch.nn.Module]:
    """Enable decoder-layer recomputation without enabling child dropout.

    ``GradientCheckpointingLayer`` gates recomputation on the decoder layer's
    own ``training`` flag.  Code2LoRA intentionally keeps the frozen Gemma base
    in eval mode, so the global checkpointing switch alone is inactive.  Set
    only each decoder layer's local flag for oversized batches; its attention,
    MLP, and dropout children remain in eval mode and therefore preserve the
    historical deterministic forward behavior.
    """

    text_model = model.model.language_model
    changed: list[torch.nn.Module] = []
    for layer in text_model.layers:
        if getattr(layer, "gradient_checkpointing", False) and not layer.training:
            layer.training = True
            changed.append(layer)
    if not changed:
        raise RuntimeError("No eval-mode Gemma decoder layers enabled checkpointing")
    return changed


def restore_decoder_eval(changed: list[torch.nn.Module]) -> None:
    for layer in changed:
        layer.training = False


def sorted_group_qas(group: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows = read_group_qas(group)
    rows.sort(
        key=lambda row: (
            length_bucket(len(row["input_ids"])),
            stable_u64(seed, row["logical_example_id"]),
        )
    )
    return rows


@torch.no_grad()
def evaluate(
    model,
    head,
    specs,
    groups,
    embeddings,
    tokenizer,
    device,
    token_budget,
    seed,
    ce_chunk_tokens,
):
    model.eval()
    head.eval()
    numerator = denominator = 0.0
    qas = 0
    started = time.time()
    for group in groups:
        key = (str(group["repo_id"]), str(group["commit_sha"]))
        head_out = head(embeddings[key].to(device).unsqueeze(0))
        inject_lora_weights(model, specs, head_out, batch_index=0)
        for rows in batch_rows(sorted_group_qas(group, seed), token_budget):
            batch = collate_qa_rows(rows, tokenizer.pad_token_id, device)
            _loss, metrics = weighted_answer_ce(
                model, batch, ce_chunk_tokens=ce_chunk_tokens
            )
            numerator += metrics["ce_numerator"]
            denominator += metrics["loss_weight"]
            qas += batch.qa_count
    return {
        "eval_loss": numerator / max(denominator, 1.0),
        "eval_loss_weight": denominator,
        "eval_qas": qas,
        "eval_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--repoqa-ready", required=True)
    parser.add_argument("--baseline-data-ready", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--eval-token-budget", type=int, default=4096)
    parser.add_argument(
        "--ce-chunk-tokens",
        type=int,
        default=512,
        help=(
            "Maximum answer positions projected over the vocabulary at once; "
            "training chunks are activation-checkpointed for bounded memory"
        ),
    )
    parser.add_argument("--target-qas", type=int, default=10_000_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--checkpoint-qas", default="1450506,3589498,6490000,10000000")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--preflight-profile", choices=("memory", "throughput"), default="memory"
    )
    parser.add_argument(
        "--preflight-context-group-id",
        default="",
        help="Run preflight on one exact frozen context group",
    )
    parser.add_argument(
        "--preflight-batch-index",
        type=int,
        default=-1,
        help="Within a targeted preflight group, run only this deterministic batch",
    )
    args = parser.parse_args()

    if args.ce_chunk_tokens <= 0:
        parser.error("--ce-chunk-tokens must be positive")
    if args.preflight_context_group_id and not args.preflight:
        parser.error("--preflight-context-group-id requires --preflight")
    if args.preflight_batch_index >= 0 and not args.preflight_context_group_id:
        parser.error("--preflight-batch-index requires --preflight-context-group-id")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events = output / "events.jsonl"
    ready = load_ready(args.repoqa_ready)
    baseline_ready_path = Path(args.baseline_data_ready)
    baseline_ready = json.loads(baseline_ready_path.read_text())
    embedding_path = baseline_ready_path.parent / "code2lora_snapshot_embeddings.parquet"
    if sha256_file(embedding_path) != baseline_ready["code2lora_embeddings"]["sha256"]:
        raise ValueError("Code2LoRA embedding index hash mismatch")
    embeddings = load_embeddings(embedding_path)
    train_groups = [
        group
        for stage in ("stage1", "stage2a")
        for group in group_rows(ready, stage, "train")
    ]
    if args.preflight_context_group_id:
        train_groups = [
            group
            for group in train_groups
            if str(group["context_group_id"]) == args.preflight_context_group_id
        ]
        if len(train_groups) != 1:
            raise ValueError(
                "Expected exactly one group for --preflight-context-group-id="
                f"{args.preflight_context_group_id!r}, found {len(train_groups)}"
            )
    elif args.preflight and args.preflight_profile == "memory":
        candidates = sorted(
            train_groups,
            key=lambda group: (
                -max([int(value) for value in group.get("qa_pack_token_counts", [])] or [0]),
                -float(group.get("answer_side_tokens", 0)) / max(1, int(group.get("qa_count", 1))),
                str(group["context_group_id"]),
            ),
        )
        candidates = candidates[:64]
        train_groups = [
            max(
                candidates,
                key=lambda group: max(
                    len(row["input_ids"]) for row in read_group_qas(group)
                ),
            )
        ]
    elif args.preflight:
        ordered = sorted(
            train_groups,
            key=lambda group: (
                sum(int(value) for value in group.get("qa_pack_token_counts", []))
                / max(1, int(group["qa_count"])),
                str(group["context_group_id"]),
            ),
        )
        train_groups = [
            ordered[int((len(ordered) - 1) * quantile)]
            for quantile in (0.1, 0.3, 0.5, 0.7, 0.9)
        ]
    val_groups = [
        group
        for stage in ("stage1", "stage2a")
        for group in group_rows(ready, stage, "val")
    ]
    needed = {
        (str(group["repo_id"]), str(group["commit_sha"]))
        for group in train_groups + val_groups
    }
    if needed - set(embeddings):
        raise ValueError(f"Missing {len(needed - set(embeddings))} snapshot embeddings")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = load_gemma4_model(
        args.model,
        dtype=torch.bfloat16,
        device=device.type,
        local_files_only=args.local_files_only,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    specs = get_module_specs(model, TARGET_MODULES)
    if len(specs) != 205:
        raise ValueError(f"Expected 205 Gemma text projections, found {len(specs)}")
    type_dims = discover_module_types_and_dims(specs)
    replace_with_lora(model, specs, rank=8, alpha=16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    head = Code2LoRAHead(2048, type_dims, hidden_dim=1024, rank=8).to(device)
    expected_head = count_head_parameters(2048, type_dims, hidden_dim=1024, rank=8)
    actual_head = sum(value.numel() for value in head.parameters())
    if expected_head != actual_head or actual_head != 746_264_604:
        raise ValueError(f"Unexpected Code2LoRA head size {actual_head:,}")
    optimizer = AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    state = {"consumed_qas": 0, "group_cursor": 0, "optimizer_steps": 0, "best_eval_loss": None}
    if args.resume:
        state = load_checkpoint(Path(args.resume), head, optimizer)
        # A targeted preflight may deliberately load production weights and
        # Adam moments while replacing the full training-group list with one
        # known stress case.  Its production cursor is not meaningful in that
        # one-group view.
        if args.preflight_context_group_id:
            state["group_cursor"] = 0
    set_progress_lr(optimizer, args.lr, int(state["consumed_qas"]), args.target_qas, args.warmup_ratio)
    append_jsonl(events, {
        "event": "start",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "head_parameters": actual_head,
        "snapshot_count": len(train_groups),
        "embedding_count": len(embeddings),
        "target_qas": args.target_qas,
        "token_budget": args.token_budget,
        **cuda_stats(),
    })
    run_config = {
        "format": "repoqa_code2lora_run_v1",
        "args": vars(args),
        "model_revision": MODEL_REVISION,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "head_parameters": actual_head,
        "target_projection_count": len(specs),
        "rank": 8,
        "alpha": 16,
        "embedding_index_sha256": baseline_ready["code2lora_embeddings"]["sha256"],
        "source_ready_sha256": baseline_ready["source_ready_sha256"],
        "loss": "answer_token_mean_ce_weighted_by_qa_family",
        "qa_loss_weights": {"ast": 1.0, "llm": 1.8},
    }
    (output / "RUN_CONFIG.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    )
    checkpoints = sorted({int(value) for value in args.checkpoint_qas.split(",") if value})
    pending = [value for value in checkpoints if value > int(state["consumed_qas"])]
    consumed = int(state["consumed_qas"])
    start_consumed = consumed
    group_cursor = int(state["group_cursor"])
    started = time.time()
    # Match the historical Code2LoRA setup: the frozen base stays in eval mode
    # while the repository-to-adapter head is optimized.
    model.eval()
    head.train()
    stop = False
    while group_cursor < len(train_groups) and not stop:
        group = train_groups[group_cursor]
        key = (str(group["repo_id"]), str(group["commit_sha"]))
        rows_for_group = sorted_group_qas(group, args.seed)
        if (
            args.preflight
            and args.preflight_profile == "memory"
            and not args.preflight_context_group_id
        ):
            rows_for_group.reverse()
        group_steps = 0
        for batch_index, rows in enumerate(batch_rows(rows_for_group, args.token_budget)):
            if (
                args.preflight_batch_index >= 0
                and batch_index != args.preflight_batch_index
            ):
                continue
            optimizer.zero_grad(set_to_none=True)
            max_sequence_tokens = max(len(row["input_ids"]) for row in rows)
            oversized_batch = max_sequence_tokens > args.token_budget
            staged_optimizer_state = (
                stage_cuda_optimizer_state_on_cpu(optimizer)
                if oversized_batch
                else []
            )
            checkpointed_decoder_layers = (
                enable_decoder_checkpointing_in_eval(model)
                if oversized_batch
                else []
            )
            if oversized_batch:
                append_jsonl(events, {
                    "event": "oversized_batch_optimizer_offload",
                    "consumed_qas": consumed,
                    "group_cursor": group_cursor,
                    "batch_index": batch_index,
                    "max_sequence_tokens": max_sequence_tokens,
                    "attention_implementation": getattr(
                        model.config.get_text_config(),
                        "_attn_implementation",
                        "sdpa",
                    ),
                    "checkpointed_decoder_layers": len(checkpointed_decoder_layers),
                    "staged_optimizer_gb": sum(
                        state_dict[key].numel() * state_dict[key].element_size()
                        for state_dict, key, _device in staged_optimizer_state
                    ) / 2**30,
                    **cuda_stats(),
                })
            head_out = head(embeddings[key].to(device).unsqueeze(0))
            inject_lora_weights(model, specs, head_out, batch_index=0)
            batch = collate_qa_rows(rows, tokenizer.pad_token_id, device)
            batch_qa_count = batch.qa_count
            loss, metrics = weighted_answer_ce(
                model, batch, ce_chunk_tokens=args.ce_chunk_tokens
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Code2LoRA CE")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            loss_value = float(loss.detach().item())
            del batch, loss, head_out
            if oversized_batch:
                restore_decoder_eval(checkpointed_decoder_layers)
            if staged_optimizer_state:
                torch.cuda.empty_cache()
                restore_staged_optimizer_state(staged_optimizer_state)
            optimizer.step()
            consumed += batch_qa_count
            state["consumed_qas"] = consumed
            state["optimizer_steps"] = int(state["optimizer_steps"]) + 1
            group_steps += 1
            state["last_loss"] = loss_value
            state["last_grad_norm"] = float(grad_norm)
            state["qas_per_second"] = (consumed - start_consumed) / max(
                time.time() - started, 1e-6
            )
            lr = set_progress_lr(optimizer, args.lr, consumed, args.target_qas, args.warmup_ratio)
            if state["optimizer_steps"] % args.log_every == 0:
                append_jsonl(events, {
                    "event": "train",
                    "consumed_qas": consumed,
                    "group_cursor": group_cursor,
                    "optimizer_steps": state["optimizer_steps"],
                    "loss": loss_value,
                    "grad_norm": float(grad_norm),
                    "learning_rate": lr,
                    "qas_per_second": state["qas_per_second"],
                    **metrics,
                    **cuda_stats(),
                })
            if (
                args.preflight
                and args.preflight_profile == "throughput"
                and group_steps >= max(1, args.max_steps or 5)
            ):
                break
            if (
                args.max_steps
                and args.preflight_profile == "memory"
                and state["optimizer_steps"] >= args.max_steps
            ):
                stop = True
                break
        if args.preflight_batch_index >= 0 and group_steps != 1:
            raise ValueError(
                f"Targeted preflight batch {args.preflight_batch_index} was not run "
                f"exactly once (ran {group_steps} times)"
            )
        group_cursor += 1
        state["group_cursor"] = group_cursor
        while pending and consumed >= pending[0]:
            milestone = pending.pop(0)
            validation = evaluate(
                model, head, specs, val_groups, embeddings, tokenizer, device,
                args.eval_token_budget, args.seed, args.ce_chunk_tokens,
            )
            state["last_validation"] = validation
            prior = state.get("best_eval_loss")
            is_best = prior is None or validation["eval_loss"] < prior
            if is_best:
                state["best_eval_loss"] = validation["eval_loss"]
                state["best_consumed_qas"] = consumed
            checkpoint = save_checkpoint(output, head, optimizer, state)
            if is_best:
                (output / "best_checkpoint.txt").write_text(str(checkpoint) + "\n")
            append_jsonl(events, {
                "event": "checkpoint",
                "requested_milestone": milestone,
                "checkpoint": str(checkpoint),
                **state,
                **validation,
                **cuda_stats(),
            })
            model.eval()
            head.train()
            gc.collect()
            torch.cuda.empty_cache()
    state["completed"] = consumed == args.target_qas and group_cursor == len(train_groups)
    state["elapsed_seconds"] = time.time() - started
    if args.preflight:
        result = {"event": "preflight_stop", **state, **cuda_stats()}
        (output / "PREFLIGHT.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        append_jsonl(events, result)
    else:
        final = save_checkpoint(output, head, optimizer, state)
        append_jsonl(events, {"event": "complete" if state["completed"] else "stopped", "checkpoint": str(final), **state, **cuda_stats()})


if __name__ == "__main__":
    main()
