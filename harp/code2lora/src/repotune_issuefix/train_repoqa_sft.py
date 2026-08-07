#!/usr/bin/env python3
"""Pure-CE full-parameter and conventional-LoRA Gemma RepoQA baselines."""

from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import math
import os
import random
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import Adafactor, AutoTokenizer

from repotune_issuefix.code2lora_gemma import load_gemma4_model
from repotune_issuefix.repoqa_baselines import (
    MaterializedQAReader,
    batch_rows,
    collate_qa_rows,
    move_optimizer_state_to_parameter_devices,
    sha256_file,
    weighted_answer_ce,
)


MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
TARGET_REGEX = r"^model\.language_model\.layers\..*\.(q_proj|k_proj|v_proj|o_proj|up_proj|gate_proj|down_proj)$"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def cuda_stats() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_allocated_gb": torch.cuda.memory_allocated() / 2**30,
        "cuda_reserved_gb": torch.cuda.memory_reserved() / 2**30,
        "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
    }


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


def set_phase_lr(
    optimizer,
    base_lr: float,
    consumed: int,
    schedule_start_qas: int,
    target_qas: int,
    warmup_ratio: float,
) -> float:
    return set_progress_lr(
        optimizer,
        base_lr,
        max(0, consumed - schedule_start_qas),
        target_qas - schedule_start_qas,
        warmup_ratio,
    )


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.named_parameters()
        if value.requires_grad
    }


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
    """Capture every RNG used by these trainers for bitwise-resumable batches."""

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


def save_checkpoint(output: Path, model, optimizer, state: dict[str, Any]) -> Path:
    checkpoint = output / f"checkpoint-qa{int(state['consumed_qas']):07d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save(trainable_state(model), checkpoint / "trainable_model.pt")
    torch.save(cpu_optimizer_state(optimizer), checkpoint / "optimizer.pt")
    torch.save(rng_state(), checkpoint / "rng_state.pt")
    temporary = checkpoint / "state.json.tmp"
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(checkpoint / "state.json")
    (output / "latest_checkpoint.txt").write_text(str(checkpoint) + "\n")
    return checkpoint


def load_checkpoint(path: Path, model, optimizer) -> dict[str, Any]:
    missing, unexpected = model.load_state_dict(
        torch.load(path / "trainable_model.pt", map_location="cpu", weights_only=True),
        strict=False,
    )
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    missing_trainable = trainable.intersection(missing)
    if missing_trainable or unexpected:
        raise ValueError(
            f"Checkpoint mismatch missing_trainable={sorted(missing_trainable)[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False))
    move_optimizer_state_to_parameter_devices(optimizer)
    rng_path = path / "rng_state.pt"
    if not rng_path.exists():
        raise ValueError(f"Checkpoint is missing exact-resume RNG state: {rng_path}")
    restore_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
    return json.loads((path / "state.json").read_text())


@torch.no_grad()
def evaluate(model, reader: MaterializedQAReader, tokenizer, device, token_budget: int) -> dict[str, float]:
    model.eval()
    numerator = 0.0
    denominator = 0.0
    qas = 0
    started = time.time()
    for rows in batch_rows(reader.iter_rows(), token_budget):
        batch = collate_qa_rows(rows, tokenizer.pad_token_id, device)
        _loss, metrics = weighted_answer_ce(model, batch)
        numerator += metrics["ce_numerator"]
        denominator += metrics["loss_weight"]
        qas += batch.qa_count
        del batch
    return {
        "eval_loss": numerator / max(denominator, 1.0),
        "eval_loss_weight": denominator,
        "eval_qas": qas,
        "eval_seconds": time.time() - started,
    }


def configure_model(args, device):
    model = load_gemma4_model(
        args.model,
        dtype=torch.bfloat16,
        device=device.type,
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    if args.mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for name, parameter in model.named_parameters():
            if name.startswith("model.language_model.") or name.startswith("lm_head."):
                parameter.requires_grad = True
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=TARGET_REGEX,
        )
        model = get_peft_model(model, config)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "lora16"), required=True)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--data-ready", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-token-budget", type=int, default=4096)
    parser.add_argument("--eval-token-budget", type=int, default=4096)
    parser.add_argument("--target-qas", type=int, default=1_000_000)
    parser.add_argument(
        "--schedule-start-qas",
        type=int,
        default=0,
        help="Start a fresh warmup/cosine phase at this cumulative QA exposure",
    )
    parser.add_argument("--full-lr", type=float, default=2e-6)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--checkpoint-qas", default="100000,250000,500000,1000000")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="Preflight-only early stop")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--preflight-profile", choices=("memory", "throughput"), default="memory"
    )
    args = parser.parse_args()

    if not 0 <= args.schedule_start_qas < args.target_qas:
        parser.error("--schedule-start-qas must be in [0, target_qas)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events = output / "events.jsonl"
    data_ready_path = Path(args.data_ready)
    data_ready = json.loads(data_ready_path.read_text())
    if data_ready.get("format") != "repoqa_baseline_data_v1":
        raise ValueError("Unexpected baseline data manifest")
    train_path = data_ready_path.parent / data_ready["train"].get(
        "path", "sft_train_1m.parquet"
    )
    val_path = data_ready_path.parent / "sft_val_12800.parquet"
    if sha256_file(train_path) != data_ready["train"]["sha256"]:
        raise ValueError("Frozen SFT training data hash mismatch")
    if sha256_file(val_path) != data_ready["validation"]["sha256"]:
        raise ValueError("Frozen validation data hash mismatch")
    train_reader = MaterializedQAReader(train_path)
    val_reader = MaterializedQAReader(val_path)
    if (
        int(data_ready.get("target", 0)) != args.target_qas
        or train_reader.total_rows != args.target_qas
        or val_reader.total_rows != 12_800
    ):
        raise ValueError("Frozen SFT row counts changed")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = configure_model(args, device)
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    total = sum(value.numel() for value in model.parameters())
    if args.mode == "lora16" and trainable != 24_158_208:
        raise ValueError(f"Expected 24,158,208 rank-16 LoRA parameters, found {trainable:,}")
    if args.mode == "full" and not (1_000_000_000 < trainable < total):
        raise ValueError(f"Unexpected full-text trainable parameter count {trainable:,}/{total:,}")
    base_lr = args.full_lr if args.mode == "full" else args.lora_lr
    if args.mode == "full":
        optimizer = Adafactor(
            (value for value in model.parameters() if value.requires_grad),
            lr=base_lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = AdamW(
            (value for value in model.parameters() if value.requires_grad),
            lr=base_lr,
            weight_decay=args.weight_decay,
        )
    state = {"consumed_qas": 0, "optimizer_steps": 0, "best_eval_loss": None}
    if args.resume:
        state = load_checkpoint(Path(args.resume), model, optimizer)
    if int(state["consumed_qas"]) < args.schedule_start_qas:
        raise ValueError(
            "Resume exposure precedes --schedule-start-qas: "
            f"{state['consumed_qas']} < {args.schedule_start_qas}"
        )
    set_phase_lr(
        optimizer,
        base_lr,
        int(state["consumed_qas"]),
        args.schedule_start_qas,
        args.target_qas,
        args.warmup_ratio,
    )
    append_jsonl(events, {
        "event": "start",
        "mode": args.mode,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_data_sha256": data_ready["train"]["sha256"],
        "target_qas": args.target_qas,
        "schedule_start_qas": args.schedule_start_qas,
        "resumed_consumed_qas": int(state["consumed_qas"]),
        "phase_target_qas": args.target_qas - args.schedule_start_qas,
        "resume_checkpoint": str(Path(args.resume).resolve()) if args.resume else None,
        "train_token_budget": args.train_token_budget,
        "base_lr": base_lr,
        **cuda_stats(),
    })
    run_config = {
        "format": "repoqa_sft_run_v1",
        "args": vars(args),
        "model_revision": MODEL_REVISION,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_data_sha256": data_ready["train"]["sha256"],
        "validation_data_sha256": data_ready["validation"]["sha256"],
        "loss": "answer_token_mean_ce_weighted_by_qa_family",
        "qa_loss_weights": {"ast": 1.0, "llm": 1.8},
    }
    (output / "RUN_CONFIG.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    )
    checkpoints = sorted({int(value) for value in args.checkpoint_qas.split(",") if value})
    pending = [value for value in checkpoints if value > int(state["consumed_qas"])]
    model.train()
    started = time.time()
    cursor = int(state["consumed_qas"])
    start_cursor = cursor
    if args.preflight and args.preflight_profile == "memory":
        offsets = train_reader.longest_offsets(max(5, args.max_steps or 5))
        training_rows = iter(
            [row for offset in offsets for row in train_reader.read_slice(offset)]
        )
        training_batches = batch_rows(training_rows, args.train_token_budget)
        effective_max_steps = args.max_steps
    elif args.preflight:
        def quantile_batches():
            steps_per_quantile = max(1, args.max_steps or 5)
            for quantile in (0.1, 0.3, 0.5, 0.7, 0.9):
                offset = int((train_reader.total_rows - 1) * quantile)
                rows = itertools.islice(train_reader.iter_rows(offset), 2048)
                for local_step, values in enumerate(
                    batch_rows(rows, args.train_token_budget)
                ):
                    if local_step >= steps_per_quantile:
                        break
                    yield values

        training_batches = quantile_batches()
        effective_max_steps = 5 * max(1, args.max_steps or 5)
    else:
        training_rows = train_reader.iter_rows(cursor)
        training_batches = batch_rows(training_rows, args.train_token_budget)
        effective_max_steps = args.max_steps
    for rows in training_batches:
        if cursor + len(rows) > args.target_qas:
            rows = rows[: args.target_qas - cursor]
        if not rows:
            break
        batch = collate_qa_rows(rows, tokenizer.pad_token_id, device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = weighted_answer_ce(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {args.mode} loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [value for value in model.parameters() if value.requires_grad],
            args.max_grad_norm,
        )
        optimizer.step()
        cursor += batch.qa_count
        state["consumed_qas"] = cursor
        state["optimizer_steps"] = int(state["optimizer_steps"]) + 1
        state["last_loss"] = float(loss.detach().item())
        state["last_grad_norm"] = float(grad_norm)
        state["qas_per_second"] = (cursor - start_cursor) / max(
            time.time() - started, 1e-6
        )
        lr = set_phase_lr(
            optimizer,
            base_lr,
            cursor,
            args.schedule_start_qas,
            args.target_qas,
            args.warmup_ratio,
        )
        if state["optimizer_steps"] % args.log_every == 0:
            elapsed = time.time() - started
            append_jsonl(events, {
                "event": "train",
                "mode": args.mode,
                "consumed_qas": cursor,
                "optimizer_steps": state["optimizer_steps"],
                "loss": float(loss.detach().item()),
                "grad_norm": float(grad_norm),
                "learning_rate": lr,
                "qas_per_second": state["qas_per_second"],
                **metrics,
                **cuda_stats(),
            })
        del loss, batch
        while pending and cursor >= pending[0]:
            milestone = pending.pop(0)
            validation = evaluate(model, val_reader, tokenizer, device, args.eval_token_budget)
            state["last_validation"] = validation
            prior = state.get("best_eval_loss")
            is_best = prior is None or validation["eval_loss"] < prior
            if is_best:
                state["best_eval_loss"] = validation["eval_loss"]
                state["best_consumed_qas"] = cursor
            checkpoint = save_checkpoint(output, model, optimizer, state)
            if is_best:
                (output / "best_checkpoint.txt").write_text(str(checkpoint) + "\n")
            append_jsonl(events, {
                "event": "checkpoint",
                "requested_milestone": milestone,
                "checkpoint": str(checkpoint),
                "consumed_qas": cursor,
                **validation,
                **cuda_stats(),
            })
            model.train()
            gc.collect()
            torch.cuda.empty_cache()
        if cursor >= args.target_qas:
            break
        if effective_max_steps and state["optimizer_steps"] >= effective_max_steps:
            break
    state["completed"] = cursor >= args.target_qas
    state["elapsed_seconds"] = time.time() - started
    if args.preflight:
        result = {"event": "preflight_stop", **state, **cuda_stats()}
        (output / "PREFLIGHT.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        append_jsonl(events, result)
    else:
        final = save_checkpoint(output, model, optimizer, state)
        append_jsonl(events, {"event": "complete" if state["completed"] else "stopped", "checkpoint": str(final), **state, **cuda_stats()})


if __name__ == "__main__":
    main()
