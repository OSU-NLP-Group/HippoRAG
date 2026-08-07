#!/usr/bin/env python3
"""Train a static Code2LoRA hypernetwork for prompt-target tasks.

Input is the merged issue-fix table planned for SWE-Fixer:

```text
repo_state_embedding -> generated Gemma LoRA -> issue prompt -> gold patch
```

This trainer is intentionally static-only: no commit GRU, no diff embeddings,
and no executable task environments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from repotune_issuefix.code2lora_gemma import (
    DEFAULT_GEMMA4_TEXT_REGEX,
    DEFAULT_TARGET_MODULES,
    Code2LoRAHead,
    count_head_parameters,
    discover_module_types_and_dims,
    get_module_specs,
    inject_lora_weights,
    load_gemma4_model,
    replace_with_lora,
    summarize_specs,
    torch_dtype,
)
from repotune_issuefix.issuefix_dataset import load_issuefix_rows, tokenize_prompt_target_batch


DEFAULT_MODEL = "google/gemma-4-E2B-it"


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def git_revision(repo_dir: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def cuda_memory() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
        "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
        "cuda_max_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "cuda_max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }


def row_target(row) -> str:
    return row.target_text


def row_token_lengths(tokenizer, prompt: str, target: str) -> Tuple[int, int, int]:
    eos = tokenizer.eos_token or ""
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    target_tokens = len(tokenizer(target + eos, add_special_tokens=False)["input_ids"])
    return prompt_tokens, target_tokens, prompt_tokens + target_tokens


def lora_tensor_stats(head_out: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
    vals = []
    max_abs = 0.0
    for family in ("A", "B"):
        for tensor in head_out[family].values():
            t = tensor.detach().float()
            vals.append(float(t.norm().item()))
            max_abs = max(max_abs, float(t.abs().max().item()))
    if not vals:
        return {}
    return {
        "generated_lora_tensor_norm_mean": sum(vals) / len(vals),
        "generated_lora_tensor_norm_max": max(vals),
        "generated_lora_abs_max": max_abs,
    }


def scale_stats(head: Code2LoRAHead) -> Dict[str, float]:
    vals = []
    for param_dict in (head.log_scale_A, head.log_scale_B):
        for p in param_dict.values():
            vals.append(float(torch.exp(p.detach()).item()))
    if not vals:
        return {}
    return {
        "lora_scale_min": min(vals),
        "lora_scale_mean": sum(vals) / len(vals),
        "lora_scale_max": max(vals),
    }


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def stable_hash(text: str) -> int:
    value = 0
    for byte in text.encode("utf-8"):
        value = ((value * 131) + byte) & 0xFFFFFFFF
    return value


def select_eval_rows(rows, *, limit: int = 0, strategy: str = "first", seed: int = 3407):
    if not limit or limit >= len(rows):
        return list(rows)
    if strategy == "first":
        return list(rows[:limit])
    if strategy == "shuffle":
        selected = list(rows)
        random.Random(seed).shuffle(selected)
        return selected[:limit]
    if strategy == "round_robin_repo":
        by_repo: Dict[str, List[Any]] = defaultdict(list)
        for row in rows:
            by_repo[row.repo_id].append(row)
        repo_order = sorted(by_repo)
        random.Random(seed).shuffle(repo_order)
        queues: Dict[str, deque[Any]] = {}
        for repo_id in repo_order:
            repo_rows = list(by_repo[repo_id])
            random.Random(seed + stable_hash(repo_id)).shuffle(repo_rows)
            queues[repo_id] = deque(repo_rows)
        selected = []
        active = deque(repo_order)
        while active and len(selected) < limit:
            repo_id = active.popleft()
            queue = queues[repo_id]
            if queue:
                selected.append(queue.popleft())
            if queue:
                active.append(repo_id)
        return selected
    raise ValueError(f"unknown eval_selection: {strategy}")


def build_train_order(rows, *, strategy: str = "global_shuffle") -> List[int]:
    if strategy == "global_shuffle":
        order = list(range(len(rows)))
        random.shuffle(order)
        return order
    if strategy == "round_robin_repo":
        by_repo: Dict[str, List[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            by_repo[row.repo_id].append(idx)
        repo_order = sorted(by_repo)
        random.shuffle(repo_order)
        queues: Dict[str, deque[int]] = {}
        for repo_id in repo_order:
            indices = list(by_repo[repo_id])
            random.shuffle(indices)
            queues[repo_id] = deque(indices)
        order: List[int] = []
        active = deque(repo_order)
        while active:
            repo_id = active.popleft()
            queue = queues[repo_id]
            if queue:
                order.append(queue.popleft())
            if queue:
                active.append(repo_id)
        return order
    raise ValueError(f"unknown train_order: {strategy}")


@torch.no_grad()
def evaluate(
    base_model,
    head,
    specs,
    tokenizer,
    rows,
    *,
    device,
    max_seq_len,
    limit: int = 0,
    embedding_mode: str = "correct",
    seed: int = 3407,
) -> Dict[str, Any]:
    head.eval()
    base_model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_prompt_tokens = 0
    rows_scored = 0
    rows_skipped = 0
    buckets: Dict[str, Dict[str, float]] = {}
    kept_rows = rows[:limit] if limit else rows
    shuffled_embeddings = []
    if embedding_mode == "shuffled" and kept_rows:
        shuffled_embeddings = [row.repo_state_embedding for row in kept_rows]
        random.Random(seed).shuffle(shuffled_embeddings)
    prompt_lens: List[int] = []
    target_lens: List[int] = []
    for row_index, row in enumerate(kept_rows):
        target = row_target(row)
        prompt_tokens, target_tokens, total_len = row_token_lengths(tokenizer, row.prompt, target)
        batch = tokenize_prompt_target_batch(tokenizer, [row.prompt], [target], max_seq_len=max_seq_len)
        if not batch:
            rows_skipped += 1
            continue
        if embedding_mode == "base":
            zero_ctx = torch.zeros_like(torch.from_numpy(row.repo_state_embedding)).to(device).unsqueeze(0)
            head_out = head(zero_ctx)
            for family in ("A", "B"):
                for key in head_out[family]:
                    head_out[family][key].zero_()
        elif embedding_mode == "zero":
            ctx = torch.zeros_like(torch.from_numpy(row.repo_state_embedding)).to(device).unsqueeze(0)
            head_out = head(ctx)
        elif embedding_mode == "shuffled":
            ctx = torch.from_numpy(shuffled_embeddings[row_index]).to(device).unsqueeze(0)
            head_out = head(ctx)
        elif embedding_mode == "correct":
            ctx = torch.from_numpy(row.repo_state_embedding).to(device).unsqueeze(0)
            head_out = head(ctx)
        else:
            raise ValueError(f"unknown embedding_mode: {embedding_mode}")
        inject_lora_weights(base_model, specs, head_out, batch_index=0)
        batch = {key: value.to(device) for key, value in batch.items()}
        out = base_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = causal_lm_loss(out.logits, batch["labels"])
        n_tokens = int((batch["labels"] != -100).sum().item())
        total_loss += float(loss.item()) * n_tokens
        total_tokens += n_tokens
        total_prompt_tokens += prompt_tokens
        prompt_lens.append(prompt_tokens)
        target_lens.append(target_tokens)
        rows_scored += 1
        if total_len <= 1024:
            bucket = "total_le_1024"
        elif total_len <= 2048:
            bucket = "total_1025_2048"
        elif total_len <= 4096:
            bucket = "total_2049_4096"
        else:
            bucket = "total_gt_4096"
        b = buckets.setdefault(bucket, {"loss_sum": 0.0, "tokens": 0.0, "rows": 0.0})
        b["loss_sum"] += float(loss.item()) * n_tokens
        b["tokens"] += float(n_tokens)
        b["rows"] += 1.0
    bucket_out = {
        key: {
            "loss": value["loss_sum"] / max(value["tokens"], 1.0),
            "tokens": value["tokens"],
            "rows": value["rows"],
        }
        for key, value in sorted(buckets.items())
    }
    prompt_arr = np.asarray(prompt_lens, dtype=np.float32)
    target_arr = np.asarray(target_lens, dtype=np.float32)
    return {
        "loss": total_loss / max(total_tokens, 1),
        "embedding_mode": embedding_mode,
        "tokens": float(total_tokens),
        "prompt_tokens": float(total_prompt_tokens),
        "prompt_tokens_mean": float(prompt_arr.mean()) if prompt_arr.size else 0.0,
        "prompt_tokens_p95": float(np.percentile(prompt_arr, 95)) if prompt_arr.size else 0.0,
        "target_tokens_mean": float(target_arr.mean()) if target_arr.size else 0.0,
        "target_tokens_p95": float(np.percentile(target_arr, 95)) if target_arr.size else 0.0,
        "rows": float(len(kept_rows)),
        "rows_scored": float(rows_scored),
        "rows_skipped": float(rows_skipped),
        "buckets": bucket_out,
    }


def save_checkpoint(
    out_dir: Path,
    name: str,
    head: Code2LoRAHead,
    args,
    type_dims: Dict[str, Any],
    specs_summary: Dict[str, Any],
    *,
    optimizer=None,
    scheduler=None,
    trainer_state: Dict[str, Any] | None = None,
    include_rng: bool = True,
) -> Path:
    path = out_dir / f"head.{name}.pt"
    payload = {
        "state_dict": head.state_dict(),
        "config": head.config_dict(),
        "type_dims": type_dims,
        "specs_summary": specs_summary,
        "args": vars(args),
        "trainer_state": trainer_state or {},
    }
    if include_rng:
        payload["rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)
    return path


def load_resume_checkpoint(path: str, *, device) -> Dict[str, Any]:
    if not path:
        return {}
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=device)


def restore_rng_state(rng_state: Dict[str, Any]) -> None:
    if not rng_state:
        return
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda"):
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", required=True, help="Merged issue-fix train parquet/jsonl.")
    parser.add_argument("--val-data", default="", help="Optional merged issue-fix validation parquet/jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-name", default="issuefix", help="Task label written to metrics/config.")
    parser.add_argument("--target-column", default="target_text", help="Preferred target column; falls back to target_patch.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--module-name-regex", default=DEFAULT_GEMMA4_TEXT_REGEX)
    parser.add_argument("--no-shape-aware-types", dest="shape_aware_types", action="store_false")
    parser.set_defaults(shape_aware_types=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--head-hidden-dim", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-base-loss", action="store_true")
    parser.add_argument("--eval-shuffled-embedding", action="store_true")
    parser.add_argument(
        "--eval-selection",
        choices=["first", "shuffle", "round_robin_repo"],
        default="first",
        help="How to select the limit-val-rows eval subset from the loaded validation table.",
    )
    parser.add_argument("--generation-eval-every", type=int, default=0, help="Reserved for short generation metrics; 0 disables.")
    parser.add_argument("--save-every", type=int, default=1000, help="Overwrite full resumable latest checkpoint every N train steps.")
    parser.add_argument("--archive-every", type=int, default=10000, help="Write lightweight archival head.step*.pt every N train steps.")
    parser.add_argument("--limit-train-rows", type=int, default=0)
    parser.add_argument("--limit-val-rows", type=int, default=200)
    parser.add_argument(
        "--train-order",
        choices=["global_shuffle", "round_robin_repo"],
        default="global_shuffle",
        help="How to order training rows each epoch. round_robin_repo gives early checkpoints broader repo coverage.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from", default="", help="Optional head checkpoint to resume full trainer state from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype = torch_dtype(args.dtype)

    train_rows = load_issuefix_rows(Path(args.train_data), limit=args.limit_train_rows, target_column=args.target_column)
    val_load_limit = args.limit_val_rows if args.eval_selection == "first" else 0
    val_rows = load_issuefix_rows(Path(args.val_data), limit=val_load_limit, target_column=args.target_column) if args.val_data else []
    if not train_rows:
        raise RuntimeError("No train rows loaded.")
    input_dim = int(train_rows[0].repo_state_embedding.shape[0])

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = load_gemma4_model(
        args.model_name,
        dtype=dtype,
        device=str(device),
        local_files_only=args.local_files_only,
    )
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    if args.gradient_checkpointing:
        base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    specs = get_module_specs(
        base_model,
        args.target_modules,
        module_name_regex=args.module_name_regex,
        shape_aware_types=args.shape_aware_types,
    )
    if not specs:
        raise RuntimeError("No LoRA target modules found.")
    type_dims = discover_module_types_and_dims(specs)
    specs_summary = summarize_specs(specs)
    head_param_count = count_head_parameters(input_dim, type_dims, hidden_dim=args.head_hidden_dim, rank=args.rank)
    print(json.dumps({
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "input_dim": input_dim,
        "specs": specs_summary,
        "head_params": head_param_count,
    }, indent=2), flush=True)

    replace_with_lora(base_model, specs, rank=args.rank, alpha=args.alpha)
    head = Code2LoRAHead(
        input_dim=input_dim,
        type_dims=type_dims,
        hidden_dim=args.head_hidden_dim,
        rank=args.rank,
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_rows))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    resume_ckpt = load_resume_checkpoint(args.resume_from, device=device)
    resume_state = resume_ckpt.get("trainer_state", {}) if resume_ckpt else {}
    start_epoch = int(resume_state.get("epoch", 0))
    start_iter = int(resume_state.get("next_iter", 0))
    best_val = float(resume_state.get("best_val", float("inf")))
    global_step = int(resume_state.get("global_step", 0))
    examples_seen = int(resume_state.get("examples_seen", 0))
    examples_skipped = int(resume_state.get("examples_skipped", 0))
    tokens_seen = int(resume_state.get("tokens_seen", 0))
    target_tokens_seen = int(resume_state.get("target_tokens_seen", 0))
    if resume_ckpt:
        head.load_state_dict(resume_ckpt["state_dict"])
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        restore_rng_state(resume_ckpt.get("rng_state", {}))

    metrics_path = out_dir / "metrics.jsonl"
    events_path = out_dir / "events.jsonl"
    run_config = {
        "event": "run_config",
        "args": vars(args),
        "task_name": args.task_name,
        "train_data": str(Path(args.train_data).resolve()),
        "val_data": str(Path(args.val_data).resolve()) if args.val_data else "",
        "output_dir": str(out_dir.resolve()),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "git_revision": git_revision(Path(__file__).resolve().parents[2]),
        "device": str(device),
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "started_unix": time.time(),
        "resume_from": args.resume_from,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")
    append_jsonl(events_path, run_config)
    model_summary = {
        "event": "model_summary",
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "input_dim": input_dim,
        "target_modules": args.target_modules,
        "module_name_regex": args.module_name_regex,
        "shape_aware_types": args.shape_aware_types,
        "rank": args.rank,
        "alpha": args.alpha,
        "head_hidden_dim": args.head_hidden_dim,
        "head_params": head_param_count,
        "specs": specs_summary,
        "type_dims": type_dims,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "max_seq_len": args.max_seq_len,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "train_order": args.train_order,
        "train_repos": len({row.repo_id for row in train_rows}),
        **cuda_memory(),
    }
    append_jsonl(events_path, model_summary)
    append_jsonl(metrics_path, model_summary)
    if resume_ckpt:
        append_jsonl(events_path, {
            "event": "resume",
            "resume_from": args.resume_from,
            "start_epoch": start_epoch,
            "start_iter": start_iter,
            "global_step": global_step,
            "examples_seen": examples_seen,
            "examples_skipped": examples_skipped,
            "tokens_seen": tokens_seen,
            "target_tokens_seen": target_tokens_seen,
            "best_val": best_val,
            **cuda_memory(),
        })
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        if resume_ckpt and epoch == start_epoch and resume_state.get("epoch_order"):
            order = list(resume_state["epoch_order"])
        else:
            order = build_train_order(train_rows, strategy=args.train_order)
        head.train()
        for it, idx in enumerate(order):
            if epoch == start_epoch and it < start_iter:
                continue
            row = train_rows[idx]
            target = row_target(row)
            prompt_tokens, target_tokens, total_tokens = row_token_lengths(tokenizer, row.prompt, target)
            batch = tokenize_prompt_target_batch(tokenizer, [row.prompt], [target], max_seq_len=args.max_seq_len)
            if not batch:
                examples_skipped += 1
                if examples_skipped <= 20 or examples_skipped % max(1, args.log_every) == 0:
                    append_jsonl(events_path, {
                        "event": "train_skip",
                        "epoch": epoch,
                        "iter": it,
                        "instance_id": row.instance_id,
                        "repo_id": row.repo_id,
                        "prompt_tokens": prompt_tokens,
                        "target_tokens": target_tokens,
                        "total_tokens": total_tokens,
                        "max_seq_len": args.max_seq_len,
                        "reason": "overlength_or_empty",
                    })
                continue
            ctx = torch.from_numpy(row.repo_state_embedding).to(device).unsqueeze(0)
            head_out = head(ctx)
            inject_lora_weights(base_model, specs, head_out, batch_index=0)
            batch = {key: value.to(device) for key, value in batch.items()}
            out = base_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = causal_lm_loss(out.logits, batch["labels"])
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                append_jsonl(events_path, {
                    "event": "nonfinite_loss",
                    "step": global_step,
                    "epoch": epoch,
                    "iter": it,
                    "loss": loss_value,
                    **cuda_memory(),
                })
                raise RuntimeError(f"Non-finite loss at step {global_step}: {loss_value}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            examples_seen += 1
            tokens_seen += total_tokens
            target_tokens_seen += target_tokens

            if global_step % max(1, args.log_every) == 0:
                elapsed = time.time() - started
                steps_per_sec = global_step / max(elapsed, 1e-9)
                remaining_steps = max(0, total_steps - global_step)
                row_metrics = {
                    "event": "train",
                    "step": global_step,
                    "epoch": epoch,
                    "iter": it,
                    "instance_id": row.instance_id,
                    "repo_id": row.repo_id,
                    "train_loss": loss_value,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "prompt_tokens": prompt_tokens,
                    "target_tokens": target_tokens,
                    "total_tokens": total_tokens,
                    "tokens_seen": tokens_seen,
                    "target_tokens_seen": target_tokens_seen,
                    "examples_seen": examples_seen,
                    "examples_skipped": examples_skipped,
                    "skip_pct_seen": 100.0 * examples_skipped / max(1, examples_seen + examples_skipped),
                    "grad_norm_pre_clip": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
                    "elapsed_min": round(elapsed / 60.0, 3),
                    "steps_per_sec": steps_per_sec,
                    "tokens_per_sec": tokens_seen / max(elapsed, 1e-9),
                    "target_tokens_per_sec": target_tokens_seen / max(elapsed, 1e-9),
                    "eta_hours": remaining_steps / max(steps_per_sec, 1e-9) / 3600.0,
                    **lora_tensor_stats(head_out),
                    **scale_stats(head),
                    **cuda_memory(),
                }
                print(json.dumps(row_metrics), flush=True)
                append_jsonl(metrics_path, row_metrics)

            if val_rows and args.eval_every and global_step % args.eval_every == 0:
                eval_rows = select_eval_rows(
                    val_rows,
                    limit=args.limit_val_rows,
                    strategy=args.eval_selection,
                    seed=args.seed + global_step,
                )
                val = evaluate(
                    base_model,
                    head,
                    specs,
                    tokenizer,
                    eval_rows,
                    device=device,
                    max_seq_len=args.max_seq_len,
                    limit=0,
                    embedding_mode="correct",
                    seed=args.seed,
                )
                evals: Dict[str, Any] = {"correct_embedding": val}
                if args.eval_base_loss:
                    evals["base_model"] = evaluate(
                        base_model,
                        head,
                        specs,
                        tokenizer,
                        eval_rows,
                        device=device,
                        max_seq_len=args.max_seq_len,
                        limit=0,
                        embedding_mode="base",
                        seed=args.seed,
                    )
                    evals["zero_embedding"] = evaluate(
                        base_model,
                        head,
                        specs,
                        tokenizer,
                        eval_rows,
                        device=device,
                        max_seq_len=args.max_seq_len,
                        limit=0,
                        embedding_mode="zero",
                        seed=args.seed,
                    )
                if args.eval_shuffled_embedding:
                    evals["shuffled_embedding"] = evaluate(
                        base_model,
                        head,
                        specs,
                        tokenizer,
                        eval_rows,
                        device=device,
                        max_seq_len=args.max_seq_len,
                        limit=0,
                        embedding_mode="shuffled",
                        seed=args.seed + global_step,
                    )
                if "base_model" in evals:
                    evals["repo_embedding_gain"] = evals["base_model"]["loss"] - val["loss"]
                if "shuffled_embedding" in evals:
                    evals["repo_sensitivity_gap"] = evals["shuffled_embedding"]["loss"] - val["loss"]
                row_metrics = {
                    "event": "eval",
                    "step": global_step,
                    "epoch": epoch,
                    "val": val,
                    "evals": evals,
                    **cuda_memory(),
                }
                print(json.dumps(row_metrics), flush=True)
                append_jsonl(metrics_path, row_metrics)
                if val["loss"] < best_val:
                    best_val = val["loss"]
                    save_checkpoint(
                        out_dir,
                        "best",
                        head,
                        args,
                        type_dims,
                        specs_summary,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        trainer_state={
                            "epoch": epoch,
                            "next_iter": it + 1,
                            "epoch_order": order,
                            "global_step": global_step,
                            "examples_seen": examples_seen,
                            "examples_skipped": examples_skipped,
                            "tokens_seen": tokens_seen,
                            "target_tokens_seen": target_tokens_seen,
                            "best_val": best_val,
                        },
                    )
            if args.save_every and global_step % args.save_every == 0:
                trainer_state = {
                    "epoch": epoch,
                    "next_iter": it + 1,
                    "epoch_order": order,
                    "global_step": global_step,
                    "examples_seen": examples_seen,
                    "examples_skipped": examples_skipped,
                    "tokens_seen": tokens_seen,
                    "target_tokens_seen": target_tokens_seen,
                    "best_val": best_val,
                }
                path = save_checkpoint(
                    out_dir,
                    "latest",
                    head,
                    args,
                    type_dims,
                    specs_summary,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    trainer_state=trainer_state,
                )
                append_jsonl(events_path, {
                    "event": "checkpoint",
                    "step": global_step,
                    "epoch": epoch,
                    "path": str(path),
                    "checkpoint_kind": "latest_full_resume",
                    **cuda_memory(),
                })
                if args.archive_every and global_step % args.archive_every == 0:
                    archive_path = save_checkpoint(
                        out_dir,
                        f"step{global_step}",
                        head,
                        args,
                        type_dims,
                        specs_summary,
                        trainer_state={"global_step": global_step, "epoch": epoch, "next_iter": it + 1},
                        include_rng=False,
                    )
                    append_jsonl(events_path, {
                        "event": "checkpoint_archive",
                        "step": global_step,
                        "epoch": epoch,
                        "path": str(archive_path),
                        "checkpoint_kind": "head_only_archive",
                        **cuda_memory(),
                    })

        trainer_state = {
            "epoch": epoch + 1,
            "next_iter": 0,
            "epoch_order": [],
            "global_step": global_step,
            "examples_seen": examples_seen,
            "examples_skipped": examples_skipped,
            "tokens_seen": tokens_seen,
            "target_tokens_seen": target_tokens_seen,
            "best_val": best_val,
        }
        ep_path = save_checkpoint(
            out_dir,
            f"ep{epoch}",
            head,
            args,
            type_dims,
            specs_summary,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
        )
        latest_path = save_checkpoint(
            out_dir,
            "latest",
            head,
            args,
            type_dims,
            specs_summary,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
        )
        append_jsonl(events_path, {
            "event": "epoch_checkpoint",
            "epoch": epoch,
            "step": global_step,
            "path": str(ep_path),
            "latest_path": str(latest_path),
            "examples_seen": examples_seen,
            "examples_skipped": examples_skipped,
            **cuda_memory(),
        })

    print(f"done: {out_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
