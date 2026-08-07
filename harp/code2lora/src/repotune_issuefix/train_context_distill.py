#!/usr/bin/env python3
"""Train Code2LoRA with context-aware teacher and context-free student prompts."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import random
import resource
import shutil
import socket
import statistics
import sys
import time
from collections import Counter, OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
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
from repotune_issuefix.context_distill_data import tokenize_prompt_target_batch
from repotune_issuefix.train_static import (
    append_jsonl,
    cuda_memory,
    git_revision,
    load_resume_checkpoint,
    lora_tensor_stats,
    restore_rng_state,
    save_checkpoint,
    scale_stats,
)

DEFAULT_TASK_TYPES = ("ntp", "qa", "ntp_dense", "issue_fix")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-data", type=Path, required=True)
    p.add_argument("--val-data", type=Path, required=True)
    p.add_argument("--repo-embeddings", type=Path,
                   help="Optional side table with repo_id/base_commit/repo_state_embedding. "
                        "When set, train/val rows may omit repo_state_embedding.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-name", default="google/gemma-4-E2B-it")
    p.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    p.add_argument("--target-shape-types", nargs="*", default=[],
                   help="Optional exact shape-aware target types, e.g. down_proj__in12288__out1536.")
    p.add_argument("--module-name-regex", default=DEFAULT_GEMMA4_TEXT_REGEX)
    p.add_argument("--no-shape-aware-types", dest="shape_aware_types", action="store_false")
    p.set_defaults(shape_aware_types=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--head-input-dim", type=int, default=0,
                   help="Optional guard for repo_state_embedding dimension; 0 infers from data.")
    p.add_argument("--head-hidden-dim", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=1000000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--ntp-ratio", type=float, default=0.60)
    p.add_argument("--task-ratios", default="",
                   help="Comma-separated task sampling ratios, e.g. ntp_dense=0.8,qa=0.15,issue_fix=0.05. "
                        "Defaults to --ntp-ratio for ntp/qa.")
    p.add_argument("--qa-subtype-temperature", type=float, default=2.0,
                   help="Temperature for QA subtype sampling; 1 follows counts and larger values flatten them.")
    p.add_argument("--kl-source", choices=("context_teacher", "base"), default="context_teacher",
                   help="context_teacher uses teacher_prompt; base uses the no-LoRA model on student_prompt.")
    p.add_argument("--lambda-kl", type=float, default=1.0)
    p.add_argument("--lambda-base-kl", type=float, default=None,
                   help="Alias for --lambda-kl when --kl-source=base.")
    p.add_argument("--lambda-ce-ntp", type=float, default=1.0)
    p.add_argument("--lambda-ce-qa", type=float, default=1.0)
    p.add_argument("--lambda-ce-default", type=float, default=1.0)
    p.add_argument("--lambda-ce-task", action="append", default=[],
                   help="Per-task CE weights as task=value; may be repeated.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--teacher-top-k", type=int, default=0,
                   help="0 uses full-vocabulary KL; positive uses conditional KL over teacher top-k.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--gradient-diagnostics-every", type=int, default=2000,
                   help="Measure NTP/QA gradient cosine this often; 0 disables the extra forwards.")
    p.add_argument("--gradient-diagnostic-elements", type=int, default=262144,
                   help="Maximum deterministic gradient coordinates retained per task for cosine estimation.")
    p.add_argument("--row-group-cache-size", type=int, default=1024,
                   help="Maximum full Parquet row groups cached as Arrow tables (training data only).")
    p.add_argument("--limit-val-rows", type=int, default=200)
    p.add_argument("--min-train-repos", type=int, default=650,
                   help="Coverage guard: refuse to train on an accidentally truncated table.")
    p.add_argument("--min-val-repos", type=int, default=30)
    p.add_argument("--qa-generation-rows", type=int, default=48)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--archive-every", type=int, default=50000)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--resume-from", default="")
    return p.parse_args()


class RepoEmbeddingLookup:
    """Small side-table lookup for large datasets that do not duplicate embeddings."""

    def __init__(self, path: Path):
        parquet = pq.ParquetFile(path)
        table = pq.read_table(path, columns=[name for name in ("repo_id", "base_commit", "commit_sha",
                                                               "repo_state_embedding")
                                             if name in parquet.schema_arrow.names])
        names = set(table.column_names)
        missing = {"repo_id", "repo_state_embedding"} - names
        if missing:
            raise ValueError(f"{path} missing embedding columns: {sorted(missing)}")
        cols = {name: table.column(name).to_pylist() for name in table.column_names}
        commits = cols.get("base_commit") or cols.get("commit_sha") or [""] * table.num_rows
        self.path = path
        self.by_key: dict[tuple[str, str], np.ndarray] = {}
        self.by_repo: dict[str, np.ndarray] = {}
        for i in range(table.num_rows):
            repo = str(cols["repo_id"][i])
            commit = str(commits[i] or "")
            vec = np.asarray(cols["repo_state_embedding"][i], dtype=np.float32)
            if commit:
                self.by_key[(repo, commit)] = vec
            self.by_repo.setdefault(repo, vec)
        if not self.by_repo:
            raise ValueError(f"{path} did not contain any repo embeddings")

    def hydrate(self, row: dict[str, Any]) -> dict[str, Any]:
        if "repo_state_embedding" in row and row["repo_state_embedding"] is not None:
            row["repo_state_embedding"] = np.asarray(row["repo_state_embedding"], dtype=np.float32)
            return row
        repo = str(row.get("repo_id", ""))
        commit = str(row.get("base_commit") or row.get("commit_sha") or "")
        vec = self.by_key.get((repo, commit))
        if vec is None:
            vec = self.by_repo.get(repo)
        if vec is None:
            raise KeyError(f"no repo embedding for repo={repo!r} commit={commit!r}")
        row["repo_state_embedding"] = vec
        return row

    def manifest(self) -> dict[str, Any]:
        dims = sorted({int(vec.shape[0]) for vec in self.by_repo.values()})
        return {"path": str(self.path), "repos": len(self.by_repo),
                "repo_commit_keys": len(self.by_key), "embedding_dims": dims}


def load_rows(path: Path, embedding_lookup: RepoEmbeddingLookup | None = None) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    required = {"repo_id", "instance_id", "student_prompt", "teacher_prompt", "target_text",
                "target_terminator", "task_type", "example_weight"}
    if embedding_lookup is None:
        required.add("repo_state_embedding")
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    for row in rows:
        if embedding_lookup is not None:
            embedding_lookup.hydrate(row)
        else:
            row["repo_state_embedding"] = np.asarray(row["repo_state_embedding"], dtype=np.float32)
        if row["task_type"] not in DEFAULT_TASK_TYPES:
            raise ValueError(f"invalid task_type: {row['task_type']}")
        if not row.get("target_terminator"):
            raise ValueError(f"row {row['instance_id']} has no target_terminator")
    return rows


def parse_task_ratios(text: str, *, ntp_ratio: float) -> dict[str, float]:
    if not text:
        ratios = {"ntp": float(ntp_ratio), "qa": 1.0 - float(ntp_ratio)}
    else:
        ratios = {}
        for part in text.split(","):
            if not part.strip():
                continue
            name, sep, value = part.partition("=")
            if not sep:
                raise ValueError(f"invalid task ratio entry {part!r}; expected task=value")
            ratios[name.strip()] = float(value)
    total = sum(value for value in ratios.values() if value > 0)
    if total <= 0:
        raise ValueError("task ratios must contain at least one positive value")
    return {task: value / total for task, value in ratios.items() if value > 0}


def parse_task_weights(entries: list[str], args: argparse.Namespace) -> dict[str, float]:
    weights = {"ntp": float(args.lambda_ce_ntp), "qa": float(args.lambda_ce_qa)}
    for task in DEFAULT_TASK_TYPES:
        weights.setdefault(task, float(args.lambda_ce_default))
    for entry in entries:
        name, sep, value = entry.partition("=")
        if not sep:
            raise ValueError(f"invalid CE task weight {entry!r}; expected task=value")
        weights[name.strip()] = float(value)
    return weights


def target_shape_group_summary(specs: list[Any], type_dims: dict[str, tuple[int, int]],
                               *, hidden_dim: int, rank: int) -> dict[str, Any]:
    """Summarize selected shape-aware targets and generated LoRA sizes."""
    counts: Counter[str] = Counter(str(sp.type) for sp in specs)
    base_types: dict[str, str] = {}
    layers_by_type: defaultdict[str, set[int]] = defaultdict(set)
    examples_by_type: dict[str, str] = {}
    for sp in specs:
        type_name = str(sp.type)
        base_types[type_name] = str(getattr(sp, "base_type", type_name))
        layer_idx = int(getattr(sp, "layer_idx", -1))
        if layer_idx >= 0:
            layers_by_type[type_name].add(layer_idx)
        examples_by_type.setdefault(type_name, str(getattr(sp, "full_name", "")))

    groups: list[dict[str, Any]] = []
    generated_total = 0
    injected_total = 0
    head_total = 0
    for type_name in sorted(type_dims):
        in_f, out_f = (int(type_dims[type_name][0]), int(type_dims[type_name][1]))
        module_count = int(counts.get(type_name, 0))
        generated_scalars = int(rank * (in_f + out_f))
        head_params = int(
            hidden_dim * (rank * in_f) + (rank * in_f)
            + hidden_dim * (out_f * rank) + (out_f * rank)
            + 2
        )
        generated_total += generated_scalars
        injected_total += module_count * generated_scalars
        head_total += head_params
        groups.append({
            "type": type_name,
            "base_type": base_types.get(type_name, type_name.split("__", 1)[0]),
            "module_count": module_count,
            "in_features": in_f,
            "out_features": out_f,
            "generated_lora_scalars": generated_scalars,
            "injected_lora_scalars": module_count * generated_scalars,
            "head_params": head_params,
            "layer_indices": sorted(layers_by_type.get(type_name, set())),
            "example_module": examples_by_type.get(type_name, ""),
        })
    return {
        "groups": groups,
        "target_shape_types": [group["type"] for group in groups],
        "target_module_count": int(sum(counts.values())),
        "target_type_count": len(groups),
        "generated_lora_scalars_per_repo": generated_total,
        "injected_lora_scalars_per_repo": injected_total,
        "head_params_by_shape_total": head_total,
    }


def ce_weight_for_task(row: dict[str, Any], args: argparse.Namespace) -> float:
    weights = getattr(args, "lambda_ce_by_task", {})
    return float(weights.get(str(row["task_type"]), getattr(args, "lambda_ce_default", 1.0)))


def task_names_from_rows(rows: Any, fallback: tuple[str, ...] = DEFAULT_TASK_TYPES) -> list[str]:
    if hasattr(rows, "task_counts"):
        names = rows.task_counts.keys()
    else:
        names = (str(r["task_type"]) for r in rows)
    return sorted(name for name in set(names) if name in fallback or name)


class ParquetRowStore:
    """Bounded-memory random access to a Parquet table plus a lightweight sampling index."""

    def __init__(self, path: Path, *, cache_row_groups: int = 1024,
                 embedding_lookup: RepoEmbeddingLookup | None = None):
        self.path = path
        self.parquet = pq.ParquetFile(path)
        self.embedding_lookup = embedding_lookup
        self.num_rows = self.parquet.metadata.num_rows
        self.row_group_starts = [0]
        for i in range(self.parquet.metadata.num_row_groups):
            self.row_group_starts.append(self.row_group_starts[-1] + self.parquet.metadata.row_group(i).num_rows)
        names = set(self.parquet.schema_arrow.names)
        required = {"repo_id", "instance_id", "student_prompt", "teacher_prompt", "target_text",
                    "target_terminator", "task_type", "example_weight"}
        if embedding_lookup is None:
            required.add("repo_state_embedding")
        missing = required - names
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        self.subtype_column = next((name for name in ("task_subtype", "sample_type", "template_id")
                                    if name in names), None)
        metadata_fields = ("qa_source", "answer_style", "task_category", "target_format", "validation_status")
        metadata_columns = [name for name in metadata_fields if name in names]
        columns = ["repo_id", "task_type", "instance_id", "example_weight"]
        if self.subtype_column:
            columns.append(self.subtype_column)
        columns.extend(metadata_columns)
        self.records: list[tuple[str, str, str]] = []
        self.repo_counts: Counter[str] = Counter()
        self.task_counts: Counter[str] = Counter()
        self.subtype_counts: Counter[str] = Counter()
        self.metadata_counts: dict[str, Counter[str]] = {name: Counter() for name in metadata_fields}
        self.repos_by_task: defaultdict[str, set[str]] = defaultdict(set)
        self.weight_count = 0
        self.weight_sum = 0.0
        self.weight_min = float("inf")
        self.weight_max = float("-inf")
        instance_hash = hashlib.sha256()
        for batch in self.parquet.iter_batches(batch_size=65536, columns=columns):
            values = batch.to_pydict()
            subtypes = values.get(self.subtype_column) if self.subtype_column else None
            for i, (repo, task, instance_id, weight) in enumerate(zip(
                    values["repo_id"], values["task_type"], values["instance_id"], values["example_weight"])):
                repo, task = sys.intern(str(repo)), sys.intern(str(task))
                if task not in DEFAULT_TASK_TYPES:
                    raise ValueError(f"invalid task_type: {task}")
                subtype = sys.intern(str(subtypes[i] or task)) if subtypes else task
                self.records.append((repo, task, subtype))
                self.repo_counts[repo] += 1
                self.task_counts[task] += 1
                self.subtype_counts[subtype] += 1
                for column in metadata_columns:
                    value = values[column][i]
                    if value is not None and str(value):
                        self.metadata_counts[column][str(value)] += 1
                self.repos_by_task[task].add(repo)
                numeric_weight = float(weight)
                self.weight_count += 1
                self.weight_sum += numeric_weight
                self.weight_min = min(self.weight_min, numeric_weight)
                self.weight_max = max(self.weight_max, numeric_weight)
                instance_hash.update(str(instance_id).encode())
                instance_hash.update(b"\n")
        if len(self.records) != self.num_rows:
            raise RuntimeError(f"sampling index has {len(self.records)} rows, expected {self.num_rows}")
        self.instance_id_ordered_sha256 = instance_hash.hexdigest()
        self.cache_row_groups = max(1, cache_row_groups)
        self.cache: OrderedDict[int, Any] = OrderedDict()

    def sampling_records(self):
        for index, (repo, task, subtype) in enumerate(self.records):
            yield index, repo, task, subtype

    def get(self, index: int) -> dict[str, Any]:
        row_group = bisect.bisect_right(self.row_group_starts, index) - 1
        offset = index - self.row_group_starts[row_group]
        table = self.cache.pop(row_group, None)
        if table is None:
            table = self.parquet.read_row_group(row_group)
        self.cache[row_group] = table
        while len(self.cache) > self.cache_row_groups:
            self.cache.popitem(last=False)
        row = table.slice(offset, 1).to_pylist()[0]
        if self.embedding_lookup is not None:
            self.embedding_lookup.hydrate(row)
        else:
            row["repo_state_embedding"] = np.asarray(row["repo_state_embedding"], dtype=np.float32)
        if not row.get("target_terminator"):
            raise ValueError(f"row {row['instance_id']} has no target_terminator")
        return row

    def manifest(self, name: str) -> dict[str, Any]:
        weight_summary = {"count": self.weight_count,
                          "min": self.weight_min if self.weight_count else None,
                          "mean": self.weight_sum / self.weight_count if self.weight_count else None,
                          "p50": None, "p90": None, "p95": None,
                          "max": self.weight_max if self.weight_count else None}
        return {"split": name, "rows": self.num_rows, "repos": len(self.repo_counts),
                "task_counts": dict(self.task_counts),
                "task_fractions": {k: v / max(1, self.num_rows) for k, v in self.task_counts.items()},
                "repos_by_task": {k: len(v) for k, v in self.repos_by_task.items()},
                "subtype_counts": dict(self.subtype_counts), "rows_per_repo": numeric_summary(self.repo_counts.values()),
                "metadata_counts": {name: dict(counts) for name, counts in self.metadata_counts.items() if counts},
                "example_weight": weight_summary,
                "instance_id_ordered_sha256": self.instance_id_ordered_sha256,
                "storage": "lazy_parquet_row_groups", "row_groups": self.parquet.metadata.num_row_groups,
                "row_group_cache_size": self.cache_row_groups,
                "external_repo_embeddings": (self.embedding_lookup.manifest()
                                             if self.embedding_lookup is not None else None)}


def numeric_summary(values) -> dict[str, float | int | None]:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p90": None, "p95": None, "max": None}
    def percentile(q: float) -> float:
        pos = (len(vals) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return vals[lo] if lo == hi else vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    return {"count": len(vals), "min": vals[0], "mean": statistics.fmean(vals), "p50": percentile(.5),
            "p90": percentile(.9), "p95": percentile(.95), "max": vals[-1]}


def task_subtype(row: dict[str, Any]) -> str:
    return str(row.get("task_subtype") or row.get("sample_type") or row.get("template_id") or
               row.get("task_type") or "unspecified")


def dataset_manifest(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    repo_counts = Counter(str(r["repo_id"]) for r in rows)
    task_counts = Counter(str(r["task_type"]) for r in rows)
    subtype_counts = Counter(str(r.get("task_subtype") or r.get("sample_type") or r.get("template_id") or "unspecified") for r in rows)
    metadata_counts = {
        column: Counter(str(r.get(column)) for r in rows if r.get(column) is not None and str(r.get(column)))
        for column in ("qa_source", "answer_style", "task_category", "target_format", "validation_status")
    }
    tasks = sorted(task_counts)
    repos_by_task = {task: len({str(r["repo_id"]) for r in rows if str(r["task_type"]) == task}) for task in tasks}
    instance_ids = sorted(str(r["instance_id"]) for r in rows)
    return {"split": name, "rows": len(rows), "repos": len(repo_counts), "task_counts": dict(task_counts),
            "task_fractions": {k: v / max(1, len(rows)) for k, v in task_counts.items()},
            "repos_by_task": repos_by_task, "subtype_counts": dict(subtype_counts),
            "metadata_counts": {name: dict(counts) for name, counts in metadata_counts.items() if counts},
            "rows_per_repo": numeric_summary(repo_counts.values()),
            "example_weight": numeric_summary(r.get("example_weight") for r in rows),
            "instance_id_sha256": hashlib.sha256("\n".join(instance_ids).encode()).hexdigest()}


def write_task_samples(output_dir: Path, train_rows, val_rows, tokenizer, seed: int, per_task: int = 8) -> None:
    selected = []
    for split, source in (("train", train_rows), ("val", val_rows)):
        for task in task_names_from_rows(source):
            values = [r for r in source if str(r["task_type"]) == task]
            random.Random(seed + sum(f"{split}:{task}".encode())).shuffle(values)
            for row in values[:per_task]:
                item = {k: row.get(k) for k in ("repo_id", "instance_id", "task_type", "task_subtype",
                        "sample_type", "template_id", "file_path", "target_terminator", "example_weight",
                        "qa_source", "answer_style", "task_category", "target_format", "validation_status")}
                item.update({"event": "task_sample", "split": split, "student_prompt": str(row["student_prompt"]),
                             "teacher_prompt": str(row["teacher_prompt"]), "target_text": str(row["target_text"])})
                item["student_prompt_tokens"] = len(tokenizer(item["student_prompt"], add_special_tokens=False)["input_ids"])
                item["teacher_prompt_tokens"] = len(tokenizer(item["teacher_prompt"], add_special_tokens=False)["input_ids"])
                item["target_tokens"] = len(tokenizer(item["target_text"] + str(item["target_terminator"]), add_special_tokens=False)["input_ids"])
                selected.append(item)
    with (output_dir / "task_samples.jsonl").open("w", encoding="utf-8") as fh:
        for item in selected:
            fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    md = ["# Context-distillation task samples", ""]
    for i, item in enumerate(selected, 1):
        md += [f"## {i}. {item['split']} / {item['task_type']} / {item['repo_id']}", "",
               f"Instance: `{item['instance_id']}`  ",
               f"Tokens: student={item['student_prompt_tokens']}, teacher={item['teacher_prompt_tokens']}, target={item['target_tokens']}", "",
               "### Student prompt", "", "```text", item["student_prompt"], "```", "",
               "### Teacher prompt", "", "```text", item["teacher_prompt"], "```", "",
               "### Supervised target", "", "```text", item["target_text"] + str(item["target_terminator"]), "```", ""]
    (output_dir / "task_samples.md").write_text("\n".join(md), encoding="utf-8")


def system_metrics(device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {"host_load_1m": os.getloadavg()[0],
                           "process_max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)}
    try:
        disk = shutil.disk_usage(os.environ.get("TMPDIR", "/tmp"))
        out.update({"scratch_free_gb": disk.free / 1e9, "scratch_used_pct": 100 * disk.used / disk.total})
    except OSError:
        pass
    if device.type == "cuda":
        try:
            free, total = torch.cuda.mem_get_info(device)
            out.update({"cuda_free_gb": free / 1e9, "cuda_total_gb": total / 1e9})
        except Exception:
            pass
        for name, fn in (("gpu_utilization_pct", "utilization"), ("gpu_memory_utilization_pct", "memory_usage"),
                         ("gpu_temperature_c", "temperature"), ("gpu_power_mw", "power_draw"),
                         ("gpu_clock_mhz", "clock_rate")):
            try:
                out[name] = float(getattr(torch.cuda, fn)(device))
            except Exception:
                pass
        if "gpu_power_mw" in out:
            out["gpu_power_w"] = out.pop("gpu_power_mw") / 1000.0
    return out


class RepoTaskSampler:
    """Task-ratio sampler, uniform repositories, and temperature-smoothed QA subtypes."""

    def __init__(self, rows, *, ntp_ratio: float, seed: int,
                 qa_subtype_temperature: float = 2.0,
                 task_ratios: dict[str, float] | None = None):
        if not 0.0 <= ntp_ratio <= 1.0:
            raise ValueError("ntp_ratio must be in [0, 1]")
        if qa_subtype_temperature <= 0:
            raise ValueError("qa_subtype_temperature must be positive")
        self.row_store = rows if hasattr(rows, "sampling_records") else None
        self.groups: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
        if self.row_store:
            for reference, repo, task, subtype in self.row_store.sampling_records():
                self.groups[(repo, task, subtype)].append(reference)
            # Group lists now own the compact integer references; release the
            # temporary (repo, task, subtype) tuple index before model loading.
            self.row_store.records.clear()
        else:
            for row in rows:
                self.groups[(str(row["repo_id"]), str(row["task_type"]), task_subtype(row))].append(row)
        self.repos = sorted({key[0] for key in self.groups})
        self.tasks = sorted({key[1] for key in self.groups})
        configured_ratios = dict(task_ratios or parse_task_ratios("", ntp_ratio=ntp_ratio))
        self.task_ratios = {task: float(configured_ratios.get(task, 0.0)) for task in self.tasks}
        if not any(value > 0 for value in self.task_ratios.values()):
            raise ValueError(f"configured task ratios {configured_ratios} do not match available tasks {self.tasks}")
        total_ratio = sum(value for value in self.task_ratios.values() if value > 0)
        self.task_ratios = {task: value / total_ratio for task, value in self.task_ratios.items() if value > 0}
        self.repos_by_task = {
            task: sorted({repo for repo, row_task, _ in self.groups if row_task == task})
            for task in self.tasks
        }
        missing = [task for task in self.task_ratios if not self.repos_by_task.get(task)]
        if missing:
            raise ValueError(f"sampler has no rows for tasks: {missing}")
        self.ntp_ratio = ntp_ratio
        self.qa_subtype_temperature = qa_subtype_temperature
        self.rng = random.Random(seed)
        self.queues: dict[tuple[str, str, str], deque[Any]] = {}
        self.subtypes_by_repo_task: dict[tuple[str, str], list[str]] = defaultdict(list)
        for repo, task, subtype in self.groups:
            self.subtypes_by_repo_task[(repo, task)].append(subtype)
        qa_counts: Counter[str] = Counter()
        for (_, task, subtype), values in self.groups.items():
            if task == "qa":
                qa_counts[subtype] += len(values)
        self.qa_subtype_weights = {
            subtype: count ** (1.0 / qa_subtype_temperature) for subtype, count in qa_counts.items()
        }

    def _take(self, key: tuple[str, str, str]) -> dict[str, Any]:
        queue = self.queues.get(key)
        if not queue:
            values = list(self.groups[key])
            self.rng.shuffle(values)
            queue = self.queues[key] = deque(values)
        value = queue.popleft()
        return self.row_store.get(value) if self.row_store else value

    def sample_task(self, task: str, batch_size: int) -> list[dict[str, Any]]:
        repo = self.rng.choice(self.repos_by_task[task])
        subtypes = self.subtypes_by_repo_task[(repo, task)]
        if task == "qa":
            subtype = self.rng.choices(subtypes, weights=[self.qa_subtype_weights[s] for s in subtypes], k=1)[0]
        else:
            subtype = self.rng.choice(subtypes)
        return [self._take((repo, task, subtype)) for _ in range(batch_size)]

    def sample(self, batch_size: int) -> list[dict[str, Any]]:
        tasks = list(self.task_ratios)
        task = self.rng.choices(tasks, weights=[self.task_ratios[t] for t in tasks], k=1)[0]
        return self.sample_task(task, batch_size)

    def preview_rows(self, per_task: int, seed: int) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        selected = []
        for task in self.tasks:
            keys = [key for key in self.groups if key[1] == task]
            rng.shuffle(keys)
            for key in keys[:per_task]:
                value = rng.choice(self.groups[key])
                selected.append(self.row_store.get(value) if self.row_store else value)
        return selected

    def summary(self) -> dict[str, Any]:
        total = sum(self.qa_subtype_weights.values())
        return {"ntp_ratio": self.ntp_ratio, "qa_ratio": 1.0 - self.ntp_ratio,
                "task_ratios": self.task_ratios,
                "repository_policy": "uniform_within_task", "within_group_policy": "cycle_without_replacement",
                "qa_subtype_temperature": self.qa_subtype_temperature,
                "qa_subtype_sampling_weights": self.qa_subtype_weights,
                "qa_subtype_global_probabilities": ({k: v / total for k, v in self.qa_subtype_weights.items()}
                                                    if total else {}),
                "repos_by_task": {k: len(v) for k, v in self.repos_by_task.items()}}


def clear_lora(model, specs) -> None:
    named = dict(model.named_modules())
    for spec in specs:
        named[spec.full_name].set_lora_weights(None, None)


def target_views(logits: torch.Tensor, labels: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    out = []
    for i in range(logits.size(0)):
        mask = shifted_labels[i] != -100
        out.append((shifted_logits[i][mask], shifted_labels[i][mask]))
    return out


def distillation_loss(student_views, teacher_views, rows, args) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    ce_values, kl_values = [], []
    for (student, labels), (teacher, teacher_labels), row in zip(student_views, teacher_views, rows):
        if student.size(0) != teacher.size(0) or not torch.equal(labels, teacher_labels):
            raise RuntimeError("teacher/student target token alignment failed")
        ce = F.cross_entropy(student.float(), labels, reduction="mean")
        t = args.temperature
        if args.teacher_top_k > 0:
            k = min(args.teacher_top_k, teacher.size(-1))
            teacher_top = torch.topk(teacher.float() / t, k=k, dim=-1)
            teacher_logp = F.log_softmax(teacher_top.values, dim=-1)
            student_selected = torch.gather(student.float() / t, -1, teacher_top.indices)
            student_logp = F.log_softmax(student_selected, dim=-1)
            kl = F.kl_div(student_logp, teacher_logp.exp(), reduction="batchmean") * (t * t)
        else:
            teacher_prob = F.softmax(teacher.float() / t, dim=-1)
            student_logp = F.log_softmax(student.float() / t, dim=-1)
            kl = F.kl_div(student_logp, teacher_prob, reduction="batchmean") * (t * t)
        ce_lambda = ce_weight_for_task(row, args)
        weight = float(row["example_weight"])
        losses.append(weight * (args.lambda_kl * kl + ce_lambda * ce))
        ce_values.append(float(ce.detach().item()))
        kl_values.append(float(kl.detach().item()))
    return torch.stack(losses).mean(), {"ce": sum(ce_values) / len(ce_values), "kl": sum(kl_values) / len(kl_values)}


def forward_pair(model, head, specs, tokenizer, rows, args, device, *, embedding_rows=None, with_grad=True):
    forward_started = time.perf_counter()
    targets = [str(r["target_text"]) for r in rows]
    terminators = [str(r["target_terminator"]) for r in rows]
    teacher_prompts = ([str(r["student_prompt"]) for r in rows]
                       if args.kl_source == "base"
                       else [str(r["teacher_prompt"]) for r in rows])
    teacher_batch = tokenize_prompt_target_batch(tokenizer, teacher_prompts, targets, terminators,
                                                 max_seq_len=args.max_seq_len)
    student_batch = tokenize_prompt_target_batch(tokenizer, [str(r["student_prompt"]) for r in rows], targets,
                                                  terminators, max_seq_len=args.max_seq_len)
    if not teacher_batch or not student_batch:
        return None
    teacher_batch = {key: value.to(device) for key, value in teacher_batch.items()}
    student_batch = {key: value.to(device) for key, value in student_batch.items()}
    target_tokens = int((student_batch["labels"] != -100).sum().item())
    student_tokens = int(student_batch["attention_mask"].sum().item())
    teacher_tokens = int(teacher_batch["attention_mask"].sum().item())
    clear_lora(model, specs)
    with torch.no_grad():
        teacher_logits = model(input_ids=teacher_batch["input_ids"], attention_mask=teacher_batch["attention_mask"]).logits
        # Clone target slices so they do not retain the much larger full-sequence
        # logits allocation while the differentiable student forward runs.
        teacher_views = [(x.detach().clone(), y.detach().clone())
                         for x, y in target_views(teacher_logits, teacher_batch["labels"])]
    del teacher_logits
    source = embedding_rows or rows
    ctx = torch.stack([torch.from_numpy(np.asarray(r["repo_state_embedding"], dtype=np.float32)) for r in source]).to(device)
    head_out = head(ctx)
    inject_lora_weights(model, specs, head_out, batch_index=0)
    # Same-repository batches intentionally share one generated adapter.
    if len(rows) > 1:
        named = dict(model.named_modules())
        for spec in specs:
            named[spec.full_name].set_lora_weights(head_out["A"][spec.type][0], head_out["B"][spec.type][0])
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        student_logits = model(input_ids=student_batch["input_ids"], attention_mask=student_batch["attention_mask"]).logits
        student_views = target_views(student_logits, student_batch["labels"])
        loss, pieces = distillation_loss(student_views, teacher_views, rows, args)
    pieces.update({"target_tokens": target_tokens, "student_tokens": student_tokens,
                   "teacher_tokens": teacher_tokens, "student_prompt_tokens": student_tokens - target_tokens,
                   "teacher_prompt_tokens": teacher_tokens - target_tokens,
                   "context_tokens": teacher_tokens - student_tokens,
                   "kl_source": args.kl_source,
                   ("base_kl" if args.kl_source == "base" else "teacher_kl"): pieces["kl"],
                   "forward_wall_seconds": time.perf_counter() - forward_started})
    return loss, pieces, head_out, student_views, teacher_views


def sampled_task_gradient(model, head, specs, tokenizer, rows, args, device) -> tuple[torch.Tensor, dict[str, Any]] | None:
    """Return a deterministic coordinate sample plus the exact full gradient norm."""
    head.zero_grad(set_to_none=True)
    result = forward_pair(model, head, specs, tokenizer, rows, args, device, with_grad=True)
    if result is None:
        return None
    loss, pieces, _, _, _ = result
    loss.backward()
    grads = [parameter.grad.detach() for parameter in head.parameters() if parameter.grad is not None]
    total_elements = sum(grad.numel() for grad in grads)
    stride = max(1, math.ceil(total_elements / max(1, args.gradient_diagnostic_elements)))
    sampled = torch.cat([grad.float().reshape(-1)[::stride].cpu() for grad in grads])
    sampled = sampled[:args.gradient_diagnostic_elements]
    full_norm = math.sqrt(sum(float(torch.linalg.vector_norm(grad.float()).item()) ** 2 for grad in grads))
    info = {"loss": float(loss.detach().item()), "ce": pieces["ce"], "kl": pieces["kl"],
            "full_grad_norm": full_norm, "sampled_elements": int(sampled.numel()),
            "repo_id": str(rows[0]["repo_id"]), "instance_id": str(rows[0]["instance_id"]),
            "task_subtype": task_subtype(rows[0])}
    head.zero_grad(set_to_none=True)
    return sampled, info


def gradient_diagnostics(model, head, specs, tokenizer, sampler, args, device) -> dict[str, Any] | None:
    by_task = {}
    vectors = {}
    diagnostic_tasks = list(sampler.task_ratios)[:2]
    if len(diagnostic_tasks) < 2:
        return None
    for task in diagnostic_tasks:
        result = sampled_task_gradient(model, head, specs, tokenizer, sampler.sample_task(task, 1), args, device)
        if result is None:
            return None
        vectors[task], by_task[task] = result
    first, second = diagnostic_tasks
    denominator = float(vectors[first].norm().item() * vectors[second].norm().item())
    cosine = float(torch.dot(vectors[first], vectors[second]).item() / denominator) if denominator else None
    return {"event": "gradient_diagnostic", "task_gradient_cosine": cosine,
            "task_gradient_pair": diagnostic_tasks,
            "task_gradients_conflict": cosine is not None and cosine < 0.0,
            "diagnostic_method": "deterministic_coordinate_sample", "by_task": by_task}


def normalize_answer(text: str) -> str:
    import re
    text = text.strip().strip("\"'`")
    text = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


@torch.no_grad()
def base_student_ce(model, specs, tokenizer, row, args, device) -> float:
    batch = tokenize_prompt_target_batch(tokenizer, [str(row["student_prompt"])], [str(row["target_text"])],
                                         [str(row["target_terminator"])], max_seq_len=args.max_seq_len)
    if not batch:
        return float("nan")
    batch = {key: value.to(device) for key, value in batch.items()}
    clear_lora(model, specs)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    view, labels = target_views(logits, batch["labels"])[0]
    return float(F.cross_entropy(view.float(), labels).item())


def balanced_eval_rows(rows: list[dict[str, Any]], limit: int, seed: int,
                       task_ratios: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Deterministically balance repositories, and tasks when a finite limit is used."""
    def round_robin(source: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        by_repo: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        for repo_id in sorted({str(r["repo_id"]) for r in source}):
            values = [r for r in source if str(r["repo_id"]) == repo_id]
            random.Random(seed + sum(repo_id.encode())).shuffle(values)
            by_repo[repo_id] = deque(values)
        chosen, active = [], deque(sorted(by_repo))
        while active and len(chosen) < count:
            repo_id = active.popleft()
            if by_repo[repo_id]:
                chosen.append(by_repo[repo_id].popleft())
            if by_repo[repo_id]:
                active.append(repo_id)
        return chosen

    if not limit:
        return round_robin(rows, len(rows))
    if not task_ratios:
        return round_robin(rows, limit)
    available = sorted({str(r["task_type"]) for r in rows})
    normalized = {task: float(task_ratios.get(task, 0.0)) for task in available}
    total = sum(v for v in normalized.values() if v > 0)
    if total <= 0:
        return round_robin(rows, limit)
    normalized = {task: value / total for task, value in normalized.items() if value > 0}
    selected: list[dict[str, Any]] = []
    remaining = limit
    for idx, task in enumerate(sorted(normalized)):
        count = remaining if idx == len(normalized) - 1 else round(limit * normalized[task])
        task_rows = [r for r in rows if str(r["task_type"]) == task]
        selected += round_robin(task_rows, count)
        remaining = max(0, limit - len(selected))
    if len(selected) < limit:
        selected_ids = {str(r["instance_id"]) for r in selected}
        selected += round_robin([r for r in rows if str(r["instance_id"]) not in selected_ids], limit - len(selected))
    return selected


@torch.no_grad()
def evaluate(model, head, specs, tokenizer, rows, args, device, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_started = time.perf_counter()
    task_ratios = getattr(args, "task_ratios_map", parse_task_ratios("", ntp_ratio=args.ntp_ratio))
    selected = balanced_eval_rows(rows, args.limit_val_rows, seed, task_ratios)
    eval_tasks = sorted({str(r["task_type"]) for r in rows})
    totals: Counter[str] = Counter()
    sums: defaultdict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    repo_to_embedding = {}
    for row in rows:
        repo_to_embedding.setdefault(str(row["repo_id"]), row)
    repo_ids = sorted(repo_to_embedding)
    for row in selected:
        result = forward_pair(model, head, specs, tokenizer, [row], args, device, with_grad=False)
        if result is None:
            continue
        loss, pieces, _, student_views, teacher_views = result
        task = str(row["task_type"])
        student_ce = float(F.cross_entropy(student_views[0][0].float(), student_views[0][1]).item())
        teacher_ce = float(F.cross_entropy(teacher_views[0][0].float(), teacher_views[0][1]).item())
        base_ce = base_student_ce(model, specs, tokenizer, row, args, device)
        wrong_candidates = [r for r in repo_ids if r != str(row["repo_id"])]
        wrong_ce = float("nan")
        if wrong_candidates:
            wrong_repo = wrong_candidates[sum(str(row["instance_id"]).encode()) % len(wrong_candidates)]
            wrong_result = forward_pair(model, head, specs, tokenizer, [row], args, device,
                                        embedding_rows=[repo_to_embedding[wrong_repo]], with_grad=False)
            if wrong_result is not None:
                wrong_ce = float(F.cross_entropy(wrong_result[3][0][0].float(), wrong_result[3][0][1]).item())
        for key, value in (("total_loss", float(loss.item())), ("student_ce", student_ce),
                           ("teacher_ce", teacher_ce), ("base_ce", base_ce),
                           ("wrong_repo_ce", wrong_ce), ("kl", pieces["kl"]),
                           ("base_kl", pieces.get("base_kl", float("nan"))),
                           ("teacher_kl", pieces.get("teacher_kl", float("nan")))):
            if math.isfinite(value):
                sums[key] += value
                sums[f"{task}_{key}"] += value
                counts[key] += 1
                counts[f"{task}_{key}"] += 1
        totals["rows"] += 1
        totals[f"{task}_rows"] += 1
        details.append({"repo_id": row["repo_id"], "instance_id": row["instance_id"], "task_type": task,
                        "task_subtype": row.get("task_subtype") or row.get("sample_type") or row.get("template_id"),
                        "loss": float(loss.item()), "student_ce": student_ce, "teacher_ce": teacher_ce,
                        "base_ce": base_ce, "wrong_repo_ce": wrong_ce if math.isfinite(wrong_ce) else None,
                        "kl": pieces["kl"], "base_kl": pieces.get("base_kl"),
                        "teacher_kl": pieces.get("teacher_kl"),
                        "target_tokens": pieces["target_tokens"]})

    qa_rows = [r for r in selected if r["task_type"] == "qa"][:args.qa_generation_rows]
    exact = 0
    for row in qa_rows:
        ctx = torch.from_numpy(np.asarray(row["repo_state_embedding"], dtype=np.float32)).unsqueeze(0).to(device)
        head_out = head(ctx)
        inject_lora_weights(model, specs, head_out, batch_index=0)
        ids = tokenizer(str(row["student_prompt"]), add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        stop_ids = [tokenizer.eos_token_id]
        eot = tokenizer.special_tokens_map.get("eot_token")
        if eot:
            eot_id = tokenizer.convert_tokens_to_ids(eot)
            if eot_id not in stop_ids:
                stop_ids.append(eot_id)
        generated = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids)
        answer = tokenizer.decode(generated[0, ids.size(1):], skip_special_tokens=True)
        is_exact = int(normalize_answer(answer) == normalize_answer(str(row["target_text"])))
        exact += is_exact
        for detail in details:
            if detail["instance_id"] == row["instance_id"]:
                detail.update({"generated_answer": answer, "gold_answer": str(row["target_text"]), "exact_match": is_exact})
                break
    eval_seconds = time.perf_counter() - eval_started
    subtype_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        subtype_groups[str(detail.get("task_subtype") or detail["task_type"])].append(detail)
    subtype_metrics = {}
    for subtype, group in subtype_groups.items():
        subtype_metrics[subtype] = {"rows": len(group)}
        for key in ("loss", "student_ce", "teacher_ce", "base_ce", "wrong_repo_ce", "kl", "base_kl", "teacher_kl"):
            subtype_metrics[subtype][key] = numeric_summary(d.get(key) for d in group)["mean"]
    out: dict[str, Any] = {"rows_scored": totals["rows"], "qa_generation_rows": len(qa_rows),
                           "qa_exact_match": exact / max(1, len(qa_rows)), "qa_exact_count": exact,
                           "eval_seconds": eval_seconds, "rows_per_sec": totals["rows"] / max(eval_seconds, 1e-9),
                           "by_task_subtype": subtype_metrics,
                           "selection_task_counts": dict(Counter(str(r["task_type"]) for r in selected)),
                           "selection_seed": seed,
                           "selected_instance_sha256": hashlib.sha256("\n".join(str(r["instance_id"]) for r in selected).encode()).hexdigest()}
    for key in ("total_loss", "student_ce", "teacher_ce", "base_ce", "wrong_repo_ce", "kl", "base_kl", "teacher_kl"):
        out[key] = sums[key] / counts[key] if counts[key] else None
        for task in eval_tasks:
            count = counts[f"{task}_{key}"]
            out[f"{task}_{key}"] = sums[f"{task}_{key}"] / count if count else None
    out["repo_sensitivity_gap"] = (out["wrong_repo_ce"] - out["student_ce"]
                                   if out["wrong_repo_ce"] is not None and out["student_ce"] is not None else None)
    out["repo_embedding_gain"] = (out["base_ce"] - out["student_ce"]
                                  if out["base_ce"] is not None and out["student_ce"] is not None else None)
    ratios = []
    for task, weight in task_ratios.items():
        student, base = out[f"{task}_student_ce"], out[f"{task}_base_ce"]
        if student is not None and base is not None and base > 0:
            ratios.append((weight, student / base))
    out["normalized_composite"] = (sum(weight * value for weight, value in ratios) /
                                   sum(weight for weight, _ in ratios)) if ratios else None

    repo_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        repo_groups[str(detail["repo_id"])].append(detail)
    per_repo = {}
    row_gains, row_gaps, row_wins = [], [], []
    for repo_id, group in sorted(repo_groups.items()):
        student = numeric_summary(d["student_ce"] for d in group)["mean"]
        base = numeric_summary(d["base_ce"] for d in group)["mean"]
        wrong = numeric_summary(d["wrong_repo_ce"] for d in group)["mean"]
        gains = [d["base_ce"] - d["student_ce"] for d in group
                 if d.get("base_ce") is not None and math.isfinite(float(d["base_ce"]))]
        gaps = [d["wrong_repo_ce"] - d["student_ce"] for d in group
                if d.get("wrong_repo_ce") is not None and math.isfinite(float(d["wrong_repo_ce"]))]
        wins = [int(gap > 0) for gap in gaps]
        row_gains.extend(gains)
        row_gaps.extend(gaps)
        row_wins.extend(wins)
        per_repo[repo_id] = {"rows": len(group), "student_ce": student, "base_ce": base, "wrong_repo_ce": wrong,
                             "embedding_gain": statistics.fmean(gains) if gains else None,
                             "sensitivity_gap": statistics.fmean(gaps) if gaps else None,
                             "correct_vs_wrong_win_rate": statistics.fmean(wins) if wins else None,
                             "by_task": {
                                 task: {"rows": len(task_group),
                                        "embedding_gain": numeric_summary(
                                            d["base_ce"] - d["student_ce"] for d in task_group)["mean"],
                                        "sensitivity_gap": numeric_summary(
                                            d["wrong_repo_ce"] - d["student_ce"] for d in task_group
                                            if d.get("wrong_repo_ce") is not None)["mean"],
                                        "correct_vs_wrong_win_rate": numeric_summary(
                                            int(d["wrong_repo_ce"] > d["student_ce"]) for d in task_group
                                            if d.get("wrong_repo_ce") is not None)["mean"]}
                                 for task in eval_tasks
                                 if (task_group := [d for d in group if d["task_type"] == task])
                             }}
    repo_gains = [v["embedding_gain"] for v in per_repo.values() if v["embedding_gain"] is not None]
    repo_gaps = [v["sensitivity_gap"] for v in per_repo.values() if v["sensitivity_gap"] is not None]
    repo_wins = [v["correct_vs_wrong_win_rate"] for v in per_repo.values()
                 if v["correct_vs_wrong_win_rate"] is not None]
    out["repo_specialization"] = {
        "repos_scored": len(per_repo), "rows_with_wrong_embedding": len(row_gaps),
        "row_embedding_gain": numeric_summary(row_gains), "row_sensitivity_gap": numeric_summary(row_gaps),
        "correct_vs_wrong_row_win_rate": statistics.fmean(row_wins) if row_wins else None,
        "per_repo_embedding_gain": numeric_summary(repo_gains),
        "per_repo_sensitivity_gap": numeric_summary(repo_gaps),
        "correct_vs_wrong_repo_win_rate": statistics.fmean(repo_wins) if repo_wins else None,
        "positive_sensitivity_repo_fraction": sum(gap > 0 for gap in repo_gaps) / max(1, len(repo_gaps)),
        "by_task": {
            task: {"row_embedding_gain": numeric_summary(
                       d["base_ce"] - d["student_ce"] for d in details if d["task_type"] == task),
                   "row_sensitivity_gap": numeric_summary(
                       d["wrong_repo_ce"] - d["student_ce"] for d in details
                       if d["task_type"] == task and d.get("wrong_repo_ce") is not None),
                   "correct_vs_wrong_row_win_rate": numeric_summary(
                       int(d["wrong_repo_ce"] > d["student_ce"]) for d in details
                       if d["task_type"] == task and d.get("wrong_repo_ce") is not None)["mean"]}
            for task in eval_tasks
        },
        "per_repo": per_repo,
    }
    return out, details


def main() -> None:
    args = parse_args()
    if args.lambda_base_kl is not None:
        args.lambda_kl = float(args.lambda_base_kl)
        args.kl_source = "base"
    args.task_ratios_map = parse_task_ratios(args.task_ratios, ntp_ratio=args.ntp_ratio)
    args.lambda_ce_by_task = parse_task_weights(args.lambda_ce_task, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    embedding_lookup = RepoEmbeddingLookup(args.repo_embeddings) if args.repo_embeddings else None
    train_store = ParquetRowStore(args.train_data, cache_row_groups=args.row_group_cache_size,
                                  embedding_lookup=embedding_lookup)
    val_rows = load_rows(args.val_data, embedding_lookup=embedding_lookup)
    train_repos = set(train_store.repo_counts)
    val_repos = {str(r["repo_id"]) for r in val_rows}
    overlap = train_repos & val_repos
    if overlap:
        raise RuntimeError(f"train/validation repository leakage: {sorted(overlap)[:10]}")
    if len(train_repos) < args.min_train_repos:
        raise RuntimeError(f"training table has only {len(train_repos)} repositories; "
                           f"minimum is {args.min_train_repos}")
    if len(val_repos) < args.min_val_repos:
        raise RuntimeError(f"validation table has only {len(val_repos)} repositories; "
                           f"minimum is {args.min_val_repos}")
    sampler = RepoTaskSampler(train_store, ntp_ratio=args.ntp_ratio, seed=args.seed,
                              qa_subtype_temperature=args.qa_subtype_temperature,
                              task_ratios=args.task_ratios_map)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True,
                                               local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_gemma4_model(args.model_name, dtype=torch_dtype(args.dtype), device=str(device),
                              local_files_only=args.local_files_only)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    specs = get_module_specs(model, args.target_modules, module_name_regex=args.module_name_regex,
                             shape_aware_types=args.shape_aware_types)
    if args.target_shape_types:
        wanted = set(args.target_shape_types)
        specs = [spec for spec in specs if spec.type in wanted]
        missing = sorted(wanted - {spec.type for spec in specs})
        if missing:
            raise ValueError(f"requested target shape types not found: {missing}")
    if not specs:
        raise ValueError("no target module specs selected")
    type_dims = discover_module_types_and_dims(specs)
    specs_summary = summarize_specs(specs)
    shape_group_summary = target_shape_group_summary(
        specs, type_dims, hidden_dim=args.head_hidden_dim, rank=args.rank
    )
    replace_with_lora(model, specs, rank=args.rank, alpha=args.alpha)
    input_dim = int(train_store.get(0)["repo_state_embedding"].shape[0])
    if args.head_input_dim and input_dim != args.head_input_dim:
        raise ValueError(f"repo_state_embedding dim {input_dim} != --head-input-dim {args.head_input_dim}")
    head = Code2LoRAHead(input_dim, type_dims, hidden_dim=args.head_hidden_dim, rank=args.rank).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, int(args.max_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, args.max_steps)
    resume = load_resume_checkpoint(args.resume_from, device=device)
    state = resume.get("trainer_state", {}) if resume else {}
    step = int(state.get("global_step", 0))
    best_qa_em = float(state.get("best_qa_exact_match", -1.0))
    best_qa_ce = float(state.get("best_qa_student_ce", float("inf")))
    best_ntp_ce = float(state.get("best_ntp_student_ce", float("inf")))
    best_composite = float(state.get("best_normalized_composite", float("inf")))
    if resume:
        head.load_state_dict(resume["state_dict"])
        if resume.get("optimizer_state_dict"):
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        if resume.get("scheduler_state_dict"):
            scheduler.load_state_dict(resume["scheduler_state_dict"])
        restore_rng_state(resume.get("rng_state", {}))
    metrics_path, events_path = args.output_dir / "metrics.jsonl", args.output_dir / "events.jsonl"
    serialized_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config = {"event": "run_config", "metrics_schema_version": 3, "args": serialized_args, "task_name": "context_distill",
                  "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                  "git_revision": git_revision(Path(__file__).resolve().parents[2]), "started_unix": time.time(),
                  "task_ratios": args.task_ratios_map, "lambda_ce_by_task": args.lambda_ce_by_task,
                  "kl_source": args.kl_source}
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True, default=str) + "\n")
    append_jsonl(events_path, run_config)
    manifests = {"train": train_store.manifest("train"), "val": dataset_manifest(val_rows, "val")}
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(manifests, indent=2, sort_keys=True) + "\n")
    write_task_samples(args.output_dir, sampler.preview_rows(8, args.seed), val_rows, tokenizer, args.seed)
    append_jsonl(events_path, {"event": "task_samples_written", "jsonl": str(args.output_dir / "task_samples.jsonl"),
                               "markdown": str(args.output_dir / "task_samples.md")})
    summary = {"event": "model_summary", "metrics_schema_version": 3,
               "train_rows": train_store.num_rows, "val_rows": len(val_rows),
               "train_repos": len(train_repos), "val_repos": len(val_repos), "total_steps": args.max_steps,
               "task_counts": dict(train_store.task_counts), "rank": args.rank,
               "alpha": args.alpha, "head_hidden_dim": args.head_hidden_dim,
               "head_input_dim": input_dim,
               "head_params": count_head_parameters(input_dim, type_dims, hidden_dim=args.head_hidden_dim, rank=args.rank),
               "specs": specs_summary,
               "target_shape_groups": shape_group_summary["groups"],
               "target_shape_types": shape_group_summary["target_shape_types"],
               "target_module_count": shape_group_summary["target_module_count"],
               "target_type_count": shape_group_summary["target_type_count"],
               "generated_lora_scalars_per_repo": shape_group_summary["generated_lora_scalars_per_repo"],
               "injected_lora_scalars_per_repo": shape_group_summary["injected_lora_scalars_per_repo"],
               "head_params_by_shape_total": shape_group_summary["head_params_by_shape_total"],
               "warmup_steps": warmup,
               "task_ratios": args.task_ratios_map,
               "lambda_ce_by_task": args.lambda_ce_by_task,
               "kl_source": args.kl_source,
               "train_order": "task_ratio_then_uniform_repo_then_temperature_subtype",
               "sampler": sampler.summary(),
               "dataset": manifests}
    append_jsonl(metrics_path, summary)
    train_tasks = sorted(train_store.task_counts)
    started = time.time()
    interval_started = time.perf_counter()
    window: list[dict[str, Any]] = []
    seen_repos: set[str] = set()
    seen_repos_by_task: defaultdict[str, set[str]] = defaultdict(set)
    seen_instances: set[str] = set()
    cumulative_tasks: Counter[str] = Counter()
    cumulative_subtypes: Counter[str] = Counter()
    repo_visit_counts: Counter[str] = Counter()
    repo_visit_counts_by_task: defaultdict[str, Counter[str]] = defaultdict(Counter)
    while step < args.max_steps:
        step_started = time.perf_counter()
        rows = sampler.sample(args.batch_size)
        head.train()
        result = forward_pair(model, head, specs, tokenizer, rows, args, device, with_grad=True)
        if result is None:
            continue
        loss, pieces, head_out, _, _ = result
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        task = str(rows[0]["task_type"])
        repo_id = str(rows[0]["repo_id"])
        subtype = task_subtype(rows[0])
        seen_repos.add(repo_id)
        seen_repos_by_task[task].add(repo_id)
        seen_instances.update(str(r["instance_id"]) for r in rows)
        cumulative_tasks[task] += len(rows)
        cumulative_subtypes[subtype] += len(rows)
        repo_visit_counts[repo_id] += len(rows)
        repo_visit_counts_by_task[task][repo_id] += len(rows)
        window.append({"loss": float(loss.item()), "ce": pieces["ce"], "kl": pieces["kl"],
                       "base_kl": pieces.get("base_kl"), "teacher_kl": pieces.get("teacher_kl"),
                       "grad": float(grad_norm), "step_seconds": time.perf_counter() - step_started,
                       "task": task, "subtype": subtype, "repo_id": repo_id,
                       **{k: pieces[k] for k in ("target_tokens", "student_tokens", "teacher_tokens",
                                    "student_prompt_tokens", "teacher_prompt_tokens", "context_tokens", "forward_wall_seconds")}})
        if step % args.log_every == 0:
            elapsed = time.time() - started
            rate = step / max(elapsed, 1e-9)
            interval_seconds = time.perf_counter() - interval_started
            task_windows = {t: [x for x in window if x["task"] == t] for t in train_tasks}
            subtype_windows = {s: [x for x in window if x["subtype"] == s]
                               for s in sorted({x["subtype"] for x in window})}
            total_tokens = sum(x["student_tokens"] + x["teacher_tokens"] for x in window)
            target_tokens = sum(x["target_tokens"] for x in window)
            visit_total = sum(repo_visit_counts.values())
            visit_entropy = (-sum((count / visit_total) * math.log(count / visit_total)
                                  for count in repo_visit_counts.values()) / math.log(len(train_repos))
                             if visit_total and len(train_repos) > 1 else 0.0)
            row = {"event": "train", "metrics_schema_version": 3, "step": step, "repo_id": repo_id,
                   "instance_id": rows[0]["instance_id"], "task_type": task,
                   "task_subtype": rows[0].get("task_subtype") or rows[0].get("sample_type") or rows[0].get("template_id"),
                   "train_loss": statistics.fmean(x["loss"] for x in window),
                   "train_ce": statistics.fmean(x["ce"] for x in window),
                   "train_kl": statistics.fmean(x["kl"] for x in window), "lr": scheduler.get_last_lr()[0],
                   "train_base_kl": numeric_summary(x.get("base_kl") for x in window)["mean"],
                   "train_teacher_kl": numeric_summary(x.get("teacher_kl") for x in window)["mean"],
                   "grad_norm_pre_clip": float(grad_norm), "examples_seen": step * args.batch_size,
                   "steps_per_sec": rate, "eta_hours": (args.max_steps - step) / max(rate, 1e-9) / 3600,
                   "interval_seconds": interval_seconds, "interval_steps_per_sec": len(window) / max(interval_seconds, 1e-9),
                   "examples_per_sec": len(window) * args.batch_size / max(interval_seconds, 1e-9),
                   "tokens_per_sec": total_tokens / max(interval_seconds, 1e-9),
                   "target_tokens_per_sec": target_tokens / max(interval_seconds, 1e-9),
                   "target_tokens": statistics.fmean(x["target_tokens"] for x in window),
                   "student_prompt_tokens": statistics.fmean(x["student_prompt_tokens"] for x in window),
                   "teacher_prompt_tokens": statistics.fmean(x["teacher_prompt_tokens"] for x in window),
                   "context_tokens": statistics.fmean(x["context_tokens"] for x in window),
                   "step_time": numeric_summary(x["step_seconds"] for x in window),
                   "loss_window": numeric_summary(x["loss"] for x in window),
                   "grad_window": numeric_summary(x["grad"] for x in window),
                   "task_window_counts": {t: len(v) for t, v in task_windows.items()},
                   "task_window": {t: {"loss": statistics.fmean(x["loss"] for x in v),
                                        "ce": statistics.fmean(x["ce"] for x in v),
                                        "kl": statistics.fmean(x["kl"] for x in v),
                                        "base_kl": numeric_summary(x.get("base_kl") for x in v)["mean"],
                                        "teacher_kl": numeric_summary(x.get("teacher_kl") for x in v)["mean"],
                                        "grad_norm": statistics.fmean(x["grad"] for x in v),
                                        "step_seconds": statistics.fmean(x["step_seconds"] for x in v),
                                        "target_tokens": statistics.fmean(x["target_tokens"] for x in v)}
                                   for t, v in task_windows.items() if v},
                   "subtype_window": {s: {"count": len(v), "loss": statistics.fmean(x["loss"] for x in v),
                                           "ce": statistics.fmean(x["ce"] for x in v),
                                           "kl": statistics.fmean(x["kl"] for x in v)}
                                      for s, v in subtype_windows.items()},
                   "cumulative_task_counts": dict(cumulative_tasks),
                   "cumulative_task_fractions": {t: count / max(1, sum(cumulative_tasks.values()))
                                                 for t, count in cumulative_tasks.items()},
                   "cumulative_subtype_counts": dict(cumulative_subtypes),
                   "unique_repos_seen": len(seen_repos), "repo_coverage_pct": 100 * len(seen_repos) / len(train_repos),
                   "unique_ntp_repos_seen": len(seen_repos_by_task["ntp"]),
                   "unique_qa_repos_seen": len(seen_repos_by_task["qa"]),
                   "unique_repos_seen_by_task": {t: len(seen_repos_by_task[t]) for t in train_tasks},
                   "unique_instances_seen": len(seen_instances),
                   "instance_coverage_pct": 100 * len(seen_instances) / train_store.num_rows,
                   "repo_visit_count": numeric_summary(repo_visit_counts.get(repo, 0) for repo in train_repos),
                   "repo_visit_count_by_task": {t: numeric_summary(repo_visit_counts_by_task[t].get(repo, 0)
                                                                    for repo in train_repos)
                                                for t in train_tasks},
                   "repo_visit_entropy_normalized": visit_entropy,
                   **lora_tensor_stats(head_out), **scale_stats(head), **cuda_memory(), **system_metrics(device)}
            print(json.dumps(row), flush=True)
            append_jsonl(metrics_path, row)
            window.clear()
            interval_started = time.perf_counter()
        if args.gradient_diagnostics_every and step % args.gradient_diagnostics_every == 0:
            head.train()
            diagnostic = gradient_diagnostics(model, head, specs, tokenizer, sampler, args, device)
            if diagnostic:
                diagnostic.update({"metrics_schema_version": 3, "step": step,
                                   **cuda_memory(), **system_metrics(device)})
                print(json.dumps(diagnostic), flush=True)
                append_jsonl(metrics_path, diagnostic)
        if args.eval_every and step % args.eval_every == 0:
            head.eval()
            val, eval_details = evaluate(model, head, specs, tokenizer, val_rows, args, device, args.seed)
            with (args.output_dir / f"eval_samples_step{step:09d}.jsonl").open("w", encoding="utf-8") as fh:
                for detail in eval_details:
                    fh.write(json.dumps(detail, ensure_ascii=False, default=str) + "\n")
            # Compatibility aliases keep the existing dashboard useful while
            # retaining the richer context-distillation metrics.
            evals = {
                "context_distill": val,
                "correct_embedding": {"loss": val["student_ce"], "rows_scored": val["rows_scored"], "rows_skipped": 0},
                "base_model": {"loss": val["base_ce"]},
                "shuffled_embedding": {"loss": val["wrong_repo_ce"]},
                "repo_embedding_gain": val["repo_embedding_gain"],
                "repo_sensitivity_gap": val["repo_sensitivity_gap"],
            }
            event = {"event": "eval", "metrics_schema_version": 3, "step": step, "val": val, "evals": evals,
                     "artifact": f"eval_samples_step{step:09d}.jsonl", **cuda_memory(), **system_metrics(device)}
            print(json.dumps(event), flush=True)
            append_jsonl(metrics_path, event)
            trainer_state = {"global_step": step, "best_qa_exact_match": best_qa_em,
                             "best_qa_student_ce": best_qa_ce, "best_ntp_student_ce": best_ntp_ce,
                             "best_normalized_composite": best_composite}
            ntp_ce, qa_ce, em, composite = (val.get("ntp_student_ce"), val.get("qa_student_ce"),
                                             val.get("qa_exact_match", 0.0), val.get("normalized_composite"))
            if ntp_ce is not None and ntp_ce < best_ntp_ce:
                best_ntp_ce = ntp_ce
                trainer_state["best_ntp_student_ce"] = best_ntp_ce
                save_checkpoint(args.output_dir, "best_ntp", head, args, type_dims, specs_summary,
                                trainer_state=trainer_state)
            if qa_ce is not None and (em > best_qa_em or (em == best_qa_em and qa_ce < best_qa_ce)):
                best_qa_em, best_qa_ce = em, qa_ce
                trainer_state.update({"best_qa_exact_match": best_qa_em, "best_qa_student_ce": best_qa_ce})
                save_checkpoint(args.output_dir, "best_qa", head, args, type_dims, specs_summary,
                                trainer_state=trainer_state)
            if composite is not None and composite < best_composite:
                best_composite = composite
                trainer_state["best_normalized_composite"] = best_composite
                save_checkpoint(args.output_dir, "best_composite", head, args, type_dims, specs_summary,
                                trainer_state=trainer_state)
        if args.save_every and step % args.save_every == 0:
            trainer_state = {"global_step": step, "best_qa_exact_match": best_qa_em,
                             "best_qa_student_ce": best_qa_ce, "best_ntp_student_ce": best_ntp_ce,
                             "best_normalized_composite": best_composite}
            path = save_checkpoint(args.output_dir, "latest", head, args, type_dims, specs_summary,
                                   optimizer=optimizer, scheduler=scheduler, trainer_state=trainer_state)
            append_jsonl(events_path, {"event": "checkpoint", "step": step, "path": str(path)})
            if args.archive_every and step % args.archive_every == 0:
                archive = save_checkpoint(args.output_dir, f"step{step}", head, args, type_dims, specs_summary,
                                          trainer_state=trainer_state, include_rng=False)
                append_jsonl(events_path, {"event": "checkpoint_archive", "step": step, "path": str(archive)})
    print(f"done: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
