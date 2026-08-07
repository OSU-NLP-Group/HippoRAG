#!/usr/bin/env python3
"""Run paired open-answer and MCQ Doc-to-LoRA broad evaluations.

The evaluator deliberately uses one implementation for base, oracle-span,
concat, and TIES conditions.  Adapter modes materialize one repository adapter
per pinned snapshot and reuse it across every selected question for that
snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from broad_eval_common import (  # noqa: E402
    atomic_write_json,
    extract_mcq_label,
    normalize_answer,
    read_jsonl,
    repo_path,
    softmax,
    token_f1,
)


MODEL_NAME = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
ADAPTER_MODES = {"concat", "ties", "streaming_ties_exact"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "base",
            "gold",
            "concat",
            "ties",
            "streaming_ties_exact",
            "full_sft",
            "lora16_sft",
            "code2lora",
        ],
        required=True,
    )
    parser.add_argument("--dataset-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--chunk-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--baseline-data-ready",
        type=Path,
        help="Frozen baseline-data READY manifest; required for Code2LoRA.",
    )
    parser.add_argument(
        "--adapter-control",
        choices=["correct", "wrong_repo", "remove_answer_chunks", "evidence_only"],
        default="correct",
    )
    parser.add_argument("--retrieval-jsonl", type=Path)
    parser.add_argument(
        "--retrieval-budget",
        choices=["none", "500", "1k", "2k", "8k", "oracle"],
        default=None,
        help="Legacy single-budget form; defaults to none.",
    )
    parser.add_argument(
        "--retrieval-budgets",
        nargs="+",
        choices=["none", "500", "1k", "2k", "8k", "oracle"],
        help=(
            "Evaluate one or more frozen retrieval budgets while reusing the "
            "same loaded model and repository adapter."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["open", "mcq", "mcq_permutation"],
        default=["open", "mcq", "mcq_permutation"],
    )
    parser.add_argument("--snapshot-shard-index", type=int)
    parser.add_argument("--num-snapshot-shards", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--max-new-tokens-open", type=int, default=512)
    parser.add_argument("--max-new-tokens-mcq", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.snapshot_shard_index is None:
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        args.snapshot_shard_index = int(task) if task is not None else 0
    if not 0 <= args.snapshot_shard_index < args.num_snapshot_shards:
        parser.error("--snapshot-shard-index must be within --num-snapshot-shards")
    if args.mode in ADAPTER_MODES:
        if not args.checkpoint:
            parser.error("--checkpoint is required for adapter modes")
        if not args.chunk_root:
            parser.error("--chunk-root is required for adapter modes")
    elif args.mode in {"full_sft", "lora16_sft", "code2lora"}:
        if not args.checkpoint:
            parser.error("--checkpoint is required for trained baseline modes")
        if args.mode == "code2lora" and not args.baseline_data_ready:
            parser.error("--baseline-data-ready is required for Code2LoRA")
    elif args.checkpoint:
        parser.error("--checkpoint is not used for this mode")
    if args.mode not in ADAPTER_MODES and args.adapter_control != "correct":
        parser.error("--adapter-control diagnostics require an adapter mode")
    if args.mode == "gold" and not args.repo_root:
        parser.error("--repo-root is required for gold mode")
    if args.retrieval_budget is not None and args.retrieval_budgets is not None:
        parser.error(
            "--retrieval-budget and --retrieval-budgets are mutually exclusive"
        )
    if args.retrieval_budgets is None:
        args.retrieval_budgets = [args.retrieval_budget or "none"]
    if len(set(args.retrieval_budgets)) != len(args.retrieval_budgets):
        parser.error("--retrieval-budgets must not contain duplicates")
    # Preserve the legacy scalar field in configs and summaries. Multi-budget
    # invocations use an explicit sentinel while each prediction stores the
    # concrete budget that produced it.
    args.retrieval_budget = (
        args.retrieval_budgets[0]
        if len(args.retrieval_budgets) == 1
        else "multiple"
    )
    if (
        any(budget != "none" for budget in args.retrieval_budgets)
        and not args.retrieval_jsonl
    ):
        parser.error("--retrieval-jsonl is required for retrieval conditions")
    return args


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def cuda_memory() -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "cuda_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }


def completed_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    return {
        (
            str(row["fact_id"]),
            str(row["evaluation_format"]),
            str(row.get("retrieval_budget") or "none"),
        )
        for row in read_jsonl(path)
        if row.get("status") == "done"
    }


def retrieval_maps(
    path: Path | None,
    budgets: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    requested = [budget for budget in budgets if budget != "none"]
    output: dict[str, dict[str, dict[str, Any]]] = {
        budget: {} for budget in requested
    }
    if not requested:
        return output
    if path is None:
        raise ValueError("retrieval bundles require a retrieval JSONL path")
    for row in read_jsonl(path):
        fact_id = str(row["fact_id"])
        bundles = row.get("bundles") or {}
        for budget in requested:
            if budget not in bundles:
                raise ValueError(f"{fact_id} lacks retrieval budget {budget}")
            output[budget][fact_id] = bundles[budget]
    return output


def retrieval_map(path: Path | None, budget: str) -> dict[str, dict[str, Any]]:
    """Backward-compatible helper for callers that request one budget."""
    if budget == "none":
        return {}
    return retrieval_maps(path, [budget])[budget]


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    snapshots = sorted(
        {
            (str(row["repo_id"]), str(row["commit_sha"]))
            for row in rows
        }
    )
    selected_snapshots = {
        snapshot
        for index, snapshot in enumerate(snapshots)
        if index % args.num_snapshot_shards == args.snapshot_shard_index
    }
    selected = [
        row
        for row in rows
        if (str(row["repo_id"]), str(row["commit_sha"])) in selected_snapshots
    ]
    if args.max_questions:
        selected = selected[: args.max_questions]
    return selected


def git_show(repo: Path, commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{path}"],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )
    return result.stdout


def oracle_context(row: dict[str, Any], root: Path) -> tuple[str, dict[str, Any]]:
    if bool(row.get("negative")) or not row.get("evidence_spans"):
        return "", {"oracle_policy": "no_oracle_for_negative_or_missing_evidence"}
    repository = repo_path(root, str(row["repo_id"]))
    commit = str(row["commit_sha"])
    cache: dict[str, list[str]] = {}
    parts = []
    truncated_parts = 0
    max_part_characters = 16_000
    for span in row["evidence_spans"]:
        path = str(span["path"])
        if path not in cache:
            cache[path] = git_show(repository, commit, path).splitlines()
        lines = cache[path]
        start = max(1, int(span["start_line"]) - 12)
        end = min(len(lines), int(span["end_line"]) + 12)
        if end - start + 1 <= 240:
            ranges = [(start, end)]
        else:
            ranges = [(start, start + 119), (end - 119, end)]
        for range_start, range_end in ranges:
            text = "\n".join(lines[range_start - 1 : range_end])
            # A small number of repositories contain minified/generated source
            # files with a single multi-megabyte line. A line-count cap alone
            # is therefore not a usable context bound. Keep both ends so the
            # policy is deterministic and does not depend on the question.
            if len(text) > max_part_characters:
                half = max_part_characters // 2
                text = (
                    text[:half]
                    + "\n[... oracle span character-truncated ...]\n"
                    + text[-half:]
                )
                truncated_parts += 1
            parts.append(
                f'<file path="{path}" lines="{range_start}-{range_end}">\n'
                f"{text}\n</file>"
            )
    return "\n\n".join(parts), {
        "oracle_policy": (
            "verified_evidence_spans_with_12_line_margin_and_240_line_cap_"
            "preserving_first_and_last_120_and_16000_character_part_cap"
        ),
        "oracle_paths": sorted(cache),
        "oracle_span_count": len(parts),
        "oracle_character_truncated_parts": truncated_parts,
    }


def render_mcq(mcq: dict[str, Any]) -> str:
    return "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(mcq["labels"], mcq["choices"])
    )


def render_user_content(
    row: dict[str, Any],
    evaluation_format: str,
    *,
    context: str,
    context_label: str,
) -> str:
    repository = str(row.get("prompt_repo_label") or row["repo_id"])
    commit = str(row.get("prompt_commit_label") or row["commit_sha"])
    prefix = (
        f"Repository: {repository}\nBase commit: {commit}\n\n"
    )
    if context:
        prefix += f"<{context_label}>\n{context}\n</{context_label}>\n\n"
    question = str(row["question"]).strip()
    if evaluation_format == "open":
        return (
            prefix
            + "Answer the repository question directly. Output only the answer. "
            "For requested ranked lists, output one item per line in rank order. "
            "If the question's premise is false, output UNKNOWN.\n\n"
            + question
        )
    mcq_key = "mcq_permutation" if evaluation_format == "mcq_permutation" else "mcq"
    mcq = row[mcq_key]
    if mcq is None:
        raise ValueError(f"{row['fact_id']} does not define {evaluation_format}")
    choice_instruction = (
        "Choose the single complete candidate answer set. "
        if mcq.get("kind") == "candidate_set"
        else "Choose the single best answer. "
    )
    return (
        prefix
        + choice_instruction
        + "Output only A, B, C, or D.\n\n"
        + question
        + "\n\n"
        + render_mcq(mcq)
    )


def render_option_scoring_content(
    row: dict[str, Any],
    *,
    context: str,
    context_label: str,
) -> str:
    """Render a choice-independent prompt for option-text likelihood scoring."""
    prefix = (
        f"Repository: {row.get('prompt_repo_label') or row['repo_id']}\n"
        f"Base commit: {row.get('prompt_commit_label') or row['commit_sha']}\n\n"
    )
    if context:
        prefix += f"<{context_label}>\n{context}\n</{context_label}>\n\n"
    answer_instruction = (
        "Return the complete answer set as a compact JSON array. "
        if row.get("mcq", {}).get("kind") == "candidate_set"
        else "Output only the answer. "
    )
    return (
        prefix
        + "Answer the repository question directly. "
        + answer_instruction
        + "If the question's premise is false, output UNKNOWN.\n\n"
        + str(row["question"]).strip()
        + "\n\nAnswer:"
    )


def tokenize_prompt(tokenizer: Any, user_content: str) -> list[int]:
    result = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content},
        ],
        tokenize=True,
        add_special_tokens=False,
        add_generation_prompt=True,
        return_attention_mask=False,
    )
    if hasattr(result, "keys"):
        result = result["input_ids"]
    if result and isinstance(result[0], list):
        result = result[0]
    return [int(value) for value in result]


def generation_eos(model: Any, tokenizer: Any) -> int | list[int] | None:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    return value if value is not None else tokenizer.eos_token_id


def generate(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
) -> tuple[str, int, float]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    first_token_time: list[float | None] = [None]

    class FirstTokenTimer(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            if first_token_time[0] is None:
                first_token_time[0] = time.time()
            return False

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    mask = torch.ones_like(input_ids)
    started = time.time()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=generation_eos(model, tokenizer),
            stopping_criteria=StoppingCriteriaList([FirstTokenTimer()]),
        )
    finished = time.time()
    first = first_token_time[0] or finished
    generate.last_profile = {
        "time_to_first_token_seconds": first - started,
        "decode_after_first_token_seconds": max(0.0, finished - first),
    }
    completion = generated[0, input_ids.shape[1] :].tolist()
    response = tokenizer.decode(
        completion,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return response, len(completion), finished - started


generate.last_profile = {
    "time_to_first_token_seconds": None,
    "decode_after_first_token_seconds": None,
}


def candidate_logprob(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    candidate: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    candidate_ids = tokenizer(
        candidate,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    if not candidate_ids:
        return {
            "candidate": candidate,
            "token_count": 0,
            "logprob_sum": float("-inf"),
            "logprob_mean": float("-inf"),
        }
    all_ids = prompt_ids + [int(value) for value in candidate_ids]
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        ).logits
    start = len(prompt_ids) - 1
    selected = logits[0, start : start + len(candidate_ids), :]
    targets = input_ids[0, len(prompt_ids) :]
    log_probs = functional.log_softmax(selected.float(), dim=-1)
    values = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    total = float(values.sum().item())
    del input_ids, logits, selected, targets, log_probs, values
    return {
        "candidate": candidate,
        "token_count": len(candidate_ids),
        "sequence_token_count": len(all_ids),
        "logprob_sum": total,
        "logprob_mean": total / len(candidate_ids),
    }


def score_mcq_options(
    model: Any,
    tokenizer: Any,
    scoring_prompt_ids: list[int],
    mcq: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    scores = [
        candidate_logprob(model, tokenizer, scoring_prompt_ids, str(choice))
        for choice in mcq["choices"]
    ]
    sum_values = [float(item["logprob_sum"]) for item in scores]
    mean_values = [float(item["logprob_mean"]) for item in scores]
    probabilities = softmax(sum_values)
    normalized_probabilities = softmax(mean_values)
    predicted_index = max(range(4), key=lambda index: sum_values[index])
    normalized_predicted_index = max(
        range(4),
        key=lambda index: mean_values[index],
    )
    gold_index = int(mcq["correct_index"])
    return {
        "option_scoring_rule": "conditional_logprob_sum",
        "option_scores": scores,
        "option_score_seconds": time.time() - started,
        "option_score_tokens": sum(int(item["token_count"]) for item in scores),
        "option_scoring_prompt_tokens": len(scoring_prompt_ids),
        "option_score_sequence_tokens": sum(
            int(item["sequence_token_count"]) for item in scores
        ),
        "likelihood_predicted_index": predicted_index,
        "likelihood_predicted_label": str(mcq["labels"][predicted_index]),
        "likelihood_correct": predicted_index == gold_index,
        "choice_probabilities": probabilities,
        "gold_choice_probability": probabilities[gold_index],
        "length_normalized_likelihood_predicted_index": (
            normalized_predicted_index
        ),
        "length_normalized_likelihood_predicted_label": str(
            mcq["labels"][normalized_predicted_index]
        ),
        "length_normalized_likelihood_correct": (
            normalized_predicted_index == gold_index
        ),
        "length_normalized_choice_probabilities": normalized_probabilities,
        "length_normalized_gold_choice_probability": (
            normalized_probabilities[gold_index]
        ),
        "brier_score": sum(
            (probability - float(index == gold_index)) ** 2
            for index, probability in enumerate(probabilities)
        ),
    }


def load_base_model(local_files_only: bool) -> tuple[Any, Any]:
    import torch
    from ctx_to_lora.model_loading import get_model_and_tokenizer

    model, tokenizer = get_model_and_tokenizer(
        MODEL_NAME,
        train=False,
        requires_grad=False,
        model_revision=MODEL_REVISION,
        use_flash_attn=False,
        model_kwargs={
            "revision": MODEL_REVISION,
            "local_files_only": local_files_only,
            "attn_implementation": "sdpa",
        },
        tokenizer_kwargs={
            "revision": MODEL_REVISION,
            "local_files_only": local_files_only,
        },
        device="cuda",
        dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def load_trained_baseline(
    checkpoint: Path,
    mode: str,
    local_files_only: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer
    from repotune_issuefix.code2lora_gemma import (
        Code2LoRAHead,
        discover_module_types_and_dims,
        get_module_specs,
        load_gemma4_model,
        replace_with_lora,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = load_gemma4_model(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device="cuda",
        local_files_only=local_files_only,
    )
    metadata: dict[str, Any] = {}
    if mode == "lora16_sft":
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=(
                    r"^model\.language_model\.layers\..*\."
                    r"(q_proj|k_proj|v_proj|o_proj|up_proj|gate_proj|down_proj)$"
                ),
            ),
        )
        state_path = checkpoint / "trainable_model.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        expected = {
            name for name, _value in model.named_parameters() if "lora_" in name
        }
        if set(state) != expected:
            raise ValueError(
                "Rank-16 LoRA evaluation checkpoint does not exactly cover "
                f"the configured adapters: saved={len(state)} expected={len(expected)}"
            )
        _missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected rank-16 LoRA checkpoint keys: {unexpected[:8]}")
        metadata["trainable_parameters"] = sum(value.numel() for value in state.values())
    elif mode == "full_sft":
        state_path = checkpoint / "trainable_model.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        expected = {
            name
            for name, _value in model.named_parameters()
            if name.startswith("model.language_model.") or name.startswith("lm_head.")
        }
        if set(state) != expected:
            raise ValueError(
                "Full-SFT evaluation checkpoint does not exactly cover the text model: "
                f"saved={len(state)} expected={len(expected)}"
            )
        _missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected full-SFT checkpoint keys: {unexpected[:8]}")
        metadata["trainable_parameters"] = sum(value.numel() for value in state.values())
    elif mode == "code2lora":
        target_modules = (
            "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"
        )
        specs = get_module_specs(model, target_modules)
        type_dims = discover_module_types_and_dims(specs)
        replace_with_lora(model, specs, rank=8, alpha=16)
        head = Code2LoRAHead(2048, type_dims, hidden_dim=1024, rank=8).to("cuda")
        head.load_state_dict(
            torch.load(checkpoint / "code2lora_head.pt", map_location="cpu", weights_only=True)
        )
        head.eval()
        metadata.update({
            "code2lora_head": head,
            "code2lora_specs": specs,
            "trainable_parameters": sum(value.numel() for value in head.parameters()),
        })
    else:
        raise ValueError(f"Unsupported trained baseline mode {mode}")
    model.eval()
    return model, tokenizer, metadata


def checkpoint_size(path: Path | None) -> int:
    if path is None:
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(value.stat().st_size for value in path.rglob("*") if value.is_file())


def load_modulated_model(
    checkpoint: Path,
    mode: str,
    local_files_only: bool,
) -> tuple[Any, Any, Any]:
    import torch
    from ctx_to_lora.model_loading import get_tokenizer
    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state,
        train=False,
        base_model_kwargs={
            "revision": MODEL_REVISION,
            "local_files_only": local_files_only,
            "attn_implementation": "sdpa",
        },
        use_flash_attn=False,
        use_sequence_packing=False,
    )
    model.eval()
    model.patch_lora_forward()
    actual = str(model.hypernet.repo_merger.config.method.value)
    if actual != mode:
        raise ValueError(f"Checkpoint merger is {actual}, requested {mode}")
    ctx_tokenizer = get_tokenizer(
        model.ctx_encoder.base_model.name_or_path,
        tokenizer_kwargs={
            "revision": MODEL_REVISION,
            "local_files_only": local_files_only,
        },
        train=False,
    )
    tokenizer = get_tokenizer(
        model.base_model.name_or_path,
        tokenizer_kwargs={
            "revision": MODEL_REVISION,
            "local_files_only": local_files_only,
        },
        train=False,
    )
    return model, tokenizer, ctx_tokenizer


def chunk_store_index(root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for audit_path in sorted((root / "repositories").glob("*/audit.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") not in {"complete", "complete_with_errors"}:
            continue
        output[str(audit["repo_id"])] = audit_path.parent
    return output


def snapshot_chunks(
    store: Path,
    commit: str,
) -> tuple[list[str], list[str], list[int]]:
    snapshot_rows = pq.read_table(
        store / "snapshots.parquet",
        filters=[("commit_sha", "=", commit)],
        memory_map=True,
    ).to_pylist()
    if not snapshot_rows:
        raise ValueError(f"No chunk snapshot for {store.name}@{commit}")
    snapshot_rows.sort(key=lambda row: int(row["chunk_index"]))
    chunk_ids = [str(row["chunk_id"]) for row in snapshot_rows]
    chunk_table = pq.read_table(
        store / "chunks.parquet",
        columns=["chunk_id", "payload_text"],
        filters=[("chunk_id", "in", chunk_ids)],
        memory_map=True,
    ).to_pylist()
    payloads = {str(row["chunk_id"]): str(row["payload_text"]) for row in chunk_table}
    if set(payloads) != set(chunk_ids):
        raise ValueError(f"Missing chunks for {store.name}@{commit}")
    return (
        chunk_ids,
        [payloads[chunk_id] for chunk_id in chunk_ids],
        [int(row["context_token_count"]) for row in snapshot_rows],
    )


def move_lora_tree(tree: dict[str, dict[str, Any]], device: Any) -> dict[str, dict[str, Any]]:
    return {
        module: {
            name: value.detach().to(device=device, non_blocking=True)
            for name, value in factors.items()
        }
        for module, factors in tree.items()
    }


def lora_tree_nbytes(tree: dict[str, dict[str, Any]]) -> int:
    return sum(
        int(value.numel()) * int(value.element_size())
        for factors in tree.values()
        for value in factors.values()
        if hasattr(value, "numel") and hasattr(value, "element_size")
    )


def concatenate_trees(
    trees: Sequence[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    import torch

    return {
        module: {
            name: torch.cat([tree[module][name] for tree in trees], dim=0)
            for name in trees[0][module]
        }
        for module in trees[0]
    }


def encode_repository_chunks(
    model: Any,
    ctx_tokenizer: Any,
    payloads: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from ctx_to_lora.data.processing import tokenize_ctx_text

    records = []
    token_counts = []
    encode_seconds = []
    need_latents = model.hypernet.repo_merger.requires_chunk_latents
    for payload in payloads:
        ids = tokenize_ctx_text({"context": [payload]}, ctx_tokenizer)["ctx_ids"][0]
        ctx_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
        ctx_mask = torch.ones_like(ctx_ids)
        started = time.time()
        with torch.inference_mode():
            generated = model.generate_weights(
                ctx_ids,
                ctx_mask,
                return_latents=need_latents,
            )
        encode_seconds.append(time.time() - started)
        token_counts.append(len(ids))
        records.append(
            {
                "tree": move_lora_tree(generated[0], "cpu"),
                "latent": generated[2].detach().cpu() if need_latents else None,
            }
        )
        del ctx_ids, ctx_mask, generated
        torch.cuda.empty_cache()
    return records, {
        "context_tokens": sum(token_counts),
        "context_chunk_token_counts_observed": token_counts,
        "context_encode_seconds": encode_seconds,
        "context_encode_seconds_total": sum(encode_seconds),
        "generated_chunk_lora_bytes": sum(
            lora_tree_nbytes(record["tree"]) for record in records
        ),
    }


def merge_encoded_chunks(
    model: Any,
    records: Sequence[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    import torch

    raw = move_lora_tree(
        concatenate_trees([record["tree"] for record in records]),
        model.device,
    )
    latents = (
        torch.cat([record["latent"] for record in records], dim=0).to(model.device)
        if records and records[0]["latent"] is not None
        else None
    )
    merge_started = time.time()
    with torch.inference_mode():
        merged = model.merge_repository_loras(
            raw,
            torch.tensor([len(records)], dtype=torch.int32, device=model.device),
            chunk_latents=latents,
        )
    merge_seconds = time.time() - merge_started
    apply_started = time.time()
    model.apply_lora_to_layers(
        model.base_model,
        model.hypernet.layer_indices,
        merged,
        torch.ones(1, dtype=torch.int32, device=model.device),
    )
    adapter_apply_seconds = time.time() - apply_started
    rank = int(next(iter(merged.values()))["A"].shape[2])
    diagnostics = {}
    for key, value in getattr(model.hypernet.repo_merger, "last_diagnostics", {}).items():
        if hasattr(value, "numel") and value.numel() == 1:
            diagnostics[key] = float(value.item())
        else:
            diagnostics[key] = str(value)
    return merged, {
        "merge_seconds": merge_seconds,
        "adapter_apply_seconds": adapter_apply_seconds,
        "merged_lora_rank": rank,
        "premerge_lora_bytes": lora_tree_nbytes(raw),
        "merged_lora_bytes": lora_tree_nbytes(merged),
        "merge_diagnostics": diagnostics,
    }


def materialize_adapter(
    model: Any,
    ctx_tokenizer: Any,
    payloads: Sequence[str],
) -> tuple[Any, dict[str, Any]]:
    records, encode_metadata = encode_repository_chunks(
        model,
        ctx_tokenizer,
        payloads,
    )
    merged, merge_metadata = merge_encoded_chunks(model, records)
    return merged, {**encode_metadata, **merge_metadata}


def size_match_wrong_repository_chunks(
    chunk_ids: Sequence[str],
    payloads: Sequence[str],
    token_counts: Sequence[int],
    target_count: int,
) -> tuple[list[str], list[str], list[int], dict[str, Any]]:
    """Match composition size while retaining only wrong-repository chunks."""
    if not chunk_ids or target_count <= 0:
        raise ValueError("wrong-repository size matching needs nonempty chunks")
    source_count = len(chunk_ids)
    if source_count >= target_count:
        # Evenly spaced indices preserve coverage across the canonical
        # repository ordering instead of keeping only a path-sorted prefix.
        indexes = [
            min(source_count - 1, (index * source_count) // target_count)
            for index in range(target_count)
        ]
        policy = (
            "identity"
            if source_count == target_count
            else "deterministic_even_downsample"
        )
    else:
        indexes = [index % source_count for index in range(target_count)]
        policy = "deterministic_cycle_no_large_enough_wrong_repository"
    return (
        [str(chunk_ids[index]) for index in indexes],
        [str(payloads[index]) for index in indexes],
        [int(token_counts[index]) for index in indexes],
        {
            "control_adapter_source_chunk_count": source_count,
            "control_adapter_size_match_policy": policy,
            "control_adapter_repeated_chunk_references": (
                target_count - len(set(indexes))
            ),
            "control_adapter_exact_chunk_count_match": len(indexes)
            == target_count,
        },
    )


def evidence_chunk_mask(
    row: dict[str, Any],
    payloads: Sequence[str],
) -> list[bool]:
    paths = {
        str(span["path"])
        for span in row.get("evidence_spans") or []
        if span.get("path")
    }
    paths.update(str(path) for path in row.get("evidence_paths") or [] if path)
    markers = [f'<<<FILE path="{path}">>>' for path in sorted(paths)]
    return [any(marker in payload for marker in markers) for payload in payloads]


def record(
    *,
    args: argparse.Namespace,
    retrieval_budget: str,
    row: dict[str, Any],
    evaluation_format: str,
    response: str,
    completion_tokens: int,
    prompt_tokens: int,
    generation_seconds: float,
    context_metadata: dict[str, Any],
    option_metadata: dict[str, Any],
    adapter_metadata: dict[str, Any],
) -> dict[str, Any]:
    mcq_key = "mcq_permutation" if evaluation_format == "mcq_permutation" else "mcq"
    mcq = row.get(mcq_key) if evaluation_format != "open" else None
    predicted_label = extract_mcq_label(response) if mcq else ""
    generated_correct = (
        predicted_label == str(mcq["correct_label"]) if mcq else None
    )
    gold = str(row["gold_answer"])
    return {
        "status": "done",
        "failure": False,
        "mode": args.mode,
        "retrieval_budget": retrieval_budget,
        "evaluation_format": evaluation_format,
        "fact_id": str(row["fact_id"]),
        "benchmark_id": str(row.get("benchmark_id") or row["fact_id"]),
        "split": str(row["split"]),
        "repo_id": str(row["repo_id"]),
        "commit_sha": str(row["commit_sha"]),
        "family": str(row["family"]),
        "subtype": str(row["subtype"]),
        "answer_type": str(row["answer_type"]),
        "negative": bool(row.get("negative")),
        "question": str(row["question"]),
        "gold_answer": gold,
        "gold_answers": row.get("gold_answers"),
        "prediction": response,
        "normalized_prediction": normalize_answer(response),
        "normalized_gold_answer": normalize_answer(gold),
        "exact_match_normalized": (
            normalize_answer(response) == normalize_answer(gold)
            if evaluation_format == "open"
            else None
        ),
        "token_f1": (
            token_f1(response, gold) if evaluation_format == "open" else None
        ),
        "generated_label": predicted_label or None,
        "correct_label": str(mcq["correct_label"]) if mcq else None,
        "mcq_kind": str(mcq.get("kind") or "single_choice") if mcq else None,
        "generated_label_correct": generated_correct,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "generation_seconds": generation_seconds,
        "tokens_per_second": completion_tokens / max(generation_seconds, 1e-9),
        "output_truncated": completion_tokens
        >= (
            args.max_new_tokens_open
            if evaluation_format == "open"
            else args.max_new_tokens_mcq
        ),
        **context_metadata,
        **option_metadata,
        **adapter_metadata,
        **cuda_memory(),
    }


def grouped_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    output = {}
    for name, values in sorted(groups.items()):
        open_values = [row for row in values if row["evaluation_format"] == "open"]
        mcq_values = [row for row in values if row["evaluation_format"] == "mcq"]
        output[name] = {
            "rows": len(values),
            "open_rows": len(open_values),
            "open_exact_match": (
                sum(bool(row["exact_match_normalized"]) for row in open_values)
                / len(open_values)
                if open_values
                else None
            ),
            "mcq_rows": len(mcq_values),
            "mcq_generated_accuracy": (
                sum(bool(row["generated_label_correct"]) for row in mcq_values)
                / len(mcq_values)
                if mcq_values
                else None
            ),
            "mcq_likelihood_accuracy": (
                sum(bool(row["likelihood_correct"]) for row in mcq_values)
                / len(mcq_values)
                if mcq_values
                else None
            ),
        }
    return output


def write_summary(args: argparse.Namespace, selected_rows: int) -> None:
    predictions_path = args.output_dir / "predictions.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    predictions = (
        [
            row
            for row in read_jsonl(predictions_path)
            if row.get("status") == "done"
        ]
        if predictions_path.exists()
        else []
    )
    completed = {
        (
            str(row["fact_id"]),
            str(row["evaluation_format"]),
            str(row.get("retrieval_budget") or "none"),
        )
        for row in predictions
    }
    errors = read_jsonl(errors_path) if errors_path.exists() else []
    unresolved = [
        row
        for row in errors
        if (
            str(row.get("fact_id")),
            str(row.get("evaluation_format")),
            str(row.get("retrieval_budget") or "none"),
        )
        not in completed
    ]
    open_rows = [row for row in predictions if row["evaluation_format"] == "open"]
    mcq_rows = [row for row in predictions if row["evaluation_format"] == "mcq"]
    summary = {
        "format": "doc_to_lora_broad_eval_run_summary_v1",
        "mode": args.mode,
        "retrieval_budget": args.retrieval_budget,
        "retrieval_budgets": args.retrieval_budgets,
        "dataset_jsonl": str(args.dataset_jsonl),
        "selected_fact_rows": selected_rows,
        "requested_formats": args.formats,
        "completed_records": len(predictions),
        "unresolved_errors": len(unresolved),
        "open_exact_match": (
            sum(bool(row["exact_match_normalized"]) for row in open_rows)
            / len(open_rows)
            if open_rows
            else None
        ),
        "open_token_f1": (
            sum(float(row["token_f1"]) for row in open_rows) / len(open_rows)
            if open_rows
            else None
        ),
        "mcq_generated_accuracy": (
            sum(bool(row["generated_label_correct"]) for row in mcq_rows)
            / len(mcq_rows)
            if mcq_rows
            else None
        ),
        "mcq_likelihood_accuracy": (
            sum(bool(row["likelihood_correct"]) for row in mcq_rows)
            / len(mcq_rows)
            if mcq_rows
            else None
        ),
        "mcq_mean_gold_probability": (
            sum(float(row["gold_choice_probability"]) for row in mcq_rows)
            / len(mcq_rows)
            if mcq_rows
            else None
        ),
        "mcq_mean_brier": (
            sum(float(row["brier_score"]) for row in mcq_rows) / len(mcq_rows)
            if mcq_rows
            else None
        ),
        "by_family": grouped_summary(predictions, "family"),
        "by_subtype": grouped_summary(predictions, "subtype"),
        "by_repository": grouped_summary(predictions, "repo_id"),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)


def main() -> int:
    args = parse_args()
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    dataset_rows = read_jsonl(args.dataset_jsonl)
    dataset_snapshot_keys = sorted(
        {
            (str(row["repo_id"]), str(row["commit_sha"]))
            for row in dataset_rows
        }
    )
    rows = select_rows(dataset_rows, args)
    if not rows:
        raise ValueError("No rows selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    if args.overwrite:
        predictions_path.unlink(missing_ok=True)
        errors_path.unlink(missing_ok=True)
    done = completed_keys(predictions_path)
    retrieval = retrieval_maps(
        args.retrieval_jsonl,
        args.retrieval_budgets,
    )
    atomic_write_json(
        args.output_dir / "run_config.json",
        {
            "format": "doc_to_lora_broad_eval_run_config_v1",
            "mode": args.mode,
            "dataset_jsonl": str(args.dataset_jsonl.resolve()),
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "baseline_data_ready": (
                str(args.baseline_data_ready.resolve())
                if args.baseline_data_ready else None
            ),
            "repo_root": str(args.repo_root.resolve()) if args.repo_root else None,
            "chunk_root": str(args.chunk_root.resolve()) if args.chunk_root else None,
            "retrieval_jsonl": (
                str(args.retrieval_jsonl.resolve()) if args.retrieval_jsonl else None
            ),
            "retrieval_budget": args.retrieval_budget,
            "retrieval_budgets": args.retrieval_budgets,
            "adapter_control": args.adapter_control,
            "formats": args.formats,
            "selected_rows": len(rows),
            "snapshot_shard_index": args.snapshot_shard_index,
            "num_snapshot_shards": args.num_snapshot_shards,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )

    model_started = time.time()
    adapter_model = None
    ctx_tokenizer = None
    trained_metadata: dict[str, Any] = {}
    code2lora_embeddings: dict[tuple[str, str], Any] = {}
    if args.mode in ADAPTER_MODES:
        adapter_model, tokenizer, ctx_tokenizer = load_modulated_model(
            args.checkpoint,
            args.mode,
            args.local_files_only,
        )
        answer_model = adapter_model.base_model
        stores = chunk_store_index(args.chunk_root)
    elif args.mode in {"full_sft", "lora16_sft", "code2lora"}:
        answer_model, tokenizer, trained_metadata = load_trained_baseline(
            args.checkpoint, args.mode, args.local_files_only
        )
        stores = {}
        if args.mode == "code2lora":
            import torch
            from repotune_issuefix.repoqa_baselines import sha256_file
            embedding_ready = json.loads(args.baseline_data_ready.read_text())
            embedding_path = (
                args.baseline_data_ready.parent
                / "code2lora_snapshot_embeddings.parquet"
            )
            if (
                sha256_file(embedding_path)
                != embedding_ready["code2lora_embeddings"]["sha256"]
            ):
                raise ValueError("Code2LoRA evaluation embedding hash mismatch")
            for value in pq.read_table(embedding_path, memory_map=True).to_pylist():
                code2lora_embeddings[(str(value["repo_id"]), str(value["commit_sha"]))] = (
                    torch.tensor(value["repo_state_embedding"], dtype=torch.float32)
                )
    else:
        answer_model, tokenizer = load_base_model(args.local_files_only)
        stores = {}
    model_load_seconds = time.time() - model_started
    loaded_checkpoint_bytes = checkpoint_size(args.checkpoint)

    # Most benchmarks materialize the adapter for the same snapshot named in
    # the question.  Snapshot-swap controls deliberately keep the question
    # fixed while selecting a different adapter from the same repository.
    # Keeping the override in the frozen dataset makes that causal control
    # explicit and auditable rather than changing prompt text at inference.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("adapter_repo_id") or row["repo_id"]),
                str(row.get("adapter_commit_sha") or row["commit_sha"]),
            )
        ].append(row)
    processed = 0
    from ctx_to_lora.modeling.lora_layer import clear_lora_from_layers
    import torch

    try:
        for group_index, ((repo_id, commit), group_rows) in enumerate(
            sorted(grouped.items()),
            start=1,
        ):
            adapter_metadata: dict[str, Any] = {
                "model_load_seconds": model_load_seconds,
                "snapshot_repo_id": repo_id,
                "snapshot_commit_sha": commit,
                "checkpoint_size_bytes": loaded_checkpoint_bytes,
            }
            if "trainable_parameters" in trained_metadata:
                adapter_metadata["trainable_parameters"] = int(
                    trained_metadata["trainable_parameters"]
                )
            encoded_chunks: list[dict[str, Any]] | None = None
            if args.mode == "code2lora":
                from repotune_issuefix.code2lora_gemma import inject_lora_weights

                snapshot_key = (repo_id, commit)
                if snapshot_key not in code2lora_embeddings:
                    raise ValueError(f"No Code2LoRA embedding for {snapshot_key}")
                torch.cuda.reset_peak_memory_stats()
                adapter_started = time.time()
                with torch.inference_mode():
                    head_out = trained_metadata["code2lora_head"](
                        code2lora_embeddings[snapshot_key]
                        .to(answer_model.device)
                        .unsqueeze(0)
                    )
                    inject_lora_weights(
                        answer_model,
                        trained_metadata["code2lora_specs"],
                        head_out,
                        batch_index=0,
                    )
                adapter_metadata.update(
                    {
                        "code2lora_adapter_generation_seconds": time.time()
                        - adapter_started,
                        "code2lora_embedding_bytes": int(
                            code2lora_embeddings[snapshot_key].numel()
                            * code2lora_embeddings[snapshot_key].element_size()
                        ),
                        **cuda_memory(),
                    }
                )
            if adapter_model is not None:
                torch.cuda.reset_peak_memory_stats()
                if repo_id not in stores:
                    raise ValueError(f"No chunk store for {repo_id}")
                chunk_ids, payloads, manifest_token_counts = snapshot_chunks(
                    stores[repo_id],
                    commit,
                )
                materialization_payloads = payloads
                materialization_ids = chunk_ids
                if args.adapter_control == "wrong_repo":
                    candidates = []
                    for candidate_repo, candidate_commit in dataset_snapshot_keys:
                        if candidate_repo == repo_id:
                            continue
                        candidate_ids, candidate_payloads, candidate_counts = snapshot_chunks(
                            stores[candidate_repo],
                            candidate_commit,
                        )
                        candidates.append(
                            (
                                abs(len(candidate_ids) - len(chunk_ids)),
                                candidate_repo,
                                candidate_commit,
                                candidate_ids,
                                candidate_payloads,
                                candidate_counts,
                            )
                        )
                    eligible_candidates = [
                        value
                        for value in candidates
                        if len(value[3]) >= len(chunk_ids)
                    ]
                    (
                        _distance,
                        control_repo,
                        control_commit,
                        materialization_ids,
                        materialization_payloads,
                        manifest_token_counts,
                    ) = min(
                        eligible_candidates or candidates,
                        key=lambda value: (value[0], value[1], value[2]),
                    )
                    (
                        materialization_ids,
                        materialization_payloads,
                        manifest_token_counts,
                        size_match_metadata,
                    ) = size_match_wrong_repository_chunks(
                        materialization_ids,
                        materialization_payloads,
                        manifest_token_counts,
                        len(chunk_ids),
                    )
                    adapter_metadata.update(
                        {
                            "control_adapter_repo_id": control_repo,
                            "control_adapter_commit_sha": control_commit,
                            "control_adapter_target_chunk_count": len(chunk_ids),
                            **size_match_metadata,
                        }
                    )
                if args.adapter_control in {"correct", "wrong_repo"}:
                    _merged, materialization = materialize_adapter(
                        adapter_model,
                        ctx_tokenizer,
                        materialization_payloads,
                    )
                    adapter_metadata.update(materialization)
                else:
                    encoded_chunks, encode_metadata = encode_repository_chunks(
                        adapter_model,
                        ctx_tokenizer,
                        payloads,
                    )
                    adapter_metadata.update(
                        {
                            f"repository_{key}": value
                            for key, value in encode_metadata.items()
                        }
                    )
                adapter_metadata.update(
                    {
                        "adapter_control": args.adapter_control,
                        "num_chunks": len(materialization_ids),
                        "selected_chunk_ids": materialization_ids,
                        "manifest_context_token_counts": manifest_token_counts,
                    }
                )
            for row in group_rows:
                fact_id = str(row["fact_id"])
                row_adapter_metadata = dict(adapter_metadata)
                if (
                    adapter_model is not None
                    and args.adapter_control
                    in {"remove_answer_chunks", "evidence_only"}
                ):
                    clear_lora_from_layers(
                        adapter_model.base_model,
                        adapter_model.hypernet.layer_indices,
                    )
                    mask = evidence_chunk_mask(row, payloads)
                    if args.adapter_control == "remove_answer_chunks":
                        selected_indexes = [
                            index for index, contains in enumerate(mask) if not contains
                        ]
                    else:
                        selected_indexes = [
                            index for index, contains in enumerate(mask) if contains
                        ]
                    if not selected_indexes:
                        raise ValueError(
                            f"{fact_id} has no chunks for {args.adapter_control}"
                        )
                    _merged, materialization = merge_encoded_chunks(
                        adapter_model,
                        [encoded_chunks[index] for index in selected_indexes],
                    )
                    row_adapter_metadata.update(materialization)
                    row_adapter_metadata.update(
                        {
                            "num_chunks": len(selected_indexes),
                            "selected_chunk_ids": [
                                chunk_ids[index] for index in selected_indexes
                            ],
                            "answer_bearing_chunk_count": sum(mask),
                        }
                    )
                for retrieval_budget in args.retrieval_budgets:
                    context = ""
                    context_metadata: dict[str, Any] = {
                        "context_mode": retrieval_budget,
                        "context_tokens_in_prompt": 0,
                    }
                    if args.mode == "gold":
                        context, context_metadata = oracle_context(
                            row,
                            args.repo_root,
                        )
                        context_metadata["context_mode"] = "oracle_spans"
                        context_label = "oracle_repository_evidence"
                    elif retrieval_budget != "none":
                        bundle = retrieval[retrieval_budget][fact_id]
                        context = str(bundle["text"])
                        context_metadata = {
                            "context_mode": f"bm25_{retrieval_budget}",
                            "context_tokens_in_prompt": int(bundle["token_count"]),
                            "retrieval_paths": bundle["paths"],
                            "retrieval_chunk_count": int(bundle["chunk_count"]),
                            "retrieval_seconds": float(
                                bundle.get("retrieval_seconds") or 0.0
                            ),
                        }
                        context_label = "retrieved_repository_evidence"
                    else:
                        context_label = "repository_evidence"
                    for evaluation_format in args.formats:
                        if evaluation_format == "mcq_permutation" and row.get(
                            "mcq_permutation"
                        ) is None:
                            continue
                        key = (
                            fact_id,
                            evaluation_format,
                            retrieval_budget,
                        )
                        if key in done:
                            continue
                        try:
                            user_content = render_user_content(
                                row,
                                evaluation_format,
                                context=context,
                                context_label=context_label,
                            )
                            prompt_ids = tokenize_prompt(tokenizer, user_content)
                            max_new = (
                                args.max_new_tokens_open
                                if evaluation_format == "open"
                                else args.max_new_tokens_mcq
                            )
                            (
                                response,
                                completion_tokens,
                                generation_seconds,
                            ) = generate(
                                answer_model,
                                tokenizer,
                                prompt_ids,
                                max_new,
                            )
                            option_metadata: dict[str, Any] = dict(
                                generate.last_profile
                            )
                            if evaluation_format != "open":
                                mcq_key = (
                                    "mcq_permutation"
                                    if evaluation_format == "mcq_permutation"
                                    else "mcq"
                                )
                                scoring_prompt_ids = tokenize_prompt(
                                    tokenizer,
                                    render_option_scoring_content(
                                        row,
                                        context=context,
                                        context_label=context_label,
                                    ),
                                )
                                option_metadata.update(
                                    score_mcq_options(
                                        answer_model,
                                        tokenizer,
                                        scoring_prompt_ids,
                                        row[mcq_key],
                                    )
                                )
                            elif row["family"] == "code_continuation":
                                gold_score = candidate_logprob(
                                    answer_model,
                                    tokenizer,
                                    prompt_ids,
                                    str(row["gold_answer"]),
                                )
                                option_metadata.update(
                                    {
                                        "gold_answer_logprob_sum": gold_score[
                                            "logprob_sum"
                                        ],
                                        "gold_answer_logprob_mean": gold_score[
                                            "logprob_mean"
                                        ],
                                        "gold_answer_token_count": gold_score[
                                            "token_count"
                                        ],
                                    }
                                )
                            if context:
                                recorded_context_tokens = int(
                                    context_metadata.get(
                                        "context_tokens_in_prompt"
                                    )
                                    or 0
                                )
                                if not recorded_context_tokens:
                                    recorded_context_tokens = len(
                                        tokenizer(
                                            context,
                                            add_special_tokens=False,
                                            return_attention_mask=False,
                                        )["input_ids"]
                                    )
                                context_metadata[
                                    "context_tokens_in_prompt"
                                ] = recorded_context_tokens
                            else:
                                context_metadata[
                                    "context_tokens_in_prompt"
                                ] = 0
                            result = record(
                                args=args,
                                retrieval_budget=retrieval_budget,
                                row=row,
                                evaluation_format=evaluation_format,
                                response=response,
                                completion_tokens=completion_tokens,
                                prompt_tokens=len(prompt_ids),
                                generation_seconds=generation_seconds,
                                context_metadata=context_metadata,
                                option_metadata=option_metadata,
                                adapter_metadata=row_adapter_metadata,
                            )
                            append_jsonl(predictions_path, result)
                            done.add(key)
                            processed += 1
                            if (
                                args.progress_every
                                and processed % args.progress_every == 0
                            ):
                                print(
                                    json.dumps(
                                        {
                                            "status": "progress",
                                            "mode": args.mode,
                                            "processed": processed,
                                            "fact_id": fact_id,
                                            "evaluation_format": evaluation_format,
                                            "retrieval_budget": retrieval_budget,
                                            "repo_id": repo_id,
                                        },
                                        sort_keys=True,
                                    ),
                                    flush=True,
                                )
                        except Exception as exc:
                            append_jsonl(
                                errors_path,
                                {
                                    "status": "error",
                                    "fact_id": fact_id,
                                    "evaluation_format": evaluation_format,
                                    "retrieval_budget": retrieval_budget,
                                    "repo_id": repo_id,
                                    "commit_sha": commit,
                                    "error": repr(exc),
                                    "time_utc": time.strftime(
                                        "%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime(),
                                    ),
                                },
                            )
                            write_summary(args, len(rows))
                            raise
            if adapter_model is not None:
                clear_lora_from_layers(
                    adapter_model.base_model,
                    adapter_model.hypernet.layer_indices,
                )
                torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "status": "snapshot_complete",
                        "mode": args.mode,
                        "snapshot_index": group_index,
                        "snapshot_count": len(grouped),
                        "repo_id": repo_id,
                        "commit_sha": commit,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    except torch.cuda.OutOfMemoryError as exc:
        # Natural OOD length evaluation is expected to discover hard method
        # limits. Preserve the complete paired design by recording every
        # unattempted rendering as an incorrect, explicitly failed observation
        # instead of silently dropping the largest repository.
        if adapter_model is not None:
            clear_lora_from_layers(
                adapter_model.base_model,
                adapter_model.hypernet.layer_indices,
            )
        torch.cuda.empty_cache()
        failure_text = repr(exc)
        failure_adapter_metadata = {
            "model_load_seconds": model_load_seconds,
            "adapter_control": args.adapter_control,
            "reported_method_limit": "cuda_out_of_memory",
        }
        append_jsonl(
            errors_path,
            {
                "status": "reported_method_limit",
                "repo_id": repo_id,
                "commit_sha": commit,
                "error": failure_text,
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        for failure_row in rows:
            failure_fact_id = str(failure_row["fact_id"])
            row_failure_metadata = {
                **failure_adapter_metadata,
                "snapshot_repo_id": str(failure_row["repo_id"]),
                "snapshot_commit_sha": str(failure_row["commit_sha"]),
                "num_chunks": (
                    len(chunk_ids)
                    if (
                        "chunk_ids" in locals()
                        and str(failure_row["repo_id"]) == repo_id
                        and str(failure_row["commit_sha"]) == commit
                    )
                    else None
                ),
                "failure_scope": (
                    "oom_snapshot"
                    if (
                        str(failure_row["repo_id"]) == repo_id
                        and str(failure_row["commit_sha"]) == commit
                    )
                    else "unattempted_after_shard_oom"
                ),
            }
            for failure_budget in args.retrieval_budgets:
                for failure_format in args.formats:
                    if failure_format == "mcq_permutation" and failure_row.get(
                        "mcq_permutation"
                    ) is None:
                        continue
                    failure_key = (
                        failure_fact_id,
                        failure_format,
                        failure_budget,
                    )
                    if failure_key in done:
                        continue
                    failure_options = (
                        {}
                        if failure_format == "open"
                        else {
                            "likelihood_correct": False,
                            "likelihood_predicted_index": None,
                            "gold_choice_probability": 0.0,
                            "choice_probabilities": None,
                            "brier_score": 1.0,
                            "option_scores": [],
                            "option_score_seconds": 0.0,
                            "option_score_sequence_tokens": 0,
                        }
                    )
                    failure_result = record(
                        args=args,
                        retrieval_budget=failure_budget,
                        row=failure_row,
                        evaluation_format=failure_format,
                        response="",
                        completion_tokens=0,
                        prompt_tokens=0,
                        generation_seconds=0.0,
                        context_metadata={
                            "context_mode": failure_budget,
                            "context_tokens_in_prompt": 0,
                        },
                        option_metadata=failure_options,
                        adapter_metadata=row_failure_metadata,
                    )
                    failure_result.update(
                        {
                            "failure": True,
                            "failure_kind": "cuda_out_of_memory",
                            "failure_detail": failure_text,
                        }
                    )
                    append_jsonl(predictions_path, failure_result)
                    done.add(failure_key)
    finally:
        if adapter_model is not None:
            clear_lora_from_layers(
                adapter_model.base_model,
                adapter_model.hypernet.layer_indices,
            )
    write_summary(args, len(rows))
    completed_rows = read_jsonl(predictions_path) if predictions_path.exists() else []
    failed_records = sum(bool(row.get("failure")) for row in completed_rows)
    atomic_write_json(
        args.output_dir / "health.json",
        {
            "status": (
                "complete_with_reported_method_limit"
                if failed_records else "complete"
            ),
            "mode": args.mode,
            "selected_rows": len(rows),
            "completed_keys": len(done),
            "failed_records": failed_records,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **cuda_memory(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
