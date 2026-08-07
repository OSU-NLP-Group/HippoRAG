import json
import logging
import os
from collections.abc import Callable
from glob import glob
from hashlib import sha256
from math import ceil, isclose
from os import path
from pathlib import Path
from random import choices
from typing import Any

import datasets
import numpy as np
import torch
from datasets import Dataset, interleave_datasets, is_caching_enabled, load_dataset
from transformers import PreTrainedTokenizerBase

from ctx_to_lora.data.definitions import (
    CTX_AFFIXES,
    DS_KWARGS,
    IGNORE_INDEX,
    RAW_DATA_DIR,
    SELF_GEN_DATA_DIR,
    TRANSFORMED_DATA_DIR,
)
from ctx_to_lora.data.packing import pack_batch
from ctx_to_lora.data.preprocessing_fn import get_preprocessing_fn
from ctx_to_lora.data.self_gen_template import QA_PROMPT_TEMPLATE
from ctx_to_lora.retrieval.bm25 import BM25Document, BM25Index
from ctx_to_lora.utils import check_is_iterable, concat_list

logger = logging.getLogger()

COLS_TO_KEEP_PREPROCESSING = [
    "context",
    "repo_contexts",
    "chunk_ids",
    "chunk_paths",
    "retrieval_query",
    "prompts",
    "responses",
    "qas",
    "variation",
    "logprobs_vals",
    "logprobs_indices",
    "input_ids",
    "ctx_ids",
    "response_start_end",
]

COLS_TO_KEEP_TOKENIZED = [
    "input_ids",
    "labels",
    "context",
    "repo_contexts",
    "chunk_ids",
    "chunk_paths",
    "ctx_ids",
    "logprobs_vals",
    "logprobs_indices",
]


def get_ds_prob(train_ds_len: list[int], total_len: int):
    # if a dataset is smaller than 1%, make it 1%
    probs = [0 for _ in train_ds_len]
    for i, ds_len in enumerate(train_ds_len):
        if ds_len / total_len <= 0.01:
            probs[i] = 0.01
    res_probs = 1 - sum(probs)
    res_total_len = sum([l for l in train_ds_len if (l / total_len) > 0.01])
    for i, ds_len in enumerate(train_ds_len):
        if (ds_len / total_len) > 0.01:
            probs[i] = ds_len / res_total_len * res_probs
    logger.debug(f"Dataset probabilities: {probs}")
    assert isclose(sum(probs), 1.0), (
        f"Probs sum to {sum(probs)} ({probs}), expected 1.0"
    )
    return probs


def load_answers(ds_name, split):
    if ds_name.startswith("longbench"):

        def extract_ans(sample):
            return {"answers": sample["answers"]}

    elif "squad" in ds_name:

        def extract_ans(sample):
            return {"answers": sample["answers"]["text"]}

    elif ds_name == "drop":

        def extract_ans(sample):
            return {"answers": sample["answers_spans"]["spans"]}

    ds_kwargs = get_ds_kwargs(ds_name, split)
    ds = load_dataset(**ds_kwargs)
    ds = ds.map(extract_ans, num_proc=8, remove_columns=ds.column_names)
    return ds


def get_ds_kwargs(ds_name: str, split: str) -> dict[str, Any]:
    # custom logic for slicing iterable datasets
    take, skip = None, None
    if ("[" in split) and split.endswith("]"):
        split, slice = split.split("[")
        slice = slice.strip("]")
        skip = slice.split(":")[0]
        take = slice.split(":")[1]

    if ds_name.endswith(".parquet") or ds_name.endswith("*.parquet"):
        # ds_name may be an absolute path/glob or a path below RAW_DATA_DIR.
        pattern = ds_name if path.isabs(ds_name) else f"{RAW_DATA_DIR}/{ds_name}"
        files = glob(pattern)
        if not files:
            raise FileNotFoundError(
                f"The provided pattern does not match any files: {pattern}"
            )
        kwargs = dict(path="parquet", data_files=files, split="train")

    elif ds_name.startswith("self_gen/"):
        # e.g., "self_gen/google/gemma-2-2b-it/pwc"
        base_model_name = "/".join(ds_name.split("/")[1:3])
        base_ds = "/".join(ds_name.split("/")[3:])
        if ("[" in split) and split.endswith("]"):
            kwargs["split"], slice = split.split("[")
            slice = slice.strip("]")
            skip = slice.split(":")[0]
            if skip:
                kwargs["skip"] = int(skip)
            take = slice.split(":")[1]
            if take:
                kwargs["take"] = int(take)
        files = glob(
            f"{SELF_GEN_DATA_DIR}/{base_model_name}/{base_ds}/{split}/*.parquet"
        )
        if not files:
            raise FileNotFoundError(
                f"No self-gen files found for base model {base_model_name} "
                f"in {SELF_GEN_DATA_DIR}/{base_model_name}/{base_ds}/"
            )
        kwargs = dict(path="parquet", data_files=files, split="train")
    elif (ds_name not in DS_KWARGS) or (split not in DS_KWARGS[ds_name]):
        kwargs = dict(path=ds_name, split=split)
        logger.warning(
            f"No dataset kwargs found for '{ds_name}' with split '{split}'.\n"
            f"Using default kwargs: {kwargs}"
        )
    else:
        kwargs = DS_KWARGS[ds_name][split]

    if skip:
        kwargs["skip"] = int(skip)
    if take:
        kwargs["take"] = int(take)

    return kwargs


def len_filter(sample, max_length: int, keys: list[str]):
    m = [len(sample[k]) <= max_length for k in keys]
    return sum(m) == len(keys)


def filter_none(sample):
    for v in sample.values():
        if v is None:
            return False
    return True


def add_negative_prompt_fn(samples):
    unique_contexts = set()
    ctxs, prompts, responses = [], [], []
    for ctx, prompt, response in zip(
        samples["context"], samples["prompt"], samples["response"]
    ):
        if ctx in unique_contexts:
            continue
        unique_contexts.add(ctx)
        ctxs.append(ctx)
        prompts.append(prompt)
        responses.append(response)

    logger.debug("Adding negative prompt...")
    logger.debug(f"# unique contexts: {len(unique_contexts)}")

    # remove one last sample if the number of samples is odd
    if len(ctxs) % 2 != 0:
        ctxs.pop()
        prompts.pop()
        responses.pop()

    # to make sure that the negative prompt/response is not the same as the original
    indices = list(np.random.permutation(len(ctxs))) + list(
        np.random.permutation(len(ctxs))
    )
    neg_ctxs, neg_prompts, neg_responses = [], [], []
    for idx in range(0, len(indices), 2):
        i = indices[idx]
        j = indices[idx + 1]
        neg_ctxs.append(ctxs[i])
        neg_prompts.append(ctxs[j] + "\n\n" + prompts[j])
        neg_responses.append(responses[j])

    return dict(
        context=neg_ctxs + samples["context"],
        prompt=neg_prompts + samples["prompt"],
        response=neg_responses + samples["response"],
    )


def load_and_process_dataset(
    ds_name: str,
    split: str,
    add_negative_prompt: bool,
    num_proc: int,
    remove_cols: bool = True,
):
    logger.info(f"Loading dataset {ds_name} with split {split}...")
    try:
        ds_kwargs = get_ds_kwargs(ds_name, split)
        skip = ds_kwargs.pop("skip", None)
        take = ds_kwargs.pop("take", None)
        ds = load_dataset(**ds_kwargs)
        if skip is not None:
            ds = ds.skip(skip)
        if take is not None:
            ds = ds.take(take)
    except ValueError as e:
        raise ValueError(
            f"Failed to load dataset {ds_name} with split {split}. Error: {e}"
        )
    cols_to_remove = None
    if remove_cols:
        cols_to_remove = [
            col for col in ds.column_names if col not in COLS_TO_KEEP_PREPROCESSING
        ]
    is_eval = split != "train"
    ds = ds.map(
        get_preprocessing_fn(ds_name, is_eval),
        remove_columns=cols_to_remove,
        num_proc=16,
    )
    ds = ds.filter(
        filter_none,
        batched=False,
        num_proc=16,
    )
    if add_negative_prompt and "context" in ds:
        ds = ds.map(
            add_negative_prompt_fn,
            batched=True,
            batch_size=100_000,
            # num_proc=num_proc,
        )
    return ds


def get_tokenized_dataset(
    ds_name: str,
    split: str,
    max_qas_len: int,
    max_qas_per_sample: int,
    base_model_max_len: int,
    tokenizer: PreTrainedTokenizerBase,
    ctx_model_max_len: int,
    ctx_tokenizer: PreTrainedTokenizerBase,
    max_ctx_chunk_len: int,
    min_ctx_chunk_len: int,
    num_chunk_probs: dict[int, float] | None,
    max_ctx_chunk_num: int,
    add_ctx_to_chat: bool,
    add_negative_prompt: bool,
    use_kl_loss: bool,
    max_new_tokens: int = 256,
    add_self_distill_template: bool = False,
    set_format: str | None = None,
    truncate_if_too_long_inp: bool = False,
    truncate_if_too_long_ctx: bool = False,
    flip_ctx_inp: bool = False,
    repo_merge_method: str = "concat",
    retrieval_top_k: int = 8,
    repoqa_lazy_frozen_chunks: bool = False,
    repoqa_ce_streaming: bool = False,
    repoqa_stage: str = "stage1",
    repoqa_qa_token_budget: int = 8192,
    repoqa_validation_panel: str = "fast",
    repoqa_streaming_seed: int = 42,
    repoqa_chunk_cache_mb: int = 256,
    repoqa_max_repo_chunks: int = 0,
    repoqa_require_bm25_full_evidence: bool = False,
) -> dict[str, Any]:
    if max_qas_len > 0:
        assert max_qas_len <= base_model_max_len, (
            f"`max_qas_len` should be <= {base_model_max_len=}, got {max_qas_len=}"
        )
    logger.info(f"Loading dataset {ds_name} with split {split}...")
    need_ctx_ids = ctx_model_max_len is not None

    if repoqa_ce_streaming:
        if repoqa_lazy_frozen_chunks:
            raise ValueError("Choose production streaming or smoke lazy RepoQA, not both")
        if use_kl_loss:
            raise ValueError("Production RepoQA CE streaming rejects teacher targets")
        from ctx_to_lora.data.repoqa_ce_streaming import (
            load_repoqa_ce_streaming_dataset,
        )

        ready_path = ds_name
        if Path(ready_path).is_dir():
            ready_path = str(Path(ready_path) / "READY.json")
        return load_repoqa_ce_streaming_dataset(
            ready_path,
            repoqa_stage,
            "val" if split == "validation" else split,
            ctx_tokenizer,
            ctx_model_max_len,
            repoqa_qa_token_budget,
            repoqa_chunk_cache_mb,
            repoqa_streaming_seed,
            repoqa_validation_panel,
        )

    if repoqa_lazy_frozen_chunks:
        if use_kl_loss:
            raise ValueError(
                "Lazy frozen RepoQA currently supplies gold-answer CE targets, "
                "not teacher log probabilities; set use_kl_loss=false or prepare "
                "a separately named distillation index."
            )
        from ctx_to_lora.data.repoqa_lazy import load_frozen_repoqa_dataset

        return load_frozen_repoqa_dataset(
            [ds_name],
            tokenizer,
            ctx_tokenizer,
            repo_merge_method,
            max_qas_len,
            ctx_model_max_len,
            repoqa_chunk_cache_mb,
            max_qas_per_sample,
            split=split,
            max_repo_chunks=repoqa_max_repo_chunks,
            require_bm25_full_evidence=repoqa_require_bm25_full_evidence,
        )

    load_and_process_kwargs = dict(
        ds_name=ds_name,
        split=split,
        add_negative_prompt=add_negative_prompt,
    )
    tokenize_kwargs = dict(
        max_qas_len=max_qas_len,
        max_qas_per_sample=max_qas_per_sample,
        base_model_max_len=base_model_max_len,
        ctx_model_max_len=ctx_model_max_len,
        add_ctx_to_chat=add_ctx_to_chat,
        max_ctx_chunk_len=max_ctx_chunk_len,
        min_ctx_chunk_len=min_ctx_chunk_len,
        num_chunk_probs=num_chunk_probs,
        max_ctx_chunk_num=max_ctx_chunk_num,
        need_ctx_ids=need_ctx_ids,
        split=split,
        max_new_tokens=max_new_tokens,
        set_format=set_format,
        repo_merge_method=repo_merge_method,
        retrieval_top_k=retrieval_top_k,
    )

    all_kwargs = {**load_and_process_kwargs, **tokenize_kwargs}
    kwargs_str = json.dumps(all_kwargs)
    kwargs_str += tokenizer.name_or_path + ctx_tokenizer.name_or_path
    kwargs_str += "assistant_label_boundary_fallback_v1"
    logger.debug(f"Tokenizing dataset with kwargs: {kwargs_str}")
    kwargs_str += repr(tokenizer) + repr(ctx_tokenizer)
    ds_hash = sha256(kwargs_str.encode()).hexdigest()
    logger.debug(f"Dataset hash: {ds_hash}")
    ds_path = f"{TRANSFORMED_DATA_DIR}/{ds_hash}"

    if path.exists(ds_path) and ("train" in split) and is_caching_enabled():
        # load the cached ds
        logger.info(f"Loaded tokenized dataset from {ds_path}")
        tokenized_ds = datasets.load_from_disk(ds_path)
        if (not use_kl_loss) and ("logprobs_vals" in tokenized_ds.column_names):
            tokenized_ds = tokenized_ds.remove_columns(
                ["logprobs_vals", "logprobs_indices"]
            )
        return tokenized_ds

    num_proc = 4
    ds = load_and_process_dataset(
        **load_and_process_kwargs,
        num_proc=num_proc,
    )
    if use_kl_loss:
        if "train" in split and "logprobs_vals" not in ds.column_names:
            raise ValueError(
                "`use_kl_loss` is set to True but 'logprobs_vals' column "
                "is not present in the dataset."
            )
    logger.info(f"Constructing and tokenizing {ds_name} with {split} split...")
    if flip_ctx_inp:

        def squeeze(sample, column):
            first_id = sample[column][0]
            if check_is_iterable(first_id):
                sample[column] = first_id
            return sample

        def unsqueeze(sample, column):
            sample[column] = [sample[column]]
            return sample

        ds = ds.rename_column("context", "context_temp")
        ds = ds.rename_column("prompts", "context")
        ds = ds.rename_column("context_temp", "prompts")
        ds = ds.map(squeeze, fn_kwargs={"column": "context"})
        ds = ds.map(unsqueeze, fn_kwargs={"column": "prompts"})

    tokenized_ds = construct_and_tokenize_ctx_qa(
        ds=ds,
        tokenizer=tokenizer,
        ctx_tokenizer=ctx_tokenizer,
        add_self_distill_template=add_self_distill_template,
        num_proc=num_proc,
        truncate_if_too_long_inp=truncate_if_too_long_inp,
        truncate_if_too_long_ctx=truncate_if_too_long_ctx,
        **tokenize_kwargs,
    )

    if ("train" in split) and is_caching_enabled():
        tokenized_ds = tokenized_ds.shuffle()
        tokenized_ds.save_to_disk(ds_path, num_proc=16)
        # force reload from disk for fingerprint consistency
        tokenized_ds = datasets.load_from_disk(ds_path)

    if (not use_kl_loss) and ("logprobs_vals" in tokenized_ds.column_names):
        tokenized_ds = tokenized_ds.remove_columns(
            ["logprobs_vals", "logprobs_indices"]
        )

    return tokenized_ds


def construct_and_tokenize_ctx_qa(
    max_qas_len,
    max_qas_per_sample,
    base_model_max_len,
    tokenizer,
    ctx_model_max_len,
    ctx_tokenizer,
    add_ctx_to_chat,
    need_ctx_ids,
    max_ctx_chunk_len,
    min_ctx_chunk_len,
    num_chunk_probs,
    max_ctx_chunk_num,
    ds,
    split,
    max_new_tokens,
    add_self_distill_template=False,
    set_format=None,
    num_proc=None,
    truncate_if_too_long_inp=False,
    truncate_if_too_long_ctx=False,
    repo_merge_method="concat",
    retrieval_top_k=8,
):
    is_train = "train" in split
    if repo_merge_method == "bm25_topk_ties":
        if "repo_contexts" not in ds.column_names:
            raise ValueError(
                "bm25_topk_ties requires untokenized repo_contexts so selection "
                "can happen before context encoding"
            )
        ds = ds.map(
            select_repo_contexts_bm25,
            fn_kwargs={"top_k": retrieval_top_k},
            num_proc=16,
        )
    # for sft + chat_model, we need to convert the dataset to chat format
    if "input_ids" in ds.column_names and "response_start_end" in ds.column_names:
        # already tokenized dataset (e.g., self-gen qa data)
        tokenized_ds = ds.map(get_labels_from_input_ids, num_proc=16)
    else:
        # construct messages from prompts and responses
        # add "messages_list" field
        ds = ds.map(
            convert_ctx_prompt_response_to_messages,
            fn_kwargs={
                "add_ctx_to_chat": add_ctx_to_chat,
                "add_self_distill_template": add_self_distill_template,
            },
            num_proc=16,
        )
        # add `input_ids`, `attention_mask`, `labels`
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        logging.debug("Tokenizing inputs")

        tokenized_ds = ds.map(
            get_sft_prompt_formatting_fn(tokenizer),
            batched=True,
            batch_size=100_000,
        )

    tokenized_ds = tokenized_ds.remove_columns(
        [col for col in tokenized_ds.column_names if col not in COLS_TO_KEEP_TOKENIZED],
    )
    tokenized_ds = tokenized_ds.filter(
        lambda x: bool(x["input_ids"]),  # remove empty "input_ids"
        num_proc=16,
    )

    if need_ctx_ids:
        if "repo_contexts" in tokenized_ds.column_names:
            if "ctx_ids" in tokenized_ds.column_names:
                raise ValueError("Provide either repo_contexts or ctx_ids, not both")
            logging.info("Tokenizing precomputed canonical repository chunks")
            tokenized_ds = tokenized_ds.map(
                tokenize_repo_contexts,
                fn_kwargs={"tokenizer": ctx_tokenizer},
                num_proc=16,
                remove_columns=["repo_contexts"],
            )
        if (
            tokenizer.name_or_path != ctx_tokenizer.name_or_path
            and "ctx_ids" in tokenized_ds.column_names
        ):
            logger.info("Detokenizing contexts...")
            tokenized_ds = tokenized_ds.map(
                detokenize_ctx_text,
                fn_kwargs={"tokenizer": tokenizer},
                batched=True,
                batch_size=100_000,
                remove_columns=["ctx_ids"],
            )

        if "ctx_ids" not in tokenized_ds.column_names:
            # tokenize the ctx_text to get ctx_ids and ctx_attn_mask
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            logging.debug("Tokenizing context")
            tokenized_ds = tokenized_ds.map(
                tokenize_ctx_text,
                fn_kwargs={"tokenizer": ctx_tokenizer},
                batched=True,
                batch_size=100_000,
                remove_columns=["context"],
            )

        # ctx chunking
        # Ideally we want to chunk the raw text directly
        # however, since the contexts are tokenized during self-gen
        # we can only chunk the tokenized context which requires some workaround
        # e.g., removing/applying template to each chunk
        # with some big caveats, e.g., losing order info
        split_ctx_kwargs = {
            "max_chunk_len": max_ctx_chunk_len,
            "min_chunk_len": min_ctx_chunk_len,
            "num_chunk_probs": num_chunk_probs,
            "max_num_split": max_ctx_chunk_num,
            "model_name_or_path": tokenizer.name_or_path,
            "is_train": is_train,
        }
        logging.info(f"Chunking context with {split_ctx_kwargs=}")
        tokenized_ds = tokenized_ds.map(
            split_too_long_ctx,
            fn_kwargs=split_ctx_kwargs,
            num_proc=16,
        )

        logging.info(f"Num samples after ctx chunking: {len(tokenized_ds)}")
        logging.info(
            f"Avg. num chunks per ctx: {np.mean(tokenized_ds['n_ctx_chunks'])}"
        )
        tokenized_ds = tokenized_ds.remove_columns(["n_ctx_chunks"])
        retrieval_metadata = [
            column
            for column in (
                "chunk_ids",
                "chunk_paths",
                "retrieval_query",
                "bm25_scores",
            )
            if column in tokenized_ds.column_names
        ]
        if retrieval_metadata:
            tokenized_ds = tokenized_ds.remove_columns(retrieval_metadata)

        split_qa_kwargs = {
            "max_qas_len": max_qas_len,
            "max_qas_per_sample": max_qas_per_sample,
        }
        logging.info(f"Split too long QAs with {split_qa_kwargs=}")
        tokenized_ds = tokenized_ds.map(
            split_too_long_qas,
            fn_kwargs=split_qa_kwargs,
            batched=True,
            batch_size=12_500,
            num_proc=16,
        )
    if "train" not in split:
        # squeeze since we always have one query per sample in eval
        tokenized_ds = tokenized_ds.map(squeeze_tokens, num_proc=num_proc)
        tokenized_ds = tokenized_ds.map(
            add_length_info,
            fn_kwargs={"columns": ["input_ids"]},
        )
        if truncate_if_too_long_inp:
            tokenized_ds = tokenized_ds.map(
                truncate_middle_if_too_long,
                fn_kwargs={
                    "max_length": base_model_max_len,
                    "columns": ["input_ids", "labels"],
                    "max_new_tokens": max_new_tokens,
                },
            )
        if "ctx_ids" in tokenized_ds.column_names:
            tokenized_ds = tokenized_ds.map(
                add_length_info,
                fn_kwargs={"columns": ["ctx_ids"]},
            )
            if truncate_if_too_long_ctx:
                print(f"Truncating ctx to {ctx_model_max_len}")
                tokenized_ds = tokenized_ds.map(
                    truncate_middle_if_too_long,
                    fn_kwargs={
                        "max_length": ctx_model_max_len,
                        "columns": ["ctx_ids"],
                        # cxt encoder doesnt need to add new_tokens
                        "max_new_tokens": 0,
                    },
                )

    if set_format:
        tokenized_ds.set_format(type=set_format)

    return tokenized_ds


def get_labels_from_input_ids(sample: dict[str, Any]) -> dict[str, Any]:
    """
    Extract labels from input_ids and response_start.

    Args:
        sample: A dictionary containing 'input_ids' and 'response_start'

    Returns:
        A dictionary with 'labels' field added
    """
    labels = []
    for input_ids_i, (start_i, end_i) in zip(
        sample["input_ids"], sample["response_start_end"]
    ):
        len_input_ids = len(input_ids_i)
        # pad labels with -100
        pad_len_left = start_i
        pad_len_right = len_input_ids - end_i
        labels.append(
            [IGNORE_INDEX] * pad_len_left
            + input_ids_i[start_i:]
            + [IGNORE_INDEX] * pad_len_right
        )

    sample["labels"] = labels
    return sample


def get_sft_prompt_formatting_fn(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Get a function that formats examples for supervised fine-tuning.

    Args:
        sft_mode: The training task type
        tokenizer: The tokenizer to use for chat template application

    Returns:
        A function that takes a training example and returns formatted data

    Raises:
        NotImplementedError: If sft_mode is not COMPLETION or tokenizer has no chat template
    """

    if tokenizer.chat_template is None:
        raise NotImplementedError("Only chat models are supported")

    def get_input_ids(tokenized):
        return tokenized["input_ids"] if hasattr(tokenized, "keys") else tokenized

    def get_labels(messages, tok_ids, masks):
        if any(masks):
            return [id_ if mask else IGNORE_INDEX for id_, mask in zip(tok_ids, masks)]

        if not messages or messages[-1]["role"] != "assistant":
            return [IGNORE_INDEX] * len(tok_ids)

        prompt_ids = get_input_ids(
            tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                add_generation_prompt=True,
            )
        )
        response_start = min(len(prompt_ids), len(tok_ids))
        return [IGNORE_INDEX] * response_start + tok_ids[response_start:]

    @torch.inference_mode()
    def f_intx(samples):
        # flatten all the messages into a list
        # tokenize, the pack back correctly
        messages_list = [x for x in samples["messages_list"]]

        n_queries = [len(x) for x in messages_list]
        messages = concat_list(messages_list)
        logger.info(f"Tokenizing {len(messages)} messages...")
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )

        labels = []
        for msg, tok_ids, masks in zip(
            messages, tokens["input_ids"], tokens["assistant_masks"]
        ):
            labels.append(get_labels(msg, tok_ids, masks))

        del tokens["assistant_masks"]
        tokens["labels"] = labels
        per_ctx_tokens = {"input_ids": [], "labels": []}
        i = 0
        for n in n_queries:
            per_ctx_tokens["input_ids"].append(tokens["input_ids"][i : i + n])
            per_ctx_tokens["labels"].append(tokens["labels"][i : i + n])
            i += n

        return per_ctx_tokens

    return f_intx


def convert_ctx_prompt_response_to_messages(
    example: dict[str, Any],
    add_ctx_to_chat: bool,
    add_self_distill_template: bool = False,
) -> dict[str, Any]:
    """
    Convert context/prompt/response format to chat messages format.

    Args:
        example: Dictionary containing 'prompt' and 'response' keys
        add_ctx_to_chat: Whether to prepend context to the user message

    Returns:
        Dictionary with added 'messages' key containing chat format

    Raises:
        ValueError: If 'prompt' or 'response' keys are missing
    """
    prompt_field = "prompts"
    res_field = "responses"

    if prompt_field not in example or res_field not in example:
        raise ValueError(
            f"'{prompt_field}' and '{res_field}' are required. Got: {example}"
        )

    system_msg = ""
    if "system_message" in example:
        system_msg = example["system_message"].strip()

    messages_list = []
    for prompt, response in zip(example[prompt_field], example[res_field]):
        user_msg = prompt.strip()
        if add_ctx_to_chat:
            if add_self_distill_template:
                user_msg = QA_PROMPT_TEMPLATE.format(
                    context=example["context"].strip(), question=user_msg
                )
            else:
                user_msg = example["context"].strip() + "\n\n" + user_msg

        messages_list.append(
            [
                {"role": "system", "content": system_msg.strip()},
                {"role": "user", "content": user_msg.strip()},
                {"role": "assistant", "content": response},
            ]
        )

    return {"messages_list": messages_list}


def split_too_long_ctx(
    sample: dict[str, Any],
    model_name_or_path: str,
    num_chunk_probs: dict[int, float] | None,
    max_chunk_len: int,
    min_chunk_len: int,
    max_num_split: int | None,
    is_train: bool,
) -> dict[str, Any]:
    """
    Split context into smaller chunks if it exceeds the maximum length.

    Args:
        samples: Dictionary containing 'ctx_ids' and 'ctx_attn_mask'
        max_chunk_len: Maximum length for each context chunk
        max_num_split: Maximum number of splits allowed

    Returns:
        Dictionary with split context data
    """

    chunk_len = max_chunk_len
    ctx_ids = sample["ctx_ids"]
    # Canonical repository chunks are serialized upstream at file boundaries
    # and independently wrapped with the context encoder's chat template. Do
    # not flatten or re-split them here: that would destroy semantic boundaries.
    if ctx_ids and check_is_iterable(ctx_ids[0]):
        chunks = [list(chunk) for chunk in ctx_ids]
        if max_num_split is not None and len(chunks) > max_num_split:
            raise ValueError(
                "Precomputed repository context has more chunks than allowed: "
                f"{len(chunks)} > {max_num_split}. Select chunks upstream."
            )
        if max_chunk_len > 0:
            oversized = [len(chunk) for chunk in chunks if len(chunk) > max_chunk_len]
            if oversized:
                raise ValueError(
                    "A canonical repository chunk exceeds max_ctx_chunk_len: "
                    f"max={max(oversized)}, configured={max_chunk_len}"
                )
        return {"ctx_ids": chunks, "n_ctx_chunks": len(chunks)}
    # Early exits
    if chunk_len <= 0 and max_num_split is None:
        return {"ctx_ids": [ctx_ids], "n_ctx_chunks": 1}
    if chunk_len > 0 and max_num_split is not None:
        max_ctx_len = chunk_len * max_num_split
        if len(ctx_ids) > max_ctx_len:
            ctx_ids = ctx_ids[:max_ctx_len]

    n_chunks = None  # will be sampled (train) or derived (eval)
    if is_train and num_chunk_probs is not None:
        # Adjust theoretical upper bound based on min_chunk_len if provided
        if min_chunk_len:
            max_num_split = ceil(len(ctx_ids) / min_chunk_len)
        # New logic: sample number of chunks from num_chunk_probs (after filtering)

        # Keep only feasible chunk counts <= max_num_split and > 0
        if max_num_split is not None:
            filt = {k: v for k, v in num_chunk_probs.items() if 0 < k <= max_num_split}
        else:
            filt = num_chunk_probs

        # Ensure each chunk will not exceed max_chunk_len; enforce minimum required chunks
        min_required = (
            max(1, ceil(len(ctx_ids) / max_chunk_len)) if max_chunk_len > 0 else 1
        )
        # Remove options that would yield chunk length > max_chunk_len
        filt = {k: v for k, v in filt.items() if k >= min_required}
        n_chunks = choices(list(filt.keys()), weights=list(filt.values()), k=1)[0]

    # Derive n_chunks if not sampled (e.g., eval or fallback)
    if n_chunks is None:
        n_chunks = ceil(len(ctx_ids) / chunk_len)
    # Safety: at least 1
    n_chunks = max(1, n_chunks)
    if n_chunks == 1:
        return {"ctx_ids": [ctx_ids], "n_ctx_chunks": 1}

    avg_len = ceil(len(ctx_ids) / n_chunks)
    chunks = [ctx_ids[i : i + avg_len] for i in range(0, len(ctx_ids), avg_len)]

    ctx_affixes = CTX_AFFIXES.get(model_name_or_path)
    if ctx_affixes is None and model_name_or_path.endswith("gemma-4-E2B-it"):
        ctx_affixes = CTX_AFFIXES["google/gemma-4-E2B-it"]
    if ctx_affixes is None:
        raise KeyError(f"No CTX_AFFIXES entry for {model_name_or_path!r}")
    prefix = ctx_affixes["prefix"]
    suffix = ctx_affixes["suffix"]
    # Apply affixes
    chunks[0] = chunks[0] + suffix
    for i in range(1, len(chunks) - 1):
        chunks[i] = prefix + chunks[i] + suffix
    chunks[-1] = prefix + chunks[-1]

    return {"ctx_ids": chunks, "n_ctx_chunks": len(chunks)}


def split_too_long_qas(
    samples: dict[str, any], max_qas_len: int, max_qas_per_sample: int
):
    # samples keys: "input_ids", "attention_mask", "labels", "ctx_ids", "ctx_attn_mask"
    # split the qas into multiple samples if they are too long
    # e.g., if max_qas_len = 512, and qas is 1024 tokens long,
    # we split it such that each sample has at most 512 tokens
    # and the ctx_ids and ctx_attn_mask are the same for all samples
    if max_qas_len < 0 and max_qas_per_sample < 0:
        return samples
    input_ids = samples["input_ids"]
    labels = samples["labels"]
    ctx_ids = samples["ctx_ids"]
    target_logprobs_vals = samples.get("logprobs_vals", None)
    target_logprobs_indices = samples.get("logprobs_indices", None)

    # Pre-calculate total lengths to check if any splitting is needed
    total_lengths = [sum(len(x) for x in seq) for seq in input_ids]
    longest_old_qas_len = max(total_lengths) if total_lengths else 0

    # Early exit if no splitting needed
    if (max_qas_len < 0 or all(length <= max_qas_len for length in total_lengths)) and (
        max_qas_per_sample < 0
        or all(len(seq) <= max_qas_per_sample for seq in input_ids)
    ):
        logger.debug(f"Longest old qas len: {longest_old_qas_len}")
        logger.debug(f"Longest new qas len: {longest_old_qas_len}")
        return samples

    out = {k: list() for k in samples}
    longest_new_qas_len = 0
    n_skip = 0
    has_target_logprobs = (
        target_logprobs_vals is not None and target_logprobs_indices is not None
    )

    # Helper function to add a batch efficiently
    def add_batch(
        inp_ids_batch,
        labels_batch,
        ctx_id,
        target_vals_batch=None,
        target_indices_batch=None,
    ):
        out["input_ids"].append(inp_ids_batch)
        out["labels"].append(labels_batch)
        out["ctx_ids"].append(ctx_id)
        if has_target_logprobs:
            out["logprobs_vals"].append(target_vals_batch)
            out["logprobs_indices"].append(target_indices_batch)

    for i, tot_inp_len in enumerate(total_lengths):
        if (max_qas_len < 0 or tot_inp_len <= max_qas_len) and (
            max_qas_per_sample < 0 or len(input_ids[i]) <= max_qas_per_sample
        ):
            # No need to split - add entire sample
            for k in samples:
                out[k].append(samples[k][i])
            continue

        # Need to split this sample
        current_ctx_id = ctx_ids[i]
        new_qas_len = 0
        new_input_ids = []
        new_labels = []
        new_target_vals = [] if has_target_logprobs else None
        new_target_indices = [] if has_target_logprobs else None

        sequences = zip(input_ids[i], labels[i])
        if has_target_logprobs:
            sequences = zip(
                input_ids[i],
                labels[i],
                target_logprobs_vals[i],
                target_logprobs_indices[i],
            )

        for seq_data in sequences:
            if has_target_logprobs:
                inp_ids, label, target_vals, target_indices = seq_data
            else:
                inp_ids, label = seq_data
                target_vals, target_indices = None, None

            inp_len = len(inp_ids)
            if (max_qas_len > 0) and (inp_len > max_qas_len):
                # Skip individual sequences that are too long
                n_skip += 1
                continue

            # Check if we can add to current batch (both length and sample count limits)
            can_add_to_current = (
                max_qas_len < 0 or new_qas_len + inp_len <= max_qas_len
            ) and (max_qas_per_sample < 0 or len(new_input_ids) < max_qas_per_sample)

            if can_add_to_current:
                # Add to current batch
                new_qas_len += inp_len
                new_input_ids.append(inp_ids)
                new_labels.append(label)
                if has_target_logprobs:
                    new_target_vals.append(target_vals)
                    new_target_indices.append(target_indices)
            else:
                # Current batch is full, save it and start new batch
                if new_input_ids:  # Only add non-empty batches
                    add_batch(
                        new_input_ids,
                        new_labels,
                        current_ctx_id,
                        new_target_vals,
                        new_target_indices,
                    )
                    longest_new_qas_len = max(longest_new_qas_len, new_qas_len)

                # Start new batch with current sequence
                new_qas_len = inp_len
                new_input_ids = [inp_ids]
                new_labels = [label]
                if has_target_logprobs:
                    new_target_vals = [target_vals]
                    new_target_indices = [target_indices]

        # Add final batch if not empty
        if new_input_ids:
            add_batch(
                new_input_ids,
                new_labels,
                current_ctx_id,
                new_target_vals,
                new_target_indices,
            )
            longest_new_qas_len = max(longest_new_qas_len, new_qas_len)

    logger.debug(f"Longest old qas len: {longest_old_qas_len}")
    logger.debug(f"Longest new qas len: {longest_new_qas_len}")
    if n_skip:
        logger.warning(
            f"Skipped {n_skip} QA pairs because they were too long (> {max_qas_len=} tokens)"
        )

    return out


def unpack_data_eval(samples):
    # n_queries always == 1 for eval
    data = samples["data"]
    out = dict(input_ids=[], labels=[])
    if "ctx_ids" in samples:
        out["ctx_ids"] = []
    for i, d in enumerate(data):
        for tokens in zip(
            d["input_ids"],
            d["labels"],
        ):
            if "ctx_ids" in samples:
                out["ctx_ids"].append(samples["ctx_ids"][i])
            out["input_ids"].append(tokens[0])
            out["labels"].append(tokens[2])
    return out


def squeeze_tokens(sample: dict[str, Any]) -> dict[str, Any]:
    """
    Squeeze the input_ids and labels to remove any extra dimensions.

    Args:
        sample: A dictionary containing 'input_ids' and 'labels'
    Returns:
        A dictionary with squeezed 'input_ids' and 'labels'
    """
    first_id = sample["input_ids"][0]
    if check_is_iterable(first_id):
        sample["input_ids"] = first_id
    first_label = sample["labels"][0]
    if check_is_iterable(first_label):
        sample["labels"] = first_label
    return sample


def add_length_info(sample: dict[str, any], columns: list[str]) -> dict[str, any]:
    out = {}
    for k in columns:
        if check_is_iterable(sample[k][0]):
            # ctx_ids
            out[f"{k}_len"] = sum([len(x) for x in sample[k]])
        else:
            # input_ids
            label_idx = None
            if k == "input_ids" and "labels" in sample:
                label_idx = np.argmax(np.array(sample["labels"]) != -100)
                out[f"{k}_len"] = len(sample[k][:label_idx])
    return out


def truncate_middle_if_too_long(
    sample: dict[str, any],
    max_length: int,
    columns: list[str],
    max_new_tokens: int = 256,
) -> dict[str, any]:
    """
    Truncate the middle of a list of tokens to fit within a maximum length.

    Args:
        tokens: List of token IDs
        max_length: Maximum length for the truncated tokens

    Returns:
        List of truncated token IDs
    """
    max_new_tokens_half = max_new_tokens // 2
    # leave max_new_tokens for generation
    half = max_length // 2 - max_new_tokens_half
    for col in columns:
        if check_is_iterable(sample[col]) and len(sample[col]) == 1:
            t = sample[col][0]
            sample[col][0] = t[:half] + t[-half:] if len(t) > max_length else t
        else:
            t = sample[col]
            sample[col] = t[:half] + t[-half:] if len(t) > max_length else t
    return sample


def detokenize_ctx_text(
    samples: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    contexts = tokenizer.batch_decode(samples["ctx_ids"])
    return dict(context=contexts)


def tokenize_ctx_text(
    samples: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    if tokenizer.chat_template:
        tokenized_text = tokenizer.apply_chat_template(
            [
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": ctx.strip()},
                ]
                if isinstance(ctx, str)
                else ctx
                for ctx in samples["context"]
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_attention_mask=False,
            padding=False,
            truncation=False,
            add_special_tokens=False,  # special tokens are already added by the chat template
            return_dict=True,
        )
    else:
        raise NotImplementedError("Only support chat models.")

    ctx_ids = tokenized_text["input_ids"]
    return dict(ctx_ids=ctx_ids)


def tokenize_repo_contexts(
    sample: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    """Tokenize each already-frozen semantic chunk as its own D2L context."""

    contexts = sample["repo_contexts"]
    if not contexts:
        raise ValueError("repo_contexts must contain at least one canonical chunk")
    return tokenize_ctx_text({"context": contexts}, tokenizer)


def select_repo_contexts_bm25(
    sample: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    """Select canonical chunks from the question before any Gemma encoding."""

    contexts = sample["repo_contexts"]
    chunk_ids = sample.get("chunk_ids")
    if not chunk_ids or len(chunk_ids) != len(contexts):
        raise ValueError("BM25 rows require one stable chunk_id per repo_context")
    paths = sample.get("chunk_paths") or [[] for _ in contexts]
    if len(paths) != len(contexts):
        raise ValueError("chunk_paths must align one-to-one with repo_contexts")

    query = sample.get("retrieval_query")
    if not query:
        query = sample.get("prompt") or sample.get("prompts")
    if isinstance(query, list):
        if len(query) != 1:
            raise ValueError(
                "BM25 adapters are question-conditioned; use one QA per row "
                "or provide an explicit retrieval_query"
            )
        query = query[0]
    if not isinstance(query, str) or not query.strip():
        raise ValueError("BM25 rows require a non-empty retrieval_query/question")

    documents = [
        BM25Document(
            chunk_id=str(chunk_id),
            text=str(context),
            paths=tuple(path_list if isinstance(path_list, list) else [path_list]),
        )
        for chunk_id, context, path_list in zip(chunk_ids, contexts, paths)
    ]
    selected = BM25Index(documents).top_k(query, top_k)
    selected_ids = [document.chunk_id for document, _score in selected]
    by_id = {
        str(chunk_id): (context, path_list)
        for chunk_id, context, path_list in zip(chunk_ids, contexts, paths)
    }
    return {
        "repo_contexts": [by_id[chunk_id][0] for chunk_id in selected_ids],
        "chunk_ids": selected_ids,
        "chunk_paths": [by_id[chunk_id][1] for chunk_id in selected_ids],
        "bm25_scores": [float(score) for _document, score in selected],
    }


def pack(
    ds_dict: dict[str, Dataset],
    max_packed_inp_len: int,
    max_packed_ctx_len: int,
    max_packed_size: int,
    seed: int,
    num_proc: int = 0,
):
    from ctx_to_lora.data.repoqa_lazy import FrozenRepoQADataset
    from ctx_to_lora.data.repoqa_ce_streaming import RepoQACEStreamingDataset

    if ds_dict and all(
        isinstance(ds, RepoQACEStreamingDataset) for ds in ds_dict.values()
    ):
        if len(ds_dict) != 1:
            raise ValueError("Production RepoQA accepts exactly one READY manifest")
        return next(iter(ds_dict.values()))

    if ds_dict and all(isinstance(ds, FrozenRepoQADataset) for ds in ds_dict.values()):
        if len(ds_dict) != 1:
            raise ValueError(
                "Lazy frozen RepoQA expects one glob/pattern per split; the pattern "
                "may match any number of Parquet index shards."
            )
        packed_ds = next(iter(ds_dict.values()))
        logging.info(
            "Using %d lazy frozen RepoQA rows without materializing packed contexts",
            len(packed_ds),
        )
        return packed_ds

    kwargs = dict(
        max_packed_inp_len=max_packed_inp_len,
        max_packed_ctx_len=max_packed_ctx_len,
        max_packed_size=max_packed_size,
    )
    train_ds_lens = [len(ds) for ds in ds_dict.values()]
    total_samples = sum(train_ds_lens)
    logging.info(f"Total samples before packing: {total_samples}")
    logging.info("Packing dataset")

    sorted_keys = sorted(ds_dict)
    ds_fingerprint = "|".join([ds_dict[k]._fingerprint for k in sorted_keys])
    ds_hash = sha256((ds_fingerprint + json.dumps(kwargs)).encode()).hexdigest()
    ds_path = f"{TRANSFORMED_DATA_DIR}/packed_{ds_hash}"
    logger.info(
        f"Packing ds {ds_hash} with {max_packed_inp_len=} and {max_packed_ctx_len=}"
    )
    if path.exists(ds_path) and is_caching_enabled():
        logger.info(f"Loading a cached packed dataset for {ds_path}")
        packed_ds = datasets.load_from_disk(ds_path)
    else:
        train_ds = interleave_datasets(
            list(ds_dict.values()),
            probabilities=get_ds_prob(train_ds_lens, total_samples),
            seed=seed,
            stopping_strategy="all_exhausted",
        )
        logger.info(f"Train dataset length: {len(train_ds)}")
        packed_ds = train_ds.map(
            pack_batch,
            fn_kwargs={
                "max_packed_inp_len": max_packed_inp_len,
                "max_packed_ctx_len": max_packed_ctx_len,
                "max_packed_size": max_packed_size,
                "metadata_path": f"{ds_path}/packing_metadata.json",
            },
            batched=True,
            batch_size=125_000,
            num_proc=num_proc,
            remove_columns=train_ds.column_names,
        )
        # this would generate another cache file for the already concat'd + packed ds
        # TODO: saving here is not space efficient at all...
        # the contexts are being duplicated for each datapoint when splitting QAs
        packed_ds.save_to_disk(ds_path, num_proc=num_proc)

    logger.info(f"Packed dataset length: {len(packed_ds)}")
    logger.info(
        f"Avg. # of samples per packed sequence: {total_samples / len(packed_ds)}"
    )

    return packed_ds
