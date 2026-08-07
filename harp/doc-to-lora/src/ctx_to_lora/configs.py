import dataclasses
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, NewType

import torch
import yaml
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)


class ArgumentParser(HfArgumentParser):
    def parse_yaml_file_with_extends(self, yaml_arg: str) -> list[dataclass]:
        """Parse a YAML config with optional relative, recursive inheritance."""

        def load(config_path: str, seen: set[str]) -> dict[str, Any]:
            config_path = os.path.abspath(config_path)
            if config_path in seen:
                raise ValueError(f"Cyclic YAML extends chain at {config_path}")
            with open(config_path, encoding="utf-8") as handle:
                values = yaml.safe_load(handle) or {}
            parent = values.pop("extends", None)
            if parent is None:
                return values
            parent_path = (
                parent
                if os.path.isabs(parent)
                else os.path.join(os.path.dirname(config_path), parent)
            )
            inherited = load(parent_path, seen | {config_path})
            inherited.update(values)
            return inherited

        return self.parse_dict(load(yaml_arg, set()))

    def parse_yaml_and_args(
        self, yaml_arg: str, other_args: list[str] | None = None
    ) -> list[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file_with_extends(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {
            arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args
        }
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys
                if arg in keys:
                    if val in ["None", "none", "null", "NULL"]:
                        val = None
                        inputs[arg] = val
                        used_args[arg] = val
                        continue
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == list[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    if base_type == dict:
                        inputs[arg] = yaml.load(val, Loader=yaml.FullLoader)

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(
                            f"Duplicate argument provided: {arg}, may cause unexpected behavior"
                        )

            obj = data_class(**inputs)
            outputs.append(obj)
        for arg in other_args:
            if arg not in used_args:
                raise ValueError(f"Argument provided not found in dataclass: {arg}")
        return outputs

    def parse(self) -> DataClassType | tuple[DataClassType]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file_with_extends(
                os.path.abspath(sys.argv[1].split("=")[-1])
            )
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(
                os.path.abspath(sys.argv[1].split("=")[-1]), sys.argv[2:]
            )
        # parse --config for the yaml path and other command line args
        elif any([arg.startswith("--config") for arg in sys.argv]):
            yaml_arg = [
                arg
                for arg in sys.argv[1:]
                if arg.startswith("--config") and arg.endswith(".yaml")
            ][0]
            other_args = [arg for arg in sys.argv[1:] if arg != yaml_arg]
            output = self.parse_yaml_and_args(
                os.path.abspath(yaml_arg.split("=")[-1]), other_args
            )
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output


class ExperimentSetup(str, Enum):
    HYPERLORA = "hyper_lora"


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="",
        metadata={"help": "Placeholder. Will be overwritten by train.py"},
    )
    output_root: str = field(
        default="train_outputs/runs",
        metadata={"help": "Root directory for training runs."},
    )
    stop_after_steps: int = field(
        default=-1,
        metadata={
            "help": (
                "Stop and checkpoint at this global step while retaining "
                "max_steps as the full-run scheduler horizon."
            )
        },
    )
    tf32: bool = field(
        default=True,
        metadata={"help": "Whether to use tf32 precision."},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bf16 precision."},
    )
    label_names: list[str] = field(
        default=("labels",),
        metadata={
            "help": "List of strings to specify the label names in the dataset. "
            "This is used to compute the loss and metrics."
        },
    )
    include_for_metrics: list[str] = field(
        default=("inputs",),
        metadata={
            "help": "List of strings to specify additional data to include in the `compute_metrics` function."
            "Options: 'inputs', 'loss'."
        },
    )
    per_device_eval_batch_size: int = field(
        default=64,
        metadata={
            "help": "Batch size for evaluation. "
            "If not set, will use the same as per_device_train_batch_size."
        },
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size for training. "
            "If not set, will use the same as per_device_eval_batch_size."
        },
    )
    # TODO: use this! (check trainer.py for proper computation)
    average_tokens_across_devices: bool = field(
        default=False,
        metadata={"help": "compute num_items_in_batch across devices."},
    )
    # mem leak if use persistent workers
    # https://github.com/pytorch/pytorch/issues/62066
    # https://github.com/huggingface/transformers/issues/30943
    dataloader_persistent_workers: bool = field(
        default=False,
        metadata={
            "help": "Whether to keep the workers alive after a dataset has been consumed once."
        },
    )
    dataloader_prefetch_factor: int = field(
        default=16,
        metadata={"help": "Number of batches loaded in advance by each worker."},
    )
    dataloader_num_workers: int = field(
        default=8,
        metadata={"help": "Number of subprocesses to use for data loading."},
    )
    neftune_noise_alpha: float = field(
        default=5.0,
        metadata={"help": "Neftune noise alpha for the optimizer."},
    )
    learning_rate: float = field(
        default=4e-5,
        metadata={"help": "Initial learning rate."},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for the optimizer."},
    )
    optim: str = field(
        default="adamw_torch_fused",
        metadata={"help": "Optimizer."},
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Adam beta 1."},
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "Adam beta 2."},
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Adam epsilon."},
    )
    lr_scheduler_type: str = field(
        default="cosine_with_min_lr",
        metadata={"help": "Learning rate scheduler type."},
    )
    lr_scheduler_kwargs: dict = field(
        default=None,
        metadata={"help": "Learning rate scheduler kwargs."},
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Number of warmup steps."},
    )
    eval_on_start: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate on the start of training."},
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy."},
    )
    eval_steps: int = field(
        default=1_000,
        metadata={"help": "Evaluation steps."},
    )
    metric_for_best_model: str = field(
        default=None,
        metadata={"help": "Metric for best model."},
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Whether to load the best model at the end of training."},
    )
    save_total_limit: int = field(
        default=2,
        metadata={"help": "Total number of checkpoints to save."},
    )
    save_strategy: str = field(
        default="steps",
    )
    save_steps: int = field(
        default=5_000,
    )
    save_safetensors: bool = field(
        default=False,
    )
    logging_strategy: str = field(
        default="steps",
    )
    logging_steps: int = field(
        default=100,
    )
    use_liger_kernel: bool = field(
        default=False,
    )
    remove_unused_columns: bool = field(
        default=False,
    )
    # needed to avoid OOM by compute the metrics batch by batch
    # w/o this the trainer stores logits of all sample in memory...
    batch_eval_metrics: bool = field(
        default=True,
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": "Whether to log the first step."},
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "Whether to find unused parameters in DDP."},
    )
    ddp_timeout: int = field(
        default=2**20,
        metadata={"help": "Timeout for distributed data parallel training."},
    )


@dataclass
class ModelArguments:
    """
    Arguments for the base model.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={"help": ("Base model name or path.")},
    )
    model_revision: str | None = field(
        default=None,
        metadata={"help": "Exact Hugging Face revision for model and tokenizer."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention."},
    )


@dataclass
class LoRAArguments:
    lora_r: int | None = field(
        default=8,
        metadata={"help": ("LoRA R value.")},
    )
    lora_dropout: float | None = field(
        default=0.0,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_alpha: float | None = field(
        default=None,
        metadata={"help": ("LoRA alpha. Defaults to upstream rank-derived scaling if unset.")},
    )
    module_name_regex: str | None = field(
        default=None,
        metadata={"help": "Regex limiting full module names eligible for LoRA wrapping."},
    )
    target_modules: list[str] | None = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    target_module_shapes: list[str] | None = field(
        default=None,
        metadata={
            "help": (
                "Optional shape-aware target names, for example "
                "down_proj__in12288__out1536. Only matching modules and layers "
                "are generated by HyperLoRA."
            )
        },
    )


@dataclass
class CtxTrainingArguments:
    exp_setup: ExperimentSetup = field(
        default=ExperimentSetup.HYPERLORA,
        metadata={"help": "Experiment setup - LoRA, HyperLoRA, or full finetuning"},
    )
    from_pretrained_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to the pretrained checkpoint."},
    )
    max_base_len: int | None = field(
        default=2**13,
        metadata={"help": "Maximum base length for training."},
    )
    use_sequence_packing: bool = field(
        default=True,
        metadata={"help": "Whether to use sequence packing."},
    )
    max_ctx_len: int = field(
        default=-1,
        metadata={"help": "Max context length. Overrides ctx tokenizer length."},
    )
    max_qas_len: int = field(
        default=2**11,
        metadata={
            "help": "Maximum question-answering token length of each sample for training. "
            "QA pairs that are longer than this value will be split up into multiple samples."
        },
    )
    max_qas_per_sample: int | None = field(
        default=-1,
        metadata={
            "help": "Max QA pair per context. If a context has more QA pairs than this value, "
            "they will be split up into multiple samples."
        },
    )
    num_chunk_probs: dict = field(
        default=None,
        metadata={"help": "Probability distribution over chunk nums."},
    )
    max_ctx_chunk_len: int = field(
        default=-1,
        metadata={
            "help": "Max context chunk length. If a context is longer than this value, "
            "it will be split up into multiple chunks."
        },
    )
    min_ctx_chunk_len: int = field(
        default=-1,
        metadata={
            "help": "Min context chunk length. Used only with random chunking training"
        },
    )
    max_ctx_chunk_num: int | None = field(
        default=None,
        metadata={"help": "Max number of context chunks per sample."},
    )
    max_packed_inp_len: int | None = field(
        default=2**14,
        metadata={"help": "Maximum packed input length for training."},
    )
    max_packed_ctx_len: int | None = field(
        # forward pass of the ctx encoder is cheaper --> longer packed len
        default=2**15,
        metadata={"help": "Maximum packed context length for training."},
    )
    max_packed_size: int = field(
        default=-1,
        metadata={"help": "Maximum number of samples per packed training item."},
    )

    max_new_tokens: int | None = field(
        default=256,
        metadata={"help": "Maximum new tokens for generation-based evaluation."},
    )
    gen_per_device_eval_batch_size: int | None = field(
        default=1,
        metadata={"help": "Per device evaluation batch size for generation."},
    )
    notes: str | None = field(
        default=None,
        metadata={"help": "Wandb notes for the experiment."},
    )
    use_kl_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use KL loss."},
    )
    use_per_ctx_average_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use per-context average loss."},
    )
    gen_lora_l1_reg_coef: float = field(
        default=0.0,
        metadata={"help": "L1 regularization coefficient for generated LoRAs."},
    )
    wrong_repo_contrastive_coef: float = field(
        default=0.0,
        metadata={"help": "Margin-loss coefficient for wrong-repository adapters."},
    )
    wrong_repo_contrastive_margin: float = field(
        default=0.5,
        metadata={"help": "Required CE margin between wrong and correct adapters."},
    )
    wrong_repo_contrastive_pack_fraction: float = field(
        default=0.25,
        metadata={"help": "Fraction of frozen QA packs receiving contrastive loss."},
    )
    add_negative_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to add negative prompt training."},
    )


@dataclass
class DataArguments:
    train_ds_names: list[str] = field(
        default=None,
        metadata={"help": "Training dataset names."},
    )

    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to use streaming dataset for training."},
    )
    repoqa_lazy_frozen_chunks: bool = field(
        default=False,
        metadata={
            "help": (
                "Load lightweight RepoQA index rows and hydrate frozen canonical "
                "chunks lazily instead of duplicating repository text per QA."
            )
        },
    )
    repoqa_ce_streaming: bool = field(
        default=False,
        metadata={
            "help": (
                "Load a production RepoQA CE or snapshot-memory READY manifest "
                "and stream pretokenized context groups with an exact resumable "
                "cursor."
            )
        },
    )
    repoqa_stage: str = field(
        default="stage1",
        metadata={"help": "Production RepoQA curriculum stage."},
    )
    repoqa_qa_token_budget: int = field(
        default=8192,
        metadata={"help": "Physical answer-side token budget per context pack."},
    )
    repoqa_validation_panel: str = field(
        default="fast",
        metadata={"help": "Named immutable validation panel from READY.json."},
    )
    repoqa_chunk_cache_mb: int = field(
        default=256,
        metadata={"help": "Per-dataloader-worker LRU size for tokenized chunks."},
    )
    repoqa_max_repo_chunks: int = field(
        default=0,
        metadata={
            "help": (
                "If positive, retain only lazy RepoQA rows whose complete "
                "repository snapshot has at most this many canonical chunks."
            )
        },
    )
    repoqa_require_bm25_full_evidence: bool = field(
        default=False,
        metadata={
            "help": (
                "For BM25 training only, retain rows whose selected chunks "
                "contain every annotated evidence chunk. Never use for eval."
            )
        },
    )
    val_ds_names: list[str] | None = field(
        default=None,
        metadata={"help": "Validation dataset names."},
    )
    test_ds_names: list[str] | None = field(
        default=None,
        metadata={"help": "Test dataset names."},
    )
    max_train_samples_per_ds: int | None = field(
        default=None,
        metadata={"help": "Maximum number of training samples per dataset."},
    )
    max_val_samples_per_ds: int | None = field(
        default=1000,
        metadata={"help": "Maximum number of validation samples per dataset."},
    )
    max_test_samples_per_ds: int | None = field(
        default=500,
        metadata={"help": "Maximum number of test samples per dataset."},
    )


@dataclass
class HypernetArguments:
    compile_hypernet: bool = field(
        default=True,
        metadata={
            "help": (
                "Compile the hypernetwork. Disable for variable-rank whole-"
                "repository concatenation to avoid a new compiled backward "
                "graph for every chunk count."
            )
        },
    )
    compile_base_model: bool = field(
        default=True,
        metadata={
            "help": (
                "Compile the answer-side base model. Disable for variable-rank "
                "whole-repository concatenation because each distinct merged "
                "rank otherwise triggers a new compiled backward graph."
            )
        },
    )
    latent_size: int = field(
        default=512,
        metadata={"help": "Latent size for HyperLoRA."},
    )
    use_light_weight_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use light-weight LoRA."},
    )
    light_weight_latent_size: int = field(
        default=128,
        metadata={"help": "Latent size for light-weight LoRA."},
    )
    dropout_rate: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for HyperLoRA."},
    )
    extra_modules: list[str] | None = field(
        default=None,
        metadata={"help": "Extra modules to train."},
    )
    per_rank_gen: bool = field(
        default=False,
        metadata={"help": "Whether to use per-rank generation."},
    )
    use_bias: bool = field(
        default=True, metadata={"help": "Whether to include data-dependent LoRA"}
    )
    use_per_rank_bias: bool = field(
        default=False, metadata={"help": "Whether to use per-rank bias."}
    )
    per_layer_processing: bool = field(
        default=False,
        metadata={"help": "Whether to use per-layer processing (after preceiver)."},
    )
    use_token_mixing: bool = field(
        default=False,
        metadata={"help": "Whether to use token mixing block."},
    )
    num_pre_head_layers: int = field(
        default=1, metadata={"help": "# of layers before hypernet head"}
    )


@dataclass
class RepositoryMergerArguments:
    repo_merge_method: Literal[
        "concat",
        "learned_fusion",
        "ties",
        "streaming_ties_exact",
        "knots_ties",
        "bm25_topk_ties",
    ] = field(
        default="concat",
        metadata={"help": "How per-chunk Doc-to-LoRA outputs are composed."},
    )
    repo_output_rank: int = field(
        default=64,
        metadata={
            "help": (
                "Repository-dependent rank before the standard rank-8 "
                "Doc-to-LoRA bias is appended."
            )
        },
    )
    ties_keep_fraction: float = field(
        default=0.2,
        metadata={"help": "Globally retained TIES parameter fraction."},
    )
    ties_sign_method: Literal["sum", "sum_of_values", "sum_of_signs"] = field(
        default="sum",
        metadata={"help": "TIES sign-election rule."},
    )
    ties_merge_type: Literal["mean", "sum"] = field(
        default="mean",
        metadata={"help": "TIES disjoint aggregation rule."},
    )
    ties_merge_scale: float = field(
        default=1.0,
        metadata={"help": "Scalar applied once to the completed TIES update."},
    )
    knots_concat_across_output: bool = field(
        default=True,
        metadata={"help": "Use the reference KnOTS column-wise concatenation."},
    )
    knots_singular_value_epsilon: float = field(
        default=1e-5,
        metadata={"help": "Reference KnOTS supported-singular-value cutoff."},
    )
    retrieval_top_k: int = field(
        default=8,
        metadata={"help": "Maximum BM25-selected chunks encoded for job 5."},
    )
    repo_fusion_num_blocks: int = field(
        default=2,
        metadata={"help": "Repository-level Set-Perceiver block count."},
    )
    repo_fusion_num_heads: int = field(
        default=8,
        metadata={"help": "Repository-level Set-Perceiver attention heads."},
    )
    repo_svd_oversample: int = field(
        default=8,
        metadata={"help": "Oversampling dimension for large truncated SVDs."},
    )
    repo_svd_power_iterations: int = field(
        default=1,
        metadata={"help": "Power iterations for large truncated SVDs."},
    )
    repo_svd_exact_max_dim: int = field(
        default=512,
        metadata={"help": "Use exact SVD when the smaller matrix axis is at most this."},
    )
    repo_svd_singular_value_epsilon: float = field(
        default=1e-7,
        metadata={
            "help": (
                "Drop numerical-null singular directions before square-root "
                "factorization to keep SVD gradients finite."
            )
        },
    )
    repo_svd_seed: int = field(
        default=17,
        metadata={"help": "Deterministic projection seed for truncated SVD."},
    )


@dataclass
class CtxEncoderArguments:
    ctx_encoder_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Context encoder model name or path."},
    )
    ctx_encoder_revision: str | None = field(
        default=None,
        metadata={"help": "Exact Hugging Face revision for the context encoder."},
    )
    ctx_encoder_type: Literal["embed_only", "per_layer_activations", "early_exit"] = (
        field(
            default="early_exit",
            metadata={
                "help": "Context encoder type. "
                "Options: 'embed_only', 'per_layer_activations', 'early_exit'."
            },
        )
    )
    # used only with `early_exit` type
    layer_idx: int | None = field(
        default=None,
        metadata={
            "help": "Layer index for context encoder. "
            "Default to L//4 where L is the number of layers of the ctx model. "
            "Only used when ctx_encoder_type==early_exit"
        },
    )
    quantize_ctx_encoder: bool = field(
        default=False, metadata={"help": "Wheter to quantize the ctx encoder."}
    )
    ctx_encoder_last_layer: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of layers for the context encoder. "
            "Only used when ctx_encoder_type==per_layer_activations"
        },
    )
    ctx_chunk_microbatch_size: int = field(
        default=0,
        metadata={
            "help": (
                "For packed contexts, process this many canonical chunks at a "
                "time. Zero preserves the original packed call; one enables "
                "the memory-safe whole-repository path."
            )
        },
    )
    sequential_ctx_layer_aggregation: bool = field(
        default=False,
        metadata={
            "help": (
                "Run the shared layer-to-layer Perceiver once per selected "
                "context layer instead of treating layers as a large batch. "
                "This preserves the computation while reducing peak memory."
            )
        },
    )
    offload_ctx_layer_inputs_to_cpu: bool = field(
        default=False,
        metadata={
            "help": (
                "When selected context-layer inputs are consumed sequentially, "
                "stage those frozen bf16 tensors in CPU memory and transfer one "
                "layer at a time inside its activation-checkpointed Perceiver "
                "call. This preserves the objective while avoiding a resident "
                "all-layer activation tensor for long contexts."
            )
        },
    )
    ctx_encoder_mlp_chunk_size: int = field(
        default=0,
        metadata={
            "help": (
                "For the frozen context encoder, evaluate each token-wise MLP "
                "in sequence chunks of this size. Zero uses the model's original "
                "full-sequence MLP. Chunking preserves global attention and all "
                "context tokens while bounding position-wise MLP temporaries."
            )
        },
    )
    ctx_encoder_flex_query_chunk_size: int = field(
        default=0,
        metadata={
            "help": (
                "For a frozen FlexAttention context encoder, evaluate attention "
                "over this many query tokens at a time while retaining the full "
                "K/V sequence. Zero uses one full-query kernel. Query rows are "
                "independent, so this bounds kernel temporaries without changing "
                "the causal/sliding mask or context coverage."
            )
        },
    )
    offload_optimizer_state_during_context: bool = field(
        default=False,
        metadata={
            "help": (
                "Temporarily stage existing optimizer-state tensors on CPU "
                "during the frozen context-encoder pass, restoring them before "
                "the trainable hypernetwork and answer passes. This changes only "
                "storage placement, not optimizer values or update ordering."
            )
        },
    )
    checkpoint_ctx_to_lora_chunks: bool = field(
        default=False,
        metadata={
            "help": (
                "Recompute the frozen context encoder plus chunk hypernetwork "
                "during backward, retaining token IDs instead of every "
                "chunk's all-layer activation tensor."
            )
        },
    )


@dataclass
class AggregatorArguments:
    aggregator_type: Literal["pooler", "perceiver"] = field(
        default="perceiver",
        metadata={"help": "Aggregator type for HyperLoRA."},
    )

    # pooler
    pooling_type: str = field(
        default="mean",
        metadata={"help": "Pooling type for HyperLoRA."},
    )
    num_latent_factor: int = field(
        default=8,
        metadata={"help": "Number of latent factors for Perceiver."},
    )
    n_latent_queries: int = field(
        default=208,  # 26 * 8
        metadata={"help": "Number of latent queries of Perceiver."},
    )

    num_blocks: int = field(
        default=8,
        metadata={"help": "Number of blocks for Perceiver."},
    )
    num_self_attn_per_block: int = field(
        default=0,
        metadata={"help": "Number of self-attention layers per block for Perceiver."},
    )
    shared_weights: bool = field(
        default=False,
        metadata={"help": "Whether to share weights across blocks for Perceiver."},
    )
    perceiver_attn_implementation: Literal["eager", "sdpa"] = field(
        default="eager",
        metadata={
            "help": (
                "Attention kernel used by the Perceiver aggregator. SDPA keeps "
                "grouped K/V heads compact and avoids materializing long-context "
                "attention probabilities."
            )
        },
    )
    perceiver_activation_checkpointing: bool = field(
        default=False,
        metadata={
            "help": (
                "Checkpoint the Perceiver modality projection and each resampler "
                "block. This preserves exact gradients while preventing all "
                "long-context block activations from coexisting during backward."
            )
        },
    )
    perceiver_modality_projection_chunk_size: int = field(
        default=0,
        metadata={
            "help": (
                "Evaluate the Perceiver's token-wise modality-projection MLP "
                "in independently checkpointed sequence chunks of this size. "
                "Zero uses one full-sequence projection. Chunking preserves "
                "the projection and attention inputs while bounding backward "
                "recomputation temporaries."
            )
        },
    )


# needed for loading model from checkpoint
# see https://github.com/huggingface/transformers/pull/34632
torch.serialization.add_safe_globals(
    [
        DataArguments,
        CtxTrainingArguments,
        ModelArguments,
        LoRAArguments,
        TrainingArguments,
        HypernetArguments,
        RepositoryMergerArguments,
        AggregatorArguments,
        CtxEncoderArguments,
    ]
)


if __name__ == "__main__":
    print(ExperimentSetup)
    print(ExperimentSetup.LORA)
    print(ExperimentSetup.HYPER_LORA)
    print(ExperimentSetup.FULL_FINETUNE)
