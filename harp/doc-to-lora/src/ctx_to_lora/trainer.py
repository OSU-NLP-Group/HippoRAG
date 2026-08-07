import gc
import json
import logging
import math
import os
import signal
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import MethodType

import torch
from accelerate.utils import DistributedDataParallelKwargs
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import IntervalStrategy, seed_worker
from transformers.utils import WEIGHTS_NAME

from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.modeling.lora_layer import clear_lora_from_layers

logger = logging.getLogger()
_REQUEUE_REQUESTED = False
_SNAPSHOT_ANSWER_CHECKPOINT_MIN_TOKENS = 4096


def _request_requeue(_signum, _frame) -> None:
    global _REQUEUE_REQUESTED
    _REQUEUE_REQUESTED = True


def _flat_sequence_bounds(position_ids: torch.Tensor) -> list[tuple[int, int]]:
    """Return half-open sequence bounds from packed, reset position IDs."""

    positions = position_ids.reshape(-1)
    starts = torch.nonzero(positions == 0, as_tuple=False).flatten().tolist()
    if not starts or starts[0] != 0:
        raise ValueError("Packed position_ids must start at zero")
    ends = starts[1:] + [positions.numel()]
    return list(zip(starts, ends))


def per_qa_mean_loss(
    token_loss: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Reduce shifted causal-token losses to one mean for each logical QA.

    ``token_loss[i]`` predicts ``labels[i + 1]``. Deriving the supervised
    indices from that shift explicitly avoids counting the next packed QA's
    first token or dropping the first answer token.
    """

    flat_loss = token_loss.reshape(-1)
    flat_labels = labels.reshape(-1)
    qa_losses: list[torch.Tensor] = []
    for start, end in _flat_sequence_bounds(position_ids):
        target_positions = torch.arange(
            start + 1, end, device=flat_labels.device, dtype=torch.long
        )
        supervised = flat_labels[target_positions] != -100
        if not bool(supervised.any()):
            raise ValueError(
                f"Logical QA at packed token range [{start}, {end}) has no "
                "supervised answer tokens"
            )
        qa_losses.append(flat_loss[target_positions[supervised] - 1].mean())
    return torch.stack(qa_losses)


def per_group_l1_regularizer(
    generated_loras: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Return the original factor L1 regularizer separately for every group."""

    if not generated_loras:
        raise ValueError("Cannot regularize an empty generated-LoRA mapping")
    per_module: list[torch.Tensor] = []
    expected_groups: int | None = None
    for lora in generated_loras.values():
        factors: list[torch.Tensor] = []
        for name in ("A", "B"):
            value = lora[name]
            if value.ndim < 2:
                raise ValueError(f"Generated LoRA {name} must have a group axis")
            if expected_groups is None:
                expected_groups = value.shape[0]
            elif value.shape[0] != expected_groups:
                raise ValueError("Generated LoRA modules disagree on group count")
            factors.append(value.abs().mean(dim=tuple(range(1, value.ndim))))
        per_module.append(factors[0] + factors[1])
    return torch.stack(per_module).mean(dim=0)


def logical_qa_weighted_l1(
    generated_loras: dict[str, dict[str, torch.Tensor]],
    n_queries: torch.Tensor,
) -> torch.Tensor:
    per_group = per_group_l1_regularizer(generated_loras)
    multiplicity = n_queries.reshape(-1).to(
        device=per_group.device, dtype=per_group.dtype
    )
    if per_group.numel() != multiplicity.numel():
        raise ValueError(
            "Generated adapter count does not match n_queries: "
            f"{per_group.numel()} != {multiplicity.numel()}"
        )
    return (per_group * multiplicity).sum()


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.path = Path(output_dir) / "metrics.jsonl"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "step": state.global_step,
            "epoch": state.epoch,
            **logs,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        if any(key.startswith("eval_") for key in logs):
            validation_path = self.path.parent / "validation_results.jsonl"
            with validation_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")


class StopAfterGlobalStepCallback(TrainerCallback):
    """Create a resumable curriculum boundary without resetting the LR schedule."""

    def __init__(self, stop_after_steps: int):
        self.stop_after_steps = stop_after_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_steps:
            control.should_save = True
            control.should_training_stop = True
        return control


class RepoQASamplerStateCallback(TrainerCallback):
    """Persist one exact cursor per DDP rank after every Trainer checkpoint."""

    def __init__(self, dataset):
        self.dataset = dataset

    def on_save(self, args, state, control, **kwargs):
        if not hasattr(self.dataset, "state_dict"):
            return control
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get("RANK", "0"))
        path = checkpoint / f"repoqa_sampler_state.rank{rank}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.dataset.state_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return control


class RepoQAExhaustionCallback(TrainerCallback):
    """Stop a production stage at data exhaustion, before its safety ceiling."""

    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        if getattr(self.dataset, "exhausted", False):
            control.should_save = True
            control.should_training_stop = True
        return control


class RepoQACoverageMetricsCallback(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
        packs = int(getattr(self.dataset, "physical_packs_consumed", 0))
        logical = int(getattr(self.dataset, "logical_qas_consumed", 0))
        logs.update(
            {
                "logical_qas_consumed_local": logical,
                "unique_logical_qas_consumed_local": int(
                    getattr(self.dataset, "unique_logical_qas_consumed", 0)
                ),
                "supervised_tokens_consumed_local": int(
                    getattr(self.dataset, "supervised_tokens_consumed", 0)
                ),
                "answer_side_tokens_consumed_local": int(
                    getattr(self.dataset, "answer_side_tokens_consumed", 0)
                ),
                "context_tokens_consumed_local": int(
                    getattr(self.dataset, "context_tokens_consumed", 0)
                ),
                "mean_qas_per_adapter_local": logical / max(1, packs),
            }
        )
        return control


class CudaMemoryMetricsCallback(TrainerCallback):
    """Log exact per-interval CUDA allocator peaks, reduced across DDP ranks."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not torch.cuda.is_available():
            return control
        device = torch.device("cuda", torch.cuda.current_device())
        allocated = torch.tensor(
            torch.cuda.max_memory_allocated(device) / 2**20,
            device=device,
            dtype=torch.float64,
        )
        reserved = torch.tensor(
            torch.cuda.max_memory_reserved(device) / 2**20,
            device=device,
            dtype=torch.float64,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(allocated, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(reserved, op=torch.distributed.ReduceOp.MAX)
        logs["cuda_peak_allocated_mib"] = float(allocated.item())
        logs["cuda_peak_reserved_mib"] = float(reserved.item())
        torch.cuda.reset_peak_memory_stats(device)
        return control


class RequeueSignalCallback(TrainerCallback):
    """Checkpoint cleanly when Slurm sends its advance USR1 signal."""

    def on_step_end(self, args, state, control, **kwargs):
        if _REQUEUE_REQUESTED:
            control.should_save = True
            control.should_training_stop = True
            if state.is_world_process_zero:
                marker = Path(args.output_dir) / "requeue_requested"
                marker.write_text(str(state.global_step) + "\n", encoding="utf-8")
        return control


def per_ctx_loss_ce(inputs, labels, loss):
    """Compatibility name: return one answer-token mean per logical QA."""

    qa_losses = per_qa_mean_loss(loss, labels, inputs["position_ids"])
    expected = int(inputs["n_queries"].sum().item())
    if qa_losses.numel() != expected:
        raise ValueError(
            f"Packed sequence count {qa_losses.numel()} != logical QA count {expected}"
        )
    return qa_losses


def per_ctx_loss_kl(inputs, labels, loss):
    # loss is compact (label indices selected)
    n_queries_per_ctx = inputs["n_queries"].tolist()

    position_ids = inputs["position_ids"].squeeze(0)
    # account only label positions
    label_mask = labels.squeeze(0) != -100
    label_pos_ids = label_mask * position_ids
    label_pos_ids_diff = label_pos_ids.diff(
        append=torch.tensor([0], device=position_ids.device)
    )
    # assumes the input starts with non-assistant tokens
    start_label_pos = torch.where((label_pos_ids_diff > 0) * ~label_mask)[0]
    end_label_pos = torch.where((label_pos_ids_diff < 0) * label_mask)[0]

    label_seq_lens = end_label_pos - start_label_pos

    # find equiv start indices in the already sliced loss vector
    cu_label_seq_lens = torch.cumsum(label_seq_lens, dim=0)
    start_indices = torch.cat(
        (
            torch.tensor([0], device=cu_label_seq_lens.device),
            cu_label_seq_lens[:-1],
        )
    )

    # these stack and split can be optimized but let's keep it simple
    # mean across tokens of each q
    qa_losses = torch.stack(
        [loss[start:end].mean() for start, end in zip(start_indices, cu_label_seq_lens)]
    )

    # mean across queries of each ctx
    per_ctx_losses = [ql.mean() for ql in torch.split(qa_losses, n_queries_per_ctx)]

    # per-ctx loss
    loss = torch.stack(per_ctx_losses)
    return loss


class ModulatedModelTrainer(Trainer):
    def _build_accelerator_args(self, **kwargs):
        """Share DDP gradient storage with reducer buckets for snapshot runs.

        The hypernetwork has roughly 1 GiB of fp32 gradients per rank.  DDP's
        default keeps an equally large reducer-bucket copy after the first
        iteration.  Bucket views make each parameter's ``.grad`` alias its
        existing reducer storage; reductions, optimizer values, and update
        ordering remain unchanged.
        """

        accelerator_args = super()._build_accelerator_args(**kwargs)
        if getattr(self, "_snapshot_ddp_gradient_bucket_views", False):
            handlers = accelerator_args.get("kwargs_handlers", [])
            ddp_handlers = [
                handler
                for handler in handlers
                if isinstance(handler, DistributedDataParallelKwargs)
            ]
            if len(ddp_handlers) != 1:
                raise RuntimeError(
                    "Expected one DDP kwargs handler for snapshot-memory training"
                )
            ddp_handlers[0].gradient_as_bucket_view = True
            logger.info(
                "Enabled DDP gradient_as_bucket_view for snapshot-memory training"
            )
        return accelerator_args

    @staticmethod
    def _log_snapshot_cuda_phase(phase: str) -> None:
        """Emit rank-local allocator state at coarse snapshot-memory phases."""

        if not torch.cuda.is_available():
            return
        device = torch.device("cuda", torch.cuda.current_device())
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            "snapshot_cuda_phase rank=%d phase=%s allocated_mib=%.1f "
            "reserved_mib=%.1f peak_allocated_mib=%.1f peak_reserved_mib=%.1f",
            rank,
            phase,
            torch.cuda.memory_allocated(device) / 2**20,
            torch.cuda.memory_reserved(device) / 2**20,
            torch.cuda.max_memory_allocated(device) / 2**20,
            torch.cuda.max_memory_reserved(device) / 2**20,
        )

    def training_step(self, *args, **kwargs):
        exact_snapshot = bool(
            getattr(self.train_dataset, "requires_exact_resume", False)
        )
        if exact_snapshot:
            self._stage_optimizer_state_for_snapshot_step()
        completed = False
        try:
            loss = super().training_step(*args, **kwargs)
            completed = True
            if exact_snapshot:
                self._log_snapshot_cuda_phase("backward_complete")
                # Non-reentrant checkpointing constructs short-lived Python
                # reference cycles around the twenty sequential Perceiver calls.
                # Collect them only after backward has consumed the complete graph,
                # then return inactive allocator blocks before restoring Adam state
                # and starting the next cost-bucketed repository.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._log_snapshot_cuda_phase("post_backward_cleanup")
            return loss
        finally:
            if exact_snapshot:
                if not completed:
                    # Make a best effort to release a failed variable-size answer
                    # graph before copying the optimizer state back to its device.
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                self._restore_optimizer_state_after_snapshot_step()

    def _get_dataloader(
        self,
        dataset,
        description,
        batch_size,
        sampler_fn=None,
        is_training=False,
        dataloader_key=None,
    ):
        """Preserve RepoQA's exact rank-strided iterable assignment.

        Accelerate normally wraps every IterableDataset in another
        ``IterableDatasetShard``. RepoQA already shards by RANK/WORLD_SIZE, so
        that wrapper silently consumes ``world_size`` local rows per optimizer
        step and trains on only one. A native DataLoader is sufficient because
        Trainer's input preparation moves every tensor to the active device.
        """
        if not getattr(dataset, "rank_strided_assignment", False):
            return super()._get_dataloader(
                dataset=dataset,
                description=description,
                batch_size=batch_size,
                sampler_fn=sampler_fn,
                is_training=is_training,
                dataloader_key=dataloader_key,
            )

        data_collator = self._get_collator_with_removed_columns(
            self.data_collator, description=description
        )
        params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
        }
        if self.args.dataloader_num_workers > 0:
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                params["worker_init_fn"] = partial(
                    seed_worker,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.args.process_index,
                )
        dataloader = DataLoader(dataset, **params)
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}
        return dataloader

    def _keep_native_precision_model_outputs(self, model) -> None:
        """Retain bf16 logits while preserving Accelerate's autocast wrapper.

        Accelerate normally promotes every tensor returned by a mixed-precision
        forward to fp32. Snapshot-memory batches can contain four 8K QA packs,
        so promoting the full-vocabulary logits creates a transient allocation
        of roughly 32 GiB. The CE implementation selects supervised answer
        positions in native precision and promotes only those rows before its
        numerically sensitive reduction, so remove only the output-promotion
        layer, not autocast itself.
        """

        target = self.accelerator.unwrap_model(
            model, keep_fp32_wrapper=True, keep_torch_compile=True
        )
        forward = target.forward
        wrapper = getattr(forward, "__wrapped__", None)
        if (
            wrapper is not None
            and wrapper.__class__.__name__ == "ConvertOutputsToFp32"
            and hasattr(wrapper, "model_forward")
        ):
            target.forward = MethodType(wrapper.model_forward, target)

    def _init_training_state(self, *args, **kwargs):
        epochs_trained, steps_trained = super()._init_training_state(*args, **kwargs)
        if getattr(self.train_dataset, "requires_exact_resume", False):
            # Optimizer/scheduler/global_step still come from TrainerState, but
            # the immutable dataset cursor is already at the exact next QA.
            # Returning zero here prevents Trainer's generic batch fast-forward
            # from skipping the corpus a second time without relying on
            # --ignore_data_skip=true.
            return 0, 0
        return epochs_trained, steps_trained

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None):
        """Save a resumable Doc-to-LoRA checkpoint in its native torch format.

        ModulatedPretrainedModel.state_dict() includes the base-model path and
        hypernetwork/context configuration objects needed by from_state_dict().
        Recent Trainer versions unconditionally use safetensors for models that
        are not PreTrainedModel subclasses, but safetensors accepts tensors only.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving Doc-to-LoRA checkpoint to %s", output_dir)

        if state_dict is None:
            state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            self.data_collator.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        if hasattr(self.train_dataset, "state_dict"):
            sampler_state = self.train_dataset.state_dict()
            sampler_path = Path(output_dir) / "repoqa_sampler_state.json"
            temporary = sampler_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(sampler_state, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(sampler_path)

    def load_repoqa_sampler_state(self, checkpoint: str) -> None:
        rank = int(os.environ.get("RANK", "0"))
        sampler_path = Path(checkpoint) / f"repoqa_sampler_state.rank{rank}.json"
        if rank == 0 and not sampler_path.exists():
            sampler_path = Path(checkpoint) / "repoqa_sampler_state.json"
        if not sampler_path.exists():
            if getattr(self.train_dataset, "requires_exact_resume", False):
                raise FileNotFoundError(
                    f"Production RepoQA checkpoint lacks {sampler_path.name}: "
                    f"{checkpoint}"
                )
            return
        if not hasattr(self.train_dataset, "load_state_dict"):
            raise TypeError("Checkpoint has RepoQA state but dataset cannot restore it")
        self.train_dataset.load_state_dict(
            json.loads(sampler_path.read_text(encoding="utf-8"))
        )

    # modified from the base Trainer to support per-context average loss
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        # only used with `use_per_ctx_average_loss=True`
        batch_samples = []
        num_items_in_batch = None

        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break

        count_num_items_in_batch = (
            len(batch_samples) > 0
            and "labels" in batch_samples[0]
            and "n_ctx_chunks" in batch_samples[0]
        )

        if count_num_items_in_batch:
            num_items_in_batch = dict()
            num_items_in_batch["ctx"] = torch.tensor(
                sum([batch["n_ctx_chunks"].numel() for batch in batch_samples])
            ).to(device)
            num_items_in_batch["logical_qas"] = sum(
                [
                    batch.get("logical_qa_count", batch["n_queries"]).sum()
                    for batch in batch_samples
                ]
            ).to(device)
            num_items_in_batch["qa_loss_weight"] = sum(
                [
                    batch.get(
                        "logical_qa_loss_weight",
                        batch.get("logical_qa_count", batch["n_queries"]),
                    ).sum()
                    for batch in batch_samples
                ]
            ).to(device)
            # should we avg over num chunks?
            # num_items_in_batch["ctx"] = sum(
            #     [(batch["ctx_position_ids"] == 0).sum() for batch in batch_samples]
            # )
            num_items_in_batch["labels"] = sum(
                [(batch["labels"].ne(-100)).sum() for batch in batch_samples]
            ).to(device)

        if num_items_in_batch is not None:
            if self.args.average_tokens_across_devices:
                for k in num_items_in_batch:
                    num_items_in_batch[k] = self.accelerator.gather(
                        num_items_in_batch[k]
                    ).sum()

            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(device)

                if self.args.n_gpu > 1 and num_items_in_batch.dim() == 0:
                    # In the DataParallel case, convert the scalar tensor into a 1-dim tensor
                    num_items_in_batch = num_items_in_batch.unsqueeze(0)

        return batch_samples, num_items_in_batch


class DistillationTrainer(ModulatedModelTrainer):
    def __init__(self, *args, **kwargs):
        self.gen_lora_l1_reg_coef = kwargs.pop("gen_lora_l1_reg_coef", 0.0)
        self.use_per_ctx_average_loss = kwargs.pop("use_per_ctx_average_loss", False)
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # NOTE: the loss output from this fn will be ***added***
        # meaning that we should always scale the loss wrt `num_items_in_batch`
        # (average over the number of items in the accumulated batch)

        # Trainer may pass a scalar num_items_in_batch during evaluation, so
        # its presence is not a reliable train/eval discriminator. The model
        # mode is authoritative and is set by Trainer before each loop.
        is_train = model.training
        labels = inputs.pop("labels", None)
        label_pos = torch.where(labels != -100)
        if "logprobs_vals" not in inputs:
            raise ValueError(
                "DistillationTrainer requires 'logprobs_vals' and "
                "'logprobs_indices' in every training and evaluation batch."
            )

        target_logp = inputs.pop("logprobs_vals").squeeze(0)
        indices = inputs.pop("logprobs_indices").squeeze(0)
        # Teacher-only fields are not valid Gemma forward arguments.
        outputs, (gen_loras, _) = model(**inputs, return_generated_lora=True)

        assert label_pos[0].shape[0] == target_logp.shape[0], (
            "Label positions and target log probabilities should have the same # tokens."
            f"Got : {label_pos[0].shape[0]=} and {target_logp.shape[0]=}"
        )

        ##### KL loss
        outputs_logits = outputs.logits[label_pos[0], label_pos[1] - 1]  # shift back 1

        logq_full_denom = torch.logsumexp(outputs_logits, dim=-1, keepdim=True)  # (N,1)
        selected_logits = outputs_logits.gather(1, indices)  # (N,K)
        # log softmax at selected indices
        logq_selected = selected_logits - logq_full_denom
        p = target_logp.exp()
        loss = -(p * logq_selected).sum(dim=-1)

        # teacher_logp = torch.full_like(outputs_logits, -torch.inf)
        # teacher_logp.scatter_(1, indices, target_logp)
        # # reduction = "batchmean" if num_items_in_batch is None else "sum"
        # p = teacher_logp.exp()
        # logq = nn.functional.log_softmax(outputs_logits, dim=-1)
        # loss = -torch.sum(p * logq, dim=-1)

        if self.use_per_ctx_average_loss:
            loss = per_ctx_loss_kl(inputs, labels, loss)

        if is_train:
            if self.use_per_ctx_average_loss:
                loss = loss.sum() / num_items_in_batch["ctx"]
            else:
                loss = loss.sum() / num_items_in_batch["labels"]
        else:
            # eval
            loss = loss.mean()

        # if reduction == "batchmean":
        #     loss = loss.mean()
        # elif reduction == "sum":
        #     # loss does not scale with grad acc
        #     # num_items_in_batch does
        #     # this works for both token-avg and ctx-avg
        #     # loss = loss.sum() / num_items_in_batch

        # `num_items_in_batch` is # tokens if `args.use_ctx_average_loss=False``
        # loss = loss.sum() / num_items_in_batch
        #####

        ##### unpack gen lora dict and compute regularization loss
        l1_norm = 0
        n_modules = len(gen_loras)
        for lora in gen_loras.values():
            l1_norm += lora["A"].abs().sum(0).mean()
            l1_norm += lora["B"].abs().sum(0).mean()
        l1_norm /= n_modules
        if is_train:
            l1_norm /= num_items_in_batch["ctx"]

        total_loss = loss + self.gen_lora_l1_reg_coef * l1_norm
        #####

        scaler = self.args.gradient_accumulation_steps if is_train else 1
        if self.args.average_tokens_across_devices and is_train:
            total_loss *= self.accelerator.num_processes
            scaler *= self.accelerator.num_processes

        # rough estimate of the losses (we only log the values from one step)
        if (self.state.global_step == 1 and self.args.logging_first_step) or (
            self.args.logging_strategy == IntervalStrategy.STEPS
            and self.state.global_step % self.state.logging_steps == 0
        ):
            # compensate `num_items_in_batch` division
            self.log(
                {
                    "kl_loss": loss.item() * scaler,
                    "gen_lora_l1_norm": l1_norm.item() * scaler,
                }
            )

        return (total_loss, outputs) if return_outputs else total_loss


def causal_lm_ce_loss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten first, then select supervised answer positions while logits are
    # still in their native (normally bf16) precision. Upcasting the complete
    # sequence-by-vocabulary tensor caused the production Stage-1 OOM even
    # though system/question targets are ignored by the exact objective.
    flat_logits = logits.reshape(-1, vocab_size)
    shift_labels = shift_labels.reshape(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(flat_logits.device)
    supervised = shift_labels != ignore_index
    if not bool(supervised.any()):
        # Preserve a differentiable, full-token-shaped zero result without
        # promoting a vocabulary-sized tensor.
        return flat_logits[:, 0].float() * 0

    supervised_logits = flat_logits[supervised].float()
    supervised_labels = shift_labels[supervised]
    supervised_loss = nn.functional.cross_entropy(
        supervised_logits,
        supervised_labels,
        reduction="none",
    )
    # per_qa_mean_loss consumes the historical full-token-shaped loss vector.
    # masked_scatter keeps that API and its causal-shift indexing exact while
    # allocating only one fp32 scalar per input token outside supervised rows.
    return supervised_loss.new_zeros(shift_labels.shape).masked_scatter(
        supervised, supervised_loss
    )


def supervised_causal_lm_targets(
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits positions and labels for one answer-only causal sequence.

    A target token at position ``t`` is predicted by the hidden state/logit at
    ``t - 1``.  Gemma 4 accepts this first tensor through ``logits_to_keep``,
    so its vocabulary projection can skip every unsupervised prompt position.
    """

    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError(
            "Answer-logit selection requires one logical QA with shape [1, length]"
        )
    target_positions = torch.nonzero(
        labels[0] != ignore_index, as_tuple=False
    ).flatten()
    if target_positions.numel() == 0:
        raise ValueError("A logical QA must contain at least one supervised token")
    if bool((target_positions == 0).any()):
        raise ValueError(
            "A supervised causal target at position zero has no preceding logit"
        )
    return target_positions - 1, labels[0, target_positions]


def compact_causal_lm_ce_loss(
    logits: torch.Tensor,
    supervised_labels: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Mean answer CE for logits already projected only at target positions."""

    flat_logits = logits.reshape(-1, vocab_size)
    supervised_labels = supervised_labels.reshape(-1).to(flat_logits.device)
    if flat_logits.shape[0] != supervised_labels.numel():
        raise ValueError(
            "Compact causal logits do not match the supervised target count"
        )
    return nn.functional.cross_entropy(
        flat_logits.float(),
        supervised_labels,
        reduction="mean",
    )


class CrossEntropyTrainer(ModulatedModelTrainer):
    def __init__(self, *args, **kwargs):
        train_dataset = kwargs.get("train_dataset")
        self._snapshot_ddp_gradient_bucket_views = bool(
            getattr(train_dataset, "requires_exact_resume", False)
        )
        self.gen_lora_l1_reg_coef = kwargs.pop("gen_lora_l1_reg_coef", 0.0)
        self.use_per_ctx_average_loss = kwargs.pop("use_per_ctx_average_loss", False)
        self.wrong_repo_contrastive_coef = kwargs.pop(
            "wrong_repo_contrastive_coef", 0.0
        )
        self.wrong_repo_contrastive_margin = kwargs.pop(
            "wrong_repo_contrastive_margin", 0.5
        )
        self.wrong_repo_contrastive_pack_fraction = kwargs.pop(
            "wrong_repo_contrastive_pack_fraction", 0.25
        )
        super().__init__(*args, **kwargs)

    def _stage_optimizer_state_for_snapshot_step(self) -> None:
        """Stage Adam state on CPU until answer CE and backward are complete.

        Adam state is not read by the context encoder, streamed answer objective,
        or backward. Keeping it off-device for the complete training step is
        therefore optimizer-equivalent while preserving headroom for the largest
        variable-length answer projection.
        """

        if getattr(self, "_snapshot_staged_optimizer_state", None):
            raise RuntimeError("Snapshot optimizer state is already staged")
        model = self.model
        target = self.accelerator.unwrap_model(
            model, keep_fp32_wrapper=True, keep_torch_compile=True
        )
        enabled = bool(
            getattr(
                target.ctx_encoder_args,
                "offload_optimizer_state_during_context",
                False,
            )
        )
        optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        moved = []
        if enabled and optimizer is not None:
            for state in optimizer.state.values():
                for key, value in tuple(state.items()):
                    if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                        cpu_value = value.to("cpu")
                        state[key] = cpu_value
                        moved.append((state, key, value.device))
        self._snapshot_staged_optimizer_state = moved
        if moved:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(
                "Staged %.1f MiB of optimizer state on CPU for snapshot "
                "forward/backward",
                sum(
                    state[key].numel() * state[key].element_size()
                    for state, key, _device in moved
                )
                / 2**20,
            )

    def _restore_optimizer_state_after_snapshot_step(self) -> None:
        moved = getattr(self, "_snapshot_staged_optimizer_state", [])
        self._snapshot_staged_optimizer_state = []
        for state, key, device in moved:
            state[key] = state[key].to(device)
        if moved:
            logger.info(
                "Restored %.1f MiB of optimizer state after snapshot backward",
                sum(
                    state[key].numel() * state[key].element_size()
                    for state, key, _device in moved
                )
                / 2**20,
            )

    def evaluate(self, *args, **kwargs):
        self._eval_ce_numerator = None
        self._eval_l1_numerator = None
        self._eval_logical_qas = None
        self._eval_loss_weight = None
        self._eval_correct_tokens = None
        self._eval_supervised_tokens = None
        metrics = super().evaluate(*args, **kwargs)

        def globally_sum(value):
            if value is None:
                value = torch.tensor(0.0, device=self.args.device)
            return self.accelerator.reduce(value, reduction="sum").item()

        logical_qas = globally_sum(self._eval_logical_qas)
        loss_weight = globally_sum(self._eval_loss_weight)
        supervised_tokens = globally_sum(self._eval_supervised_tokens)
        ce = globally_sum(self._eval_ce_numerator) / max(1.0, loss_weight)
        l1 = globally_sum(self._eval_l1_numerator) / max(1.0, loss_weight)
        custom = {
            "eval_answer_ce": ce,
            "eval_perplexity": math.exp(min(ce, 80.0)),
            "eval_answer_token_accuracy": (
                globally_sum(self._eval_correct_tokens) / max(1.0, supervised_tokens)
            ),
            "eval_gen_lora_l1_norm": l1,
            "eval_logical_qas": logical_qas,
            "eval_loss_weight": loss_weight,
            "eval_supervised_tokens": supervised_tokens,
        }
        metrics.update(custom)
        self.log(custom)
        return metrics

    def _stream_snapshot_qa_packs(
        self,
        model,
        inputs,
        labels,
        qa_weights,
        qa_loss_weights,
        logical_qa_count,
        logical_qa_loss_weight,
        qa_pack_counts,
        repo_key,
        num_items_in_batch,
    ):
        """Backpropagate bounded answer packs through one generated adapter."""

        # The context hypernetwork graph must remain alive while answer-side
        # gradients are accumulated and until Trainer invokes backward on the
        # surrogate.  Its saved activations otherwise coexist with persistent
        # optimizer/DDP state and leave only ~3 GiB on an 80-GiB H100 for some
        # exact production rounds.  PyTorch's saved-tensor hook copies only
        # tensors needed by backward to pinned host memory and restores those
        # exact tensors during recomputation; model outputs, data, loss,
        # gradients, optimizer semantics, and curriculum are unchanged.
        saved_tensor_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
            if torch.cuda.is_available()
            else nullcontext()
        )
        with saved_tensor_context:
            generated_loras = model(
                ctx_ids=inputs.pop("ctx_ids"),
                ctx_position_ids=inputs.pop("ctx_position_ids"),
                n_ctx_chunks=inputs.pop("n_ctx_chunks"),
                generate_lora_only=True,
            )
        self._log_snapshot_cuda_phase("context_forward_complete")
        target = self.accelerator.unwrap_model(
            model, keep_fp32_wrapper=True, keep_torch_compile=True
        )
        # The answer model consumes one logical QA and immediately extracts its
        # generated-adapter gradient. Transformer-level non-reentrant
        # checkpoint frames nevertheless retain every short QA activation
        # across later autograd.grad calls (about 14 GiB per production round
        # on the measured rank), whereas an ordinary forward releases the graph
        # at each explicit boundary below. The frozen answer model has no
        # trainable weights; context/Perceiver checkpointing is separate and
        # remains enabled. Disable only this redundant answer-side policy.
        if getattr(target.base_model, "is_gradient_checkpointing", False):
            target.base_model.gradient_checkpointing_disable()
            logger.info(
                "Disabled answer-model gradient checkpointing for streamed QAs"
            )
        factor_keys = [
            (module, name)
            for module in sorted(generated_loras)
            for name in ("A", "B")
        ]
        original_factors = [
            generated_loras[module][name] for module, name in factor_keys
        ]
        accumulated_grads = [torch.zeros_like(value) for value in original_factors]
        wrong_original_factors = None
        wrong_accumulated_grads = None
        pack_counts = [int(value) for value in qa_pack_counts.reshape(-1).tolist()]
        sequence_bounds = _flat_sequence_bounds(inputs["position_ids"])
        if sum(pack_counts) != len(sequence_bounds):
            raise ValueError("Frozen QA pack counts do not cover every logical QA")
        denominator = num_items_in_batch["qa_loss_weight"]
        if not isinstance(denominator, torch.Tensor):
            denominator = labels.new_tensor(denominator)
        denominator = denominator.to(labels.device).clamp_min(1)
        contrastive_enabled = (
            self.wrong_repo_contrastive_coef > 0
            and repo_key is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        contrastive_pack_count = int(
            round(len(pack_counts) * self.wrong_repo_contrastive_pack_fraction)
        )
        if self.wrong_repo_contrastive_pack_fraction > 0 and pack_counts:
            contrastive_pack_count = max(1, contrastive_pack_count)
        contrastive_pack_count = min(len(pack_counts), contrastive_pack_count)
        contrastive_denominator = labels.new_ones((), dtype=torch.float32)
        if contrastive_enabled and contrastive_pack_count:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            gathered_repo_keys = [
                torch.zeros_like(repo_key) for _ in range(world_size)
            ]
            torch.distributed.all_gather(gathered_repo_keys, repo_key)
            source_rank = next(
                (
                    (rank + offset) % world_size
                    for offset in range(1, world_size)
                    if int(gathered_repo_keys[(rank + offset) % world_size].item())
                    != int(repo_key.item())
                ),
                None,
            )
            if source_rank is None:
                contrastive_enabled = False
            else:
                from torch.distributed.nn.functional import all_gather

                wrong_original_factors = []
                for factor in original_factors:
                    rank_size = torch.tensor(
                        factor.shape[2], device=factor.device, dtype=torch.long
                    )
                    torch.distributed.all_reduce(
                        rank_size, op=torch.distributed.ReduceOp.MAX
                    )
                    padding = int(rank_size.item()) - factor.shape[2]
                    padded = (
                        torch.cat(
                            [
                                factor,
                                factor.new_zeros(
                                    *factor.shape[:2],
                                    padding,
                                    *factor.shape[3:],
                                ),
                            ],
                            dim=2,
                        )
                        if padding
                        else factor
                    )
                    wrong_original_factors.append(all_gather(padded)[source_rank])
                wrong_accumulated_grads = [
                    torch.zeros_like(value) for value in wrong_original_factors
                ]
                selected_qas = sum(pack_counts[:contrastive_pack_count])
                local_contrastive_qas = (
                    labels.new_tensor(selected_qas, dtype=torch.float32)
                    if qa_loss_weights is None
                    else qa_loss_weights.reshape(-1)[:selected_qas]
                    .detach()
                    .to(device=labels.device, dtype=torch.float32)
                    .sum()
                )
                torch.distributed.all_reduce(
                    local_contrastive_qas, op=torch.distributed.ReduceOp.SUM
                )
                contrastive_denominator = local_contrastive_qas.clamp_min(1)
        ce_numerator = labels.new_zeros((), dtype=torch.float32)
        contrastive_numerator = labels.new_zeros((), dtype=torch.float32)
        query_offset = 0
        for pack_index, pack_count in enumerate(pack_counts):
            leaf_factors = [
                value.detach().requires_grad_(True) for value in original_factors
            ]
            leaf_loras = {}
            for (module, name), value in zip(factor_keys, leaf_factors):
                leaf_loras.setdefault(module, {})[name] = value
            for local_query in range(pack_count):
                query_index = query_offset + local_query
                first_token, last_token = sequence_bounds[query_index]
                query_token_count = last_token - first_token
                checkpoint_long_answer = (
                    query_token_count >= _SNAPSHOT_ANSWER_CHECKPOINT_MIN_TOKENS
                )
                if checkpoint_long_answer:
                    # Ordinary forwards release each short answer graph cleanly,
                    # but the longest frozen answers can otherwise retain tens
                    # of GiB of transformer activations before their compact
                    # FP32 vocabulary loss. Non-reentrant recomputation preserves
                    # the same forward and gradients while bounding that peak.
                    target.base_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                    rank = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_available()
                        and torch.distributed.is_initialized()
                        else 0
                    )
                    logger.info(
                        "Enabled answer gradient checkpointing rank=%d "
                        "query_tokens=%d threshold=%d",
                        rank,
                        query_token_count,
                        _SNAPSHOT_ANSWER_CHECKPOINT_MIN_TOKENS,
                    )
                wrong_loss_value = None
                wrong_grads = None
                if contrastive_enabled and pack_index < contrastive_pack_count:
                    wrong_leaf_factors = [
                        value.detach().requires_grad_(True)
                        for value in wrong_original_factors
                    ]
                    wrong_leaf_loras = {}
                    for (module, name), value in zip(
                        factor_keys, wrong_leaf_factors
                    ):
                        wrong_leaf_loras.setdefault(module, {})[name] = value
                    wrong_inputs = {
                        "input_ids": inputs["input_ids"][
                            :, first_token:last_token
                        ],
                        "position_ids": inputs["position_ids"][
                            :, first_token:last_token
                        ],
                        "n_queries": inputs["n_queries"].new_tensor([1]),
                        "generated_loras_override": wrong_leaf_loras,
                    }
                    wrong_labels = labels[:, first_token:last_token]
                    (
                        wrong_logit_positions,
                        wrong_supervised_labels,
                    ) = supervised_causal_lm_targets(wrong_labels)
                    wrong_inputs["logits_to_keep"] = wrong_logit_positions
                    wrong_outputs = target(**wrong_inputs)
                    wrong_qa_loss = compact_causal_lm_ce_loss(
                        wrong_outputs.logits,
                        wrong_supervised_labels,
                        self.model.vocab_size,
                    )
                    wrong_grads = torch.autograd.grad(
                        wrong_qa_loss,
                        wrong_leaf_factors,
                        retain_graph=False,
                        create_graph=False,
                    )
                    wrong_loss_value = wrong_qa_loss.detach()
                    del (
                        wrong_outputs,
                        wrong_qa_loss,
                        wrong_logit_positions,
                        wrong_supervised_labels,
                        wrong_leaf_loras,
                        wrong_leaf_factors,
                    )
                query_inputs = {
                    "input_ids": inputs["input_ids"][:, first_token:last_token],
                    "position_ids": inputs["position_ids"][
                        :, first_token:last_token
                    ],
                    "n_queries": inputs["n_queries"].new_tensor([1]),
                    "generated_loras_override": leaf_loras,
                }
                query_labels = labels[:, first_token:last_token]
                (
                    logit_positions,
                    supervised_labels,
                ) = supervised_causal_lm_targets(query_labels)
                query_inputs["logits_to_keep"] = logit_positions
                outputs = target(**query_inputs)
                qa_loss = compact_causal_lm_ce_loss(
                    outputs.logits,
                    supervised_labels,
                    self.model.vocab_size,
                )
                weight = (
                    qa_loss.new_ones(())
                    if qa_loss_weights is None
                    else qa_loss_weights.reshape(-1)[query_index].to(
                        qa_loss.device, qa_loss.dtype
                    )
                )
                numerator = qa_loss * weight
                contrastive_active = (
                    wrong_loss_value is not None
                    and bool(
                        self.wrong_repo_contrastive_margin
                        + qa_loss.detach()
                        - wrong_loss_value
                        > 0
                    )
                )
                correct_objective = numerator / denominator
                if contrastive_active:
                    correct_objective = correct_objective + (
                        self.wrong_repo_contrastive_coef
                        * numerator
                        / contrastive_denominator
                    )
                grads = torch.autograd.grad(
                    correct_objective,
                    leaf_factors,
                    retain_graph=False,
                    create_graph=False,
                )
                for index, grad in enumerate(grads):
                    accumulated_grads[index].add_(grad.detach())
                ce_numerator += numerator.detach().float()
                if contrastive_active:
                    margin_loss = (
                        self.wrong_repo_contrastive_margin
                        + qa_loss.detach()
                        - wrong_loss_value
                    ) * weight.detach()
                    contrastive_numerator += margin_loss.float()
                    scale = (
                        -self.wrong_repo_contrastive_coef
                        * weight
                        / contrastive_denominator
                    )
                    for index, grad in enumerate(wrong_grads):
                        wrong_accumulated_grads[index].add_(
                            grad.detach() * scale
                        )
                del (
                    outputs,
                    qa_loss,
                    logit_positions,
                    supervised_labels,
                    grads,
                    numerator,
                    correct_objective,
                    query_inputs,
                    query_labels,
                )
                if wrong_grads is not None:
                    del wrong_grads
                if checkpoint_long_answer:
                    target.base_model.gradient_checkpointing_disable()
                    gc.collect()
                # Each QA is a complete graph boundary after its generated-LoRA
                # gradients have been copied into the fixed-size accumulators.
                # Return variable-shaped vocabulary blocks immediately rather
                # than retaining them until the end of a potentially very large
                # frozen QA pack.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            query_offset += pack_count
            del leaf_loras, leaf_factors

        self._log_snapshot_cuda_phase("answer_stream_complete")
        # The last answer forward leaves its detached A/B tensors bound in
        # every patched base-model module. Their gradients have already been
        # copied into the accumulators, so release those closures before the
        # context surrogate backward.
        clear_lora_from_layers(target.base_model, target.hypernet.layer_indices)

        # The surrogate's derivative with respect to each generated factor is
        # exactly its accumulated answer gradient. Autograd normally saves all
        # of those large constant gradient tensors on the GPU until Trainer's
        # later backward call. Store those saved tensors on pinned CPU instead;
        # their values and the resulting gradients remain bit-for-bit the same.
        surrogate_saved_tensor_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
            if torch.cuda.is_available()
            else nullcontext()
        )
        with surrogate_saved_tensor_context:
            surrogate = sum(
                (value * grad).sum()
                for value, grad in zip(original_factors, accumulated_grads)
            )
            if wrong_original_factors is not None:
                surrogate = surrogate + sum(
                    (value * grad).sum()
                    for value, grad in zip(
                        wrong_original_factors, wrong_accumulated_grads
                    )
                )
        del accumulated_grads
        if wrong_accumulated_grads is not None:
            del wrong_accumulated_grads
        if wrong_original_factors is not None:
            del wrong_original_factors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log_snapshot_cuda_phase("surrogate_complete")
        ce_loss = ce_numerator / denominator
        contrastive_loss = (
            self.wrong_repo_contrastive_coef
            * contrastive_numerator
            / contrastive_denominator
        )
        loss = (
            surrogate
            - surrogate.detach()
            + ce_loss.detach()
            + contrastive_loss.detach()
        )
        return loss, ce_numerator, generated_loras

    def _repository_merge_metrics(self, model) -> dict[str, float]:
        """Expose the latest local merger diagnostics to JSONL/dashboard logs."""

        target = self.accelerator.unwrap_model(
            model, keep_fp32_wrapper=True, keep_torch_compile=True
        )
        merger = getattr(getattr(target, "hypernet", None), "repo_merger", None)
        diagnostics = getattr(merger, "last_diagnostics", {}) if merger else {}
        metrics: dict[str, float] = {}
        for key, value in diagnostics.items():
            if key == "method" or isinstance(value, str):
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().float().item()
            if isinstance(value, (int, float)):
                metrics[f"repo_merge_local_{key}"] = float(value)
        return metrics

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        How the loss is computed by Trainer.
        By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """

        self._keep_native_precision_model_outputs(model)

        # Trainer may pass a scalar num_items_in_batch during evaluation, so
        # its presence is not a reliable train/eval discriminator. The model
        # mode is authoritative and is set by Trainer before each loop.
        is_train = model.training
        labels = inputs.pop("labels", None)
        qa_weights = inputs.pop("qa_weights", None)
        qa_loss_weights = inputs.pop("qa_loss_weights", qa_weights)
        logical_qa_count = inputs.pop("logical_qa_count", inputs["n_queries"])
        logical_qa_loss_weight = inputs.pop(
            "logical_qa_loss_weight", logical_qa_count
        )
        qa_pack_counts = inputs.pop("qa_pack_counts", None)
        repo_key = inputs.pop("repo_key", None)
        stream_packs = (
            is_train
            and qa_pack_counts is not None
        )
        outputs = None
        if stream_packs:
            loss, ce_numerator, gen_loras = self._stream_snapshot_qa_packs(
                model,
                inputs,
                labels,
                qa_weights,
                qa_loss_weights,
                logical_qa_count,
                logical_qa_loss_weight,
                qa_pack_counts,
                repo_key,
                num_items_in_batch,
            )
        else:
            outputs, (gen_loras, _) = model(**inputs, return_generated_lora=True)
            # [1, tot_seq_len]
            logits = outputs.logits

            # [tot_seq_len]
            loss = causal_lm_ce_loss(logits, labels, self.model.vocab_size)

            if self.use_per_ctx_average_loss:
                loss = per_ctx_loss_ce(inputs, labels, loss)
                if qa_loss_weights is not None:
                    weights = qa_loss_weights.reshape(-1).to(loss.device, loss.dtype)
                    if weights.numel() != loss.numel():
                        raise ValueError("qa_weights does not match packed logical QAs")
                    loss = loss * weights
            ce_numerator = loss.sum()

            if is_train:
                if self.use_per_ctx_average_loss:
                    loss = loss.sum() / num_items_in_batch["qa_loss_weight"]
                else:
                    loss = loss.sum() / num_items_in_batch["labels"]
            else:
                # eval
                loss = (
                    loss.sum() / logical_qa_loss_weight.sum().clamp_min(1)
                    if self.use_per_ctx_average_loss
                    else loss.mean()
                )

        #####
        # if is_train:
        #     if self.use_per_ctx_average_loss:
        #         loss_kwargs["num_items_in_batch"] = num_items_in_batch["ctx"]
        #     else:
        #         loss_kwargs["num_items_in_batch"] = num_items_in_batch["labels"]
        # inputs = {**inputs, **loss_kwargs}
        # outputs, (gen_loras, _) = model(**inputs, return_generated_lora=True)

        # # Save past state if it exists
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]

        # if labels is not None:
        #     unwrapped_model = self.accelerator.unwrap_model(model)
        #     if _is_peft_model(unwrapped_model):
        #         model_name = unwrapped_model.base_model.model._get_name()
        #     else:
        #         model_name = unwrapped_model._get_name()
        #     # User-defined compute_loss function
        #     if self.compute_loss_func is not None:
        #         loss = self.compute_loss_func(
        #             outputs, labels, num_items_in_batch=num_items_in_batch["labels"]
        #         )
        #     elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, "
        #             "only the following keys: "
        #             f"{','.join(outputs.keys())}. "
        #             "For reference, the inputs it received are "
        #             f"{','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        #####

        ##### unpack gen lora dict and compute regularization loss
        if self.use_per_ctx_average_loss:
            l1_numerator = logical_qa_weighted_l1(
                gen_loras, logical_qa_loss_weight
            )
            denominator = (
                num_items_in_batch["qa_loss_weight"]
                if is_train
                else logical_qa_loss_weight.sum().clamp_min(1)
            )
            l1_norm = l1_numerator / denominator
        else:
            l1_norm = 0
            for lora in gen_loras.values():
                l1_norm += lora["A"].abs().sum(0).mean()
                l1_norm += lora["B"].abs().sum(0).mean()
            l1_norm /= len(gen_loras)
            if is_train:
                l1_norm /= num_items_in_batch["ctx"]

        total_loss = loss + self.gen_lora_l1_reg_coef * l1_norm
        #####

        if not is_train and self.use_per_ctx_average_loss:
            def accumulate(name, value):
                current = getattr(self, name, None)
                setattr(self, name, value.detach() if current is None else current + value.detach())

            weights = (
                qa_weights.reshape(-1).to(logits.device, torch.float32)
                if qa_weights is not None
                else torch.ones(
                    int(inputs["n_queries"].sum().item()), device=logits.device
                )
            )
            correct = logits.new_zeros((), dtype=torch.float32)
            supervised_tokens = logits.new_zeros((), dtype=torch.float32)
            flat_labels = labels.reshape(-1)
            flat_predictions = logits.reshape(-1, logits.shape[-1]).argmax(dim=-1)
            for index, (start, end) in enumerate(
                _flat_sequence_bounds(inputs["position_ids"])
            ):
                targets = torch.arange(start + 1, end, device=logits.device)
                mask = flat_labels[targets] != -100
                weight = weights[index]
                correct += (
                    flat_predictions[targets[mask] - 1] == flat_labels[targets[mask]]
                ).sum() * weight
                supervised_tokens += mask.sum() * weight
            accumulate("_eval_ce_numerator", ce_numerator)
            accumulate("_eval_l1_numerator", l1_numerator)
            accumulate("_eval_logical_qas", logical_qa_count.sum())
            accumulate("_eval_loss_weight", logical_qa_loss_weight.sum())
            accumulate("_eval_correct_tokens", correct)
            accumulate("_eval_supervised_tokens", supervised_tokens)

        scaler = self.args.gradient_accumulation_steps if is_train else 1
        if self.args.average_tokens_across_devices and is_train:
            total_loss *= self.accelerator.num_processes
            scaler *= self.accelerator.num_processes

        # rough estimate of the losses (we only log the values from one step)
        if (self.state.global_step == 1 and self.args.logging_first_step) or (
            self.args.logging_strategy == IntervalStrategy.STEPS
            and self.state.global_step % self.state.logging_steps == 0
        ):
            # compensate `num_items_in_batch` division
            log_values = {
                "ce_loss": loss.item() * scaler,
                "gen_lora_l1_norm": l1_norm.item() * scaler,
            }
            log_values.update(self._repository_merge_metrics(model))
            self.log(log_values)

        return (total_loss, outputs) if return_outputs else total_loss


def get_decay_parameter_names(model) -> list[str]:
    """
    Get all parameter names that weight decay will be applied to.

    This function filters out parameters in two ways:
    1. By layer type (nn.Embedding)
    2. By parameter name patterns (containing 'bias', 'layernorm', 'rmsnorm'
       or 'latents_q' [perceiver's latent queries]).
    """
    decay_parameters = get_parameter_names(
        model,
        [nn.Embedding, nn.LayerNorm],
        ["scaler", "bias", "layernorm", "rmsnorm", "latents_q"],
    )
    return decay_parameters


def train_model(
    model,
    training_args,
    train_dataset=None,
    val_dataset=None,
    train_collator=None,
    compute_metrics=None,
):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        logger.info(f"Resuming from the checkpoint: {checkpoint}")

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,
        compute_metrics=compute_metrics,
    )

    is_modulated_model = isinstance(model, ModulatedPretrainedModel)
    trainer_cls = Trainer
    if is_modulated_model:
        logger.info("Training with modulated model.")
        trainer_cls = CrossEntropyTrainer
        trainer_kwargs["gen_lora_l1_reg_coef"] = training_args.gen_lora_l1_reg_coef
        trainer_kwargs["use_per_ctx_average_loss"] = (
            training_args.use_per_ctx_average_loss
        )
        trainer_kwargs["wrong_repo_contrastive_coef"] = (
            training_args.wrong_repo_contrastive_coef
        )
        trainer_kwargs["wrong_repo_contrastive_margin"] = (
            training_args.wrong_repo_contrastive_margin
        )
        trainer_kwargs["wrong_repo_contrastive_pack_fraction"] = (
            training_args.wrong_repo_contrastive_pack_fraction
        )
        del training_args.gen_lora_l1_reg_coef
        del training_args.use_per_ctx_average_loss
        del training_args.wrong_repo_contrastive_coef
        del training_args.wrong_repo_contrastive_margin
        del training_args.wrong_repo_contrastive_pack_fraction

        if training_args.use_kl_loss:
            logger.info("Training with distillation loss. Using DistillationTrainer.")
            trainer_cls = DistillationTrainer
            del training_args.use_kl_loss

    if training_args.auto_find_batch_size:
        # set the batch size to some high number
        # which will be lowered by the Trainer
        training_args.per_device_train_batch_size = 128

    trainer = trainer_cls(**trainer_kwargs)
    signal.signal(signal.SIGUSR1, _request_requeue)
    trainer.add_callback(RequeueSignalCallback())
    if hasattr(train_dataset, "state_dict"):
        trainer.add_callback(RepoQACoverageMetricsCallback(train_dataset))
    trainer.add_callback(CudaMemoryMetricsCallback())
    trainer.add_callback(JsonlMetricsCallback(training_args.output_dir))
    if hasattr(train_dataset, "state_dict"):
        trainer.add_callback(RepoQASamplerStateCallback(train_dataset))
        trainer.add_callback(RepoQAExhaustionCallback(train_dataset))
    if training_args.stop_after_steps > 0:
        if training_args.stop_after_steps > training_args.max_steps:
            raise ValueError("stop_after_steps cannot exceed max_steps")
        trainer.add_callback(
            StopAfterGlobalStepCallback(training_args.stop_after_steps)
        )
    # if getattr(trainer, "use_per_ctx_average_loss", False):
    #     trainer.get_batch_samples = trainer.get_batch_samples_ctx

    # MONKEY PATCH: remove embedding layers from weight decay
    trainer.get_decay_parameter_names = get_decay_parameter_names

    # Trainer loads the best model after training
    # is done when load_best_model_at_end=True
    if checkpoint is not None:
        trainer.load_repoqa_sampler_state(checkpoint)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    completion = None
    if getattr(train_dataset, "requires_exact_resume", False):
        values = torch.tensor(
            [
                int(getattr(train_dataset, "logical_qas_consumed", 0)),
                int(getattr(train_dataset, "unique_logical_qas_consumed", 0)),
                int(bool(getattr(train_dataset, "exhausted", False))),
            ],
            device=training_args.device,
            dtype=torch.long,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        world_size = trainer.accelerator.num_processes
        completion = {
            "format": "doc_to_lora_repoqa_stage_completion_v1",
            "stage": getattr(train_dataset, "stage", ""),
            "ready_sha256": getattr(train_dataset, "ready_sha256", ""),
            "world_size": world_size,
            "global_step": trainer.state.global_step,
            "logical_qas_consumed": int(values[0].item()),
            "unique_logical_qas_consumed": int(values[1].item()),
            "expected_logical_qas": int(train_dataset.expected_logical_qas),
            "expected_unique_logical_qas": int(
                train_dataset.expected_unique_logical_qas
            ),
            "exhausted_ranks": int(values[2].item()),
        }
        completion["passed"] = (
            completion["logical_qas_consumed"]
            == completion["expected_logical_qas"]
            and completion["unique_logical_qas_consumed"]
            == completion["expected_unique_logical_qas"]
            and completion["exhausted_ranks"] == world_size
        )
        bounded_stop = (
            training_args.stop_after_steps > 0
            and trainer.state.global_step >= training_args.stop_after_steps
        )
        completion["bounded_stop"] = bounded_stop
        if not completion["passed"] and not bounded_stop:
            raise RuntimeError(
                "Production RepoQA stage stopped before exact exhaustion: "
                + json.dumps(completion, sort_keys=True)
            )
    trainer.save_model()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if completion is not None and trainer.state.is_world_process_zero:
        filename = (
            "stage_completion.json"
            if completion["passed"]
            else "bounded_stop.json"
        )
        path = Path(training_args.output_dir) / filename
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    # TODO: add benchmark eval?
    # clear_gpu()
