import logging
from collections.abc import Iterable
from contextlib import contextmanager
from enum import Enum
from types import MethodType

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from transformers import PreTrainedModel
from transformers.integrations.flex_attention import (
    flex_attention_forward as transformers_flex_attention_forward,
)
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ctx_to_lora.configs import CtxEncoderArguments
from ctx_to_lora.utils import get_base_model, get_layers, set_layers

logger = logging.getLogger()


def _slice_flex_block_mask_queries(
    block_mask: BlockMask,
    start: int,
    end: int,
) -> BlockMask:
    """Return the exact global mask rows for one query-token interval."""

    query_block_size = int(block_mask.BLOCK_SIZE[0])
    if start % query_block_size:
        raise ValueError(
            "Flex query chunks must start on a BlockMask query-block boundary"
        )
    row_start = start // query_block_size
    row_end = (end + query_block_size - 1) // query_block_size
    rows = slice(row_start, row_end)
    original_mask_mod = block_mask.mask_mod

    def shifted_mask_mod(batch, head, query, key_value):
        return original_mask_mod(batch, head, query + start, key_value)

    full_num_blocks = (
        block_mask.full_kv_num_blocks[..., rows]
        if block_mask.full_kv_num_blocks is not None
        else None
    )
    full_indices = (
        block_mask.full_kv_indices[..., rows, :]
        if block_mask.full_kv_indices is not None
        else None
    )
    return BlockMask.from_kv_blocks(
        block_mask.kv_num_blocks[..., rows],
        block_mask.kv_indices[..., rows, :],
        full_num_blocks,
        full_indices,
        BLOCK_SIZE=block_mask.BLOCK_SIZE,
        mask_mod=shifted_mask_mod,
        seq_lengths=(end - start, block_mask.seq_lengths[1]),
        compute_q_blocks=False,
    )


def context_query_chunked_flex_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    **kwargs,
):
    """Run frozen context attention in independent bounded query intervals."""

    chunk_size = int(
        getattr(module.config, "ctx_encoder_flex_query_chunk_size", 0)
    )
    flex_kwargs = dict(kwargs)
    kernel_options = dict(flex_kwargs.get("kernel_options") or {})
    # The default forward autotuner includes BLOCK_N=128 candidates whose
    # double-buffered K/V tiles exceed an H100's per-block shared-memory
    # limit at Gemma 4's 256-wide head dimension. These smaller tiles change
    # only kernel scheduling, not the full-K/V attention calculation. Apply
    # them to both chunked and short-context fallback calls.
    kernel_options.setdefault("BLOCK_M", 64)
    kernel_options.setdefault("BLOCK_N", 64)
    kernel_options.setdefault("num_warps", 4)
    kernel_options.setdefault("num_stages", 1)
    flex_kwargs["kernel_options"] = kernel_options
    if (
        chunk_size <= 0
        or query.shape[-2] <= chunk_size
        or torch.is_grad_enabled()
        or not isinstance(attention_mask, BlockMask)
    ):
        return transformers_flex_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            **flex_kwargs,
        )

    query_block_size = int(attention_mask.BLOCK_SIZE[0])
    if chunk_size % query_block_size:
        raise ValueError(
            "ctx_encoder_flex_query_chunk_size must be divisible by the "
            f"BlockMask query block size {query_block_size}"
        )
    batch, heads, query_length, _ = query.shape
    output = value.new_empty(
        (batch, query_length, heads, value.shape[-1])
    )
    lse_output = None
    for start in range(0, query_length, chunk_size):
        end = min(query_length, start + chunk_size)
        chunk_mask = _slice_flex_block_mask_queries(
            attention_mask, start, end
        )
        chunk_output, chunk_lse = transformers_flex_attention_forward(
            module,
            query[:, :, start:end],
            key,
            value,
            chunk_mask,
            **flex_kwargs,
        )
        output[:, start:end].copy_(chunk_output)
        if chunk_lse is not None:
            if lse_output is None:
                lse_output = chunk_lse.new_empty(
                    (batch, heads, query_length)
                )
            lse_output[:, :, start:end].copy_(chunk_lse)
    return output, lse_output


ALL_ATTENTION_FUNCTIONS.register(
    "ctx_flex_attention",
    context_query_chunked_flex_attention_forward,
)
ALL_MASK_ATTENTION_FUNCTIONS.register(
    "ctx_flex_attention",
    ALL_MASK_ATTENTION_FUNCTIONS["flex_attention"],
)


def _token_chunked_mlp_forward(
    module: nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Evaluate an already-patched position-wise MLP in bounded token chunks."""

    chunk_size = int(module._ctx_to_lora_token_chunk_size)
    original_forward = module._ctx_to_lora_unchunked_forward
    if hidden_states.shape[-2] <= chunk_size:
        return original_forward(hidden_states)
    return torch.cat(
        [
            original_forward(chunk)
            for chunk in hidden_states.split(chunk_size, dim=-2)
        ],
        dim=-2,
    )


def enable_token_chunked_context_mlps(
    base_model: nn.Module, chunk_size: int
) -> None:
    """Bound frozen decoder-MLP temporaries without changing token coverage.

    Transformer MLPs are position-wise, so splitting only the sequence axis
    preserves the function. Attention remains full-context and unchanged.
    Patching ``forward`` in place also preserves parameter names and state dicts.
    """

    chunk_size = int(chunk_size)
    if chunk_size < 0:
        raise ValueError("ctx_encoder_mlp_chunk_size must be non-negative")
    if chunk_size == 0:
        return
    layers = get_layers(base_model)
    patched = 0
    for index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, nn.Module):
            raise ValueError(
                f"Context encoder layer {index} has no token-wise MLP to chunk"
            )
        if not hasattr(mlp, "_ctx_to_lora_unchunked_forward"):
            mlp._ctx_to_lora_unchunked_forward = mlp.forward
            mlp.forward = MethodType(_token_chunked_mlp_forward, mlp)
        mlp._ctx_to_lora_token_chunk_size = chunk_size
        patched += 1
    logger.info(
        "Enabled %d context-encoder token-wise MLPs with chunk size %d",
        patched,
        chunk_size,
    )


def _attention_implementation(model: nn.Module) -> str | None:
    config = model.config
    value = getattr(config, "_attn_implementation", None)
    if value is None:
        value = getattr(
            getattr(config, "text_config", None), "_attn_implementation", None
        )
    return value


def _gemma4_flex_kernel_options() -> dict[str, int | str | bool]:
    # Avoid PyTorch's short-query flex-decoding heuristic for Gemma 4 GQA.
    # It has no valid H100 kernel for some 8K shapes in torch 2.11. The regular
    # Triton FlexAttention kernel is exact and also handles the 130K path.
    return {
        "BACKEND": "TRITON",
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "num_stages": 1,
        "num_warps": 4,
    }


@contextmanager
def early_exit(base_model: PreTrainedModel, exit_layer: int):
    try:
        layers = get_layers(base_model)
        set_layers(base_model, layers[:exit_layer])
        yield base_model
    finally:
        set_layers(base_model, layers)


@contextmanager
def maybe_add_batch_dim(kwargs):
    try:
        batched_input = False
        batched_attn_mask = False
        if (
            "input_ids" in kwargs
            and kwargs["input_ids"] is not None
            and len(kwargs["input_ids"].shape) == 1
        ):
            kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
            batched_input = True
        if (
            "attention_mask" in kwargs
            and kwargs["attention_mask"] is not None
            and isinstance(kwargs["attention_mask"], torch.Tensor)
            and len(kwargs["attention_mask"].shape) == 1
        ):
            kwargs["attention_mask"] = kwargs["attention_mask"].unsqueeze(0)
            batched_attn_mask = True
        yield batched_input, batched_attn_mask
    finally:
        if batched_input:
            kwargs["input_ids"] = kwargs["input_ids"].squeeze(0)
        if batched_attn_mask:
            kwargs["attention_mask"] = kwargs["attention_mask"].squeeze(0)


class EarlyExit(nn.Module):
    def __init__(self, base_model: PreTrainedModel, config: CtxEncoderArguments):
        super().__init__()
        base_model = get_base_model(base_model)
        if "gte" in base_model.config.name_or_path:
            base_model.encoder.layer = base_model.encoder.layer[: config.layer_idx]
        else:
            set_layers(base_model, get_layers(base_model)[: config.layer_idx])

        self.base_model = base_model

    @property
    def config(self):
        return self.base_model.config

    @torch.no_grad()
    def forward(self, **kwargs):
        model_outputs = self.base_model(**kwargs)
        return model_outputs.last_hidden_state


class EmbeddingOnly(nn.Module):
    def __init__(self, base_model: PreTrainedModel, config: CtxEncoderArguments):
        super().__init__()
        self.base_model = base_model

    @property
    def config(self):
        return self.base_model.config

    @torch.no_grad()
    def forward(self, **kwargs):
        if _attention_implementation(self.base_model) == "flex_attention":
            # Gemma 4 E2B uses head_dim=256. FlexAttention's autotuned default
            # can request 256 KiB of shared memory, above this H100 build's
            # 232 KiB limit. Smaller tiles/staging preserve exact attention
            # while staying within the hardware resource ceiling.
            kwargs.setdefault(
                "kernel_options",
                _gemma4_flex_kernel_options(),
            )
        kwargs["output_hidden_states"] = True  # Force output of hidden states
        outputs = self.base_model(**kwargs)
        # Return the embeddings only
        return outputs.hidden_states[0]  # The first hidden state is the embeddings


class PerLayerActivations(nn.Module):
    def __init__(self, base_model: PreTrainedModel, config: CtxEncoderArguments):
        super().__init__()
        self.keep_lm_head = getattr(config, "keep_lm_head", False)
        self.offload_selected_layer_inputs_to_cpu = bool(
            getattr(config, "offload_ctx_layer_inputs_to_cpu", False)
        )
        if not self.keep_lm_head:
            base_model = get_base_model(base_model)  # remove lm head
        else:
            base_model.lm_head = nn.Identity()

        # A legacy explicit cutoff remains available for old checkpoints. With
        # no cutoff (the production full-corpus path), execute the complete
        # context stack and retain the input to every transformer block below.
        if config.ctx_encoder_last_layer is not None:
            last_layer = config.ctx_encoder_last_layer - 1
            if self.keep_lm_head:
                base_model.model.layers = base_model.model.layers[:last_layer]
            else:
                set_layers(base_model, get_layers(base_model)[:last_layer])
        enable_token_chunked_context_mlps(
            base_model,
            getattr(config, "ctx_encoder_mlp_chunk_size", 0),
        )
        text_config = getattr(base_model.config, "text_config", base_model.config)
        text_config.ctx_encoder_flex_query_chunk_size = int(
            getattr(config, "ctx_encoder_flex_query_chunk_size", 0)
        )
        self.base_model = base_model
        self._selected_layer_indices: tuple[int, ...] | None = None

    def select_layer_inputs(self, layer_indices: Iterable[int]) -> None:
        """Retain only the transformer-block inputs consumed by the hypernetwork.

        The complete context encoder still executes. Forward pre-hooks capture
        the requested block inputs without asking Transformers to keep every
        hidden state alive for the full sequence.
        """
        if self.keep_lm_head:
            raise ValueError(
                "select_layer_inputs is incompatible with keep_lm_head=True"
            )
        selected = tuple(int(index) for index in layer_indices)
        if not selected:
            raise ValueError("At least one context-encoder layer must be selected")
        if len(set(selected)) != len(selected):
            raise ValueError(f"Layer indices must be unique, got {selected}")
        layers = get_layers(self.base_model)
        invalid = [index for index in selected if index < 0 or index >= len(layers)]
        if invalid:
            raise ValueError(
                f"Layer indices {invalid} are outside [0, {len(layers)})"
            )
        self._selected_layer_indices = selected

    @property
    def config(self):
        return self.base_model.config

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.base_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.base_model.set_decoder(decoder)

    def get_decoder(self):
        return self.base_model.get_decoder()

    @torch.no_grad()
    def forward(self, **kwargs):
        if _attention_implementation(self.base_model) == "flex_attention":
            # Gemma 4 E2B uses head_dim=256. FlexAttention's autotuned default
            # can request 256 KiB of shared memory, above this H100 build's
            # 232 KiB limit. Smaller tiles/staging preserve exact attention.
            kwargs.setdefault(
                "kernel_options",
                _gemma4_flex_kernel_options(),
            )
        if self.keep_lm_head:
            kwargs["output_hidden_states"] = True
            return self.base_model(**kwargs)

        selected = self._selected_layer_indices
        if selected is None:
            kwargs["output_hidden_states"] = True
            outputs = self.base_model(**kwargs)
            # hidden_states[i] is the input to transformer block i. Exclude only
            # the final model output, after executing the complete stack.
            return torch.stack(outputs.hidden_states[:-1], dim=1)

        captured: dict[int, torch.Tensor] = {}
        handles = []

        def capture_input(
            layer_index: int,
            _module: nn.Module,
            args: tuple,
            hook_kwargs: dict,
        ) -> None:
            hidden = args[0] if args else hook_kwargs.get("hidden_states")
            if not isinstance(hidden, torch.Tensor):
                raise RuntimeError(
                    f"Could not capture tensor input for context layer {layer_index}"
                )
            captured[layer_index] = (
                hidden.detach().to("cpu")
                if self.offload_selected_layer_inputs_to_cpu
                else hidden
            )

        layers = get_layers(self.base_model)
        try:
            for layer_index in selected:
                handles.append(
                    layers[layer_index].register_forward_pre_hook(
                        lambda module, args, hook_kwargs, index=layer_index: (
                            capture_input(index, module, args, hook_kwargs)
                        ),
                        with_kwargs=True,
                    )
                )
            kwargs["output_hidden_states"] = False
            self.base_model(**kwargs)
        finally:
            for handle in handles:
                handle.remove()

        missing = [index for index in selected if index not in captured]
        if missing:
            raise RuntimeError(
                f"Context encoder did not execute selected layers {missing}"
            )
        ordered = tuple(captured[index] for index in selected)
        if self.offload_selected_layer_inputs_to_cpu:
            # Keep the tensors separate. Stacking here would allocate a second
            # full all-layer context tensor and defeat the memory bound.
            return ordered
        # Shape: (batch_size, selected_layers, seq_len, hidden_size).
        return torch.stack(ordered, dim=1)


class CTX_ENCODER_TYPE(str, Enum):
    EARLY_EXIT = "early_exit"
    EMBED_ONLY = "embed_only"
    PER_LAYER_ACTIVATIONS = "per_layer_activations"


CTX_ENCODER_CLS = {
    CTX_ENCODER_TYPE.EARLY_EXIT: EarlyExit,
    CTX_ENCODER_TYPE.EMBED_ONLY: EmbeddingOnly,
    CTX_ENCODER_TYPE.PER_LAYER_ACTIVATIONS: PerLayerActivations,
}
