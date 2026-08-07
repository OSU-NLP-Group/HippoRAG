# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Idefics2 model."""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_utils import PreTrainedModel
from transformers.models.idefics2.configuration_idefics2 import Idefics2Config
from transformers.utils import (
    add_start_docstrings,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)

if is_flash_attn_2_available():
    from flash_attn.bert_padding import unpad_input
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)


class Idefics2PerceiverConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the perceiver block.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        n_latents (`int`, *optional*, defaults to 64):
            Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).
        resampler_depth (`int`, *optional*, defaults to 3):
            Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (<= 3).
        n_heads (`int`, *optional*, defaults to 16):
            Number of heads in each Transformer block (for multi-headed self-attention).
        head_dim (`int`, *optional*, defaults to 96):
            Dimensionality of each head projection in the Transformer block.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key-value heads in the perceiver attention block.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    """

    model_type = "idefics2_perceiver"

    def __init__(
        self,
        input_size: int,
        num_blocks: int,
        num_self_attn_per_block: int,
        shared_weights: bool,
        intermediate_size_factor: int,
        hidden_act="silu",
        hidden_size=4096,
        rms_norm_eps=1e-06,
        n_latents=64,
        n_heads=16,
        head_dim=128,
        num_key_value_heads=4,
        attention_dropout=0.0,
        activation_checkpointing=False,
        modality_projection_chunk_size=0,
        **kwargs,
    ):
        self.num_blocks = num_blocks
        self.num_self_attn_per_block = num_self_attn_per_block
        self.shared_weights = shared_weights

        self.input_size = input_size
        self.intermediate_size_factor = intermediate_size_factor
        # for perceiver
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.n_latents = n_latents
        self.n_heads = n_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_dropout = attention_dropout
        self.activation_checkpointing = activation_checkpointing
        self.modality_projection_chunk_size = int(modality_projection_chunk_size)
        if self.modality_projection_chunk_size < 0:
            raise ValueError(
                "modality_projection_chunk_size must be non-negative"
            )
        if self.num_key_value_heads > self.n_heads:
            raise ValueError(
                f"num_key_value_heads={self.num_key_value_heads} must be less than or equal to"
                f" n_heads={self.n_heads}"
            )
        super().__init__(**kwargs)


class Idefics2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


IDEFICS2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Idefics2Config`] or [`Idefics2VisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Idefics2 Model outputting raw hidden-states without any specific head on top.",
    IDEFICS2_START_DOCSTRING,
)
class Idefics2PreTrainedModel(PreTrainedModel):
    config_class = Idefics2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Idefics2VisionAttention",
        "Idefics2MLP",
        "Idefics2PerceiverLayer",
        "Idefics2DecoderLayer",
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else 0.02
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Idefics2
class Idefics2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Idefics2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Idefics2PerceiverAttention(nn.Module):
    def __init__(self, config, layer_idx: int | None = None) -> None:
        """Perceiver Cross-Attention Module --> let long-form inputs be `context`, resampled embeddings be `latents`"""
        super().__init__()
        self.config = config
        self.layer_idx = None
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.is_causal = False

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        """
        Runs Perceiver Self-Attention, with special (context, latents) appended along the `seq` dimension!

        Args:
            latents (`torch.Tensor`): Tensor of shape [bsz, n_latents, embed_dim] representing fixed length latents to compress to.
            context (`torch.Tensor`): Tensor of shape [bsz, seq, embed_dim] representing long-form context to resample.
            attention_mask (`torch.Tensor`, *optional*): Tensor of shape [bsz, 1, seq, n_latents] representing attention mask.
            position_ids (`torch.LongTensor`, *optional*): Tensor of shape [bsz, seq] representing position indices of each input token.
            past_key_value (`Tuple[torch.Tensor]`, *optional*): Tuple of tensors containing cached key and value states.
            output_attentions (`bool`, *optional*, defaults to `False`): Whether to return attention weights.
            use_cache (`bool`, *optional*, defaults to `False`): Whether to use past_key_value for caching.
        """
        bsz, q_len, _ = latents.size()
        kv_seq_len = q_len + context.size()[1]

        hidden_states = torch.concat([context, latents], dim=-2)

        query_states = self.q_proj(latents)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, kv_seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, kv_seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# NO LONGER EXIST Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2 with MistralAttention->Idefics2PerceiverAttention,MistralFlashAttention->Idefics2PerceiverFlashAttention,Mistral->Idefics2
# TODO cyril: modular
class Idefics2PerceiverFlashAttention2(Idefics2PerceiverAttention):
    """
    Idefics2 flash attention module. This module inherits from `Idefics2PerceiverAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # Ignore copy
    def forward(
        self,
        latents: torch.Tensor,
        is_cross_attn: bool,
        context: torch.Tensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        bsz, q_len, _ = latents.size()
        query_states = self.q_proj(latents)
        if is_cross_attn:
            kv_inp = context
        else:
            kv_inp = latents

        key_states = self.k_proj(kv_inp)
        value_states = self.v_proj(kv_inp)

        # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        query_states = query_states.view(
            *latents.shape[:2], self.num_heads, self.head_dim
        )

        key_states = key_states.view(
            *kv_inp.shape[:2], self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            *kv_inp.shape[:2], self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            position_ids=position_ids,
            sliding_window=None,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Idefics2PerceiverSdpaAttention(Idefics2PerceiverAttention):
    """Memory-efficient equivalent of the eager Perceiver attention.

    The eager implementation repeats grouped K/V heads and materializes the
    complete attention-probability tensor.  For a 130k-token context those
    temporary tensors can exceed H100 memory during activation recomputation.
    PyTorch SDPA accepts grouped-query attention directly and selects a fused
    kernel on CUDA, while retaining the same context-plus-latents K/V sequence
    used by the eager implementation.
    """

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if output_attentions:
            raise ValueError(
                "SDPA Perceiver attention does not materialize attention weights"
            )

        bsz, q_len, _ = latents.size()
        hidden_states = torch.concat([context, latents], dim=-2)
        kv_seq_len = hidden_states.shape[1]

        query_states = self.q_proj(latents)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, kv_seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, kv_seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )

        dropout_p = self.attention_dropout if self.training else 0.0
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=False,
            enable_gqa=self.num_key_value_groups > 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)

        present_key_value = (
            (key_states, value_states) if use_cache else past_key_value
        )
        return attn_output, None, present_key_value


IDEFICS2_PERCEIVER_ATTENTION_CLASSES = {
    "eager": Idefics2PerceiverAttention,
    "sdpa": Idefics2PerceiverSdpaAttention,
    "flash_attention_2": Idefics2PerceiverFlashAttention2,
}


class Idefics2PerceiverLayer(nn.Module):
    def __init__(self, config, is_cross_attn: bool):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_latents = config.n_latents
        self.rms_norm_eps = config.rms_norm_eps
        self.is_cross_attn = is_cross_attn

        self.input_latents_layernorm = Idefics2RMSNorm(
            self.hidden_size, eps=self.rms_norm_eps
        )
        self.input_context_layernorm = (
            Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
            if self.is_cross_attn
            else torch.nn.Identity()
        )
        self.self_attn = IDEFICS2_PERCEIVER_ATTENTION_CLASSES[
            config._attn_implementation
        ](config)
        self.post_attention_layernorm = Idefics2RMSNorm(
            self.hidden_size, eps=self.rms_norm_eps
        )
        self.pre_ff_layernorm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.post_ff_layernorm = Idefics2RMSNorm(
            self.hidden_size, eps=self.rms_norm_eps
        )
        self.mlp = Idefics2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size * 4,
            output_size=config.hidden_size,
            hidden_act=config.hidden_act,
        )

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            latents (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            context (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = latents

        latents = self.input_latents_layernorm(latents)
        context = self.input_context_layernorm(context)

        latents, self_attn_weights, present_key_value = self.self_attn(
            latents=latents,
            is_cross_attn=self.is_cross_attn,
            context=context,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        latents = self.post_attention_layernorm(latents)
        latents = residual + latents
        residual = latents

        # latents = self.post_attention_layernorm(latents)
        latents = self.pre_ff_layernorm(latents)
        latents = self.mlp(latents)
        latents = self.post_ff_layernorm(latents)
        latents = residual + latents

        outputs = (latents,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


IDEFICS2_INPUTS_DOCSTRING = r"""
    Args:
        context (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_dim)`):
            The hidden states of the image after vision encoder and modality projection.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
"""


@add_start_docstrings(
    "Idefics2 perceiver resampler model that performs `depth` blocks of cross-attention with a fixed ",
    "`n_latents` inputs to decrease embedding sequence length. The Resampler acts as a form of learned pooling and ",
    "is derived from [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)",
    IDEFICS2_START_DOCSTRING,
)
class Idefics2PerceiverResampler(Idefics2PreTrainedModel):
    _supports_sdpa = True
    config_class = Idefics2PerceiverConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_blocks = config.num_blocks
        self.num_self_attn_per_block = config.num_self_attn_per_block
        self.shared_weights = config.shared_weights
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act
        self.n_latents = config.n_latents
        self.rms_norm_eps = config.rms_norm_eps

        # Create Latents for Perceiver
        self.latents_q = nn.Parameter(torch.randn(self.n_latents, self.hidden_size))

        # First block
        assert config.num_blocks > 0
        first_x_attn = [Idefics2PerceiverLayer(config, is_cross_attn=True)]
        first_self_attn_block = [
            Idefics2PerceiverLayer(config, is_cross_attn=False)
            for _ in range(config.num_self_attn_per_block)
        ]

        self.layers = nn.ModuleList(first_x_attn + first_self_attn_block)
        for layer_idx in range(1, config.num_blocks):
            # cross-attention at the beginning of each block
            if self.shared_weights:
                if layer_idx == 1:
                    second_x_attn = Idefics2PerceiverLayer(config, is_cross_attn=True)
                x_attn = second_x_attn
            else:
                x_attn = Idefics2PerceiverLayer(config, is_cross_attn=True)
            self.layers.append(x_attn)

            # self-attention
            for i in range(config.num_self_attn_per_block):
                if self.shared_weights:
                    self_attn = first_self_attn_block[i]
                else:
                    self_attn = Idefics2PerceiverLayer(config, is_cross_attn=False)
                self.layers.append(self_attn)

        self.layernorm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.activation_checkpointing = bool(
            getattr(config, "activation_checkpointing", False)
        )

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # seq embed -> bsz seq embed
        if position_ids is None:
            bsz = context.shape[0]
        else:
            # flattened packed sequence
            bsz = int(torch.where(position_ids == 0, 1, 0).sum().item())

        latents = self.latents_q.unsqueeze(0).expand((bsz, *self.latents_q.size()))

        if not self._use_flash_attention_2:
            if position_ids is not None and attention_mask is None:
                flat_position_ids = position_ids.flatten()
                flat_context = context.reshape(-1, context.shape[-1])
                if flat_context.shape[0] != flat_position_ids.numel():
                    raise ValueError(
                        "Packed Perceiver context/position lengths disagree: "
                        f"context has {flat_context.shape[0]} tokens but "
                        f"position_ids has {flat_position_ids.numel()} tokens."
                    )
                starts = torch.nonzero(flat_position_ids == 0, as_tuple=False).flatten()
                ends = torch.cat(
                    [
                        starts[1:],
                        torch.tensor(
                            [flat_position_ids.numel()],
                            device=flat_position_ids.device,
                            dtype=starts.dtype,
                        ),
                    ]
                )
                lengths = ends - starts
                max_len = int(lengths.max().item())
                padded_context = flat_context.new_zeros(
                    (len(starts), max_len, flat_context.shape[-1])
                )
                context_attention_mask = torch.zeros(
                    (len(starts), max_len),
                    device=flat_context.device,
                    dtype=torch.long,
                )
                for i, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
                    seq_len = end - start
                    padded_context[i, :seq_len] = flat_context[start:end]
                    context_attention_mask[i, :seq_len] = 1
                context = padded_context
                position_ids = None
                attention_mask = context_attention_mask

            if attention_mask is not None:
                latent_attention_mask = torch.ones(
                    (attention_mask.shape[0], self.n_latents),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = _prepare_4d_attention_mask(
                    torch.cat([attention_mask, latent_attention_mask], dim=1),
                    latents.dtype,
                    tgt_len=self.n_latents,
                )

        compressed_context = latents

        cu_seq_lens_q = torch.tensor(
            [self.n_latents] * (bsz + 1), device=context.device, dtype=torch.int32
        ) * torch.arange(bsz + 1, device=context.device, dtype=torch.int32)
        max_length_q = self.n_latents
        # cu_seq_lens_k = None
        # max_length_k = None
        if attention_mask is not None and self._use_flash_attention_2:
            logger.warning_once("Using attention mask for resampler")
            context, _, cu_seq_lens_k, max_length_k, _ = unpad_input(
                context, attention_mask
            )
            context = context.unsqueeze(0)
            position_ids = True  # goes down flash attn path that uses cu_seq_lens

        elif position_ids is not None:
            logger.warning_once("Using position ids for resampler")

            position_ids = position_ids.flatten()
            indices = torch.arange(
                position_ids.size(0), device=position_ids.device, dtype=torch.int32
            )
            # [bsz + 1]
            cu_seq_lens_k = torch.cat(
                (
                    indices[position_ids == 0],
                    torch.tensor(
                        position_ids.size(),
                        device=position_ids.device,
                        dtype=torch.int32,
                    ),
                )
            )

            max_length_k = position_ids.max() + 1

        else:
            cu_seq_lens_k = None
            max_length_k = None
        x_attn_kwargs = dict(
            position_ids=position_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
        )
        self_attn_position_ids = torch.arange(
            self.n_latents, device=context.device, dtype=torch.int32
        ).repeat(1, bsz)
        self_attn_kwargs = dict(
            # attention_mask=self_attn_mask,
            position_ids=self_attn_position_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_q,
            max_length_q=max_length_q,
            max_length_k=max_length_q,
        )
        for i, layer in enumerate(self.layers):
            inp_kwargs = dict(
                latents=compressed_context,
                context=context,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            if layer.is_cross_attn:
                attn_kwargs = {**inp_kwargs, **x_attn_kwargs}
            else:
                attn_kwargs = {**inp_kwargs, **self_attn_kwargs}

            if self.training and self.activation_checkpointing:
                checkpoint_attn_kwargs = (
                    x_attn_kwargs if layer.is_cross_attn else self_attn_kwargs
                )

                def checkpointed_layer(
                    latents,
                    layer_context,
                    checkpoint_layer=layer,
                    checkpoint_kwargs=checkpoint_attn_kwargs,
                ):
                    return checkpoint_layer(
                        latents=latents,
                        context=layer_context,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                        **checkpoint_kwargs,
                    )[0]

                compressed_context = checkpoint(
                    checkpointed_layer,
                    compressed_context,
                    context,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(**attn_kwargs)
                compressed_context = layer_outputs[0]

        compressed_context = self.layernorm(compressed_context)

        return compressed_context


class Idefics2Perceiver(Idefics2PreTrainedModel):
    def __init__(
        self,
        encoder_config: Idefics2PerceiverConfig,
        decoder_config: Idefics2PerceiverConfig,
    ):
        super().__init__(encoder_config)
        self.modality_projection = Idefics2MLP(
            hidden_size=encoder_config.input_size,
            intermediate_size=encoder_config.intermediate_size_factor
            * encoder_config.input_size,
            output_size=encoder_config.hidden_size,
            hidden_act=encoder_config.hidden_act,
        )
        self.encoder = Idefics2PerceiverResampler._from_config(encoder_config)
        self.decoder = Idefics2PerceiverResampler._from_config(decoder_config)

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ):
        if position_ids is None:
            bsz = context.shape[0]
        else:
            bsz = torch.where(position_ids == 0, 1, 0).sum()
        activation_checkpointing = self.training and bool(
            getattr(self.config, "activation_checkpointing", False)
        )
        projection_chunk_size = int(
            getattr(self.config, "modality_projection_chunk_size", 0)
        )
        context_chunks = (
            context.split(projection_chunk_size, dim=-2)
            if projection_chunk_size
            and context.shape[-2] > projection_chunk_size
            else (context,)
        )
        projected_chunks = []
        for context_chunk in context_chunks:
            projected_chunks.append(
                checkpoint(
                    self.modality_projection,
                    context_chunk,
                    use_reentrant=False,
                )
                if activation_checkpointing
                else self.modality_projection(context_chunk)
            )
        projected_inputs = (
            projected_chunks[0]
            if len(projected_chunks) == 1
            else torch.cat(projected_chunks, dim=-2)
        )

        # [bsz, n_latents, dim]
        latents = self.encoder(
            context=projected_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        latent_position_ids = torch.arange(
            self.encoder.n_latents, device=context.device
        ).unsqueeze(0)
        latent_position_ids = torch.tile(latent_position_ids, (1, bsz))
        outputs = self.decoder(latents, position_ids=latent_position_ids)

        return outputs


__all__ = [
    "Idefics2Perceiver",
]
