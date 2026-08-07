import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from einops import rearrange, repeat, unpack
from jaxtyping import Float, Integer
from torch import Tensor, nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)

from ctx_to_lora.configs import (
    AggregatorArguments,
)
from ctx_to_lora.modeling.idefics2 import Idefics2Perceiver, Idefics2PerceiverConfig
from ctx_to_lora.pooling import POOL_FN
from ctx_to_lora.utils import (
    get_num_layers,
)

logger = logging.getLogger()


class AGGREGATOR_TYPE(str, Enum):
    POOLER = "pooler"
    PERCEIVER = "perceiver"


@dataclass
class AggregatorConfig:
    aggregator_type: AGGREGATOR_TYPE
    num_layers: int
    num_modules: int
    num_extra_modules: int
    output_size: int
    feature_size: int

    # pooler
    pooling_type: POOL_FN

    # perceiver
    num_latent_factor: int
    lora_r: int
    per_rank_gen: bool

    n_latent_queries: int
    num_blocks: int
    num_self_attn_per_block: int
    shared_weights: bool
    layer_to_layer_ctx_encoder: bool
    perceiver_attn_implementation: Literal["eager", "sdpa"] = "eager"
    perceiver_activation_checkpointing: bool = False
    perceiver_modality_projection_chunk_size: int = 0


def get_aggregator_config(
    model: PreTrainedModel,
    ctx_encoder_model_config: PretrainedConfig,
    layer_to_layer_ctx_encoder: bool,
    output_size: int,
    num_modules: int,
    num_extra_modules: int,
    lora_r: int,
    per_rank_gen: bool,
    aggregator_args: AggregatorArguments,
    num_layers: int | None = None,
):
    return AggregatorConfig(
        feature_size=ctx_encoder_model_config.hidden_size,
        output_size=output_size,
        num_layers=num_layers if num_layers is not None else get_num_layers(model),
        num_modules=num_modules,
        num_extra_modules=num_extra_modules,
        lora_r=lora_r,
        per_rank_gen=per_rank_gen,
        layer_to_layer_ctx_encoder=layer_to_layer_ctx_encoder,
        **vars(aggregator_args),
    )


class Perceiver(nn.Module):
    """perceiver w/ bottleneck size = n_modules * n_layers"""

    def __init__(
        self,
        feature_size,
        output_size,
        num_layers,
        num_modules,
        num_extra_modules,
        per_rank_gen,
        lora_r,
        num_latent_factor,  # unused
        layer_to_layer_ctx_encoder,
        n_latent_queries,
        perceiver_attn_implementation="eager",
        perceiver_activation_checkpointing=False,
        perceiver_modality_projection_chunk_size=0,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert num_extra_modules == 0
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.num_extra_modules = num_extra_modules
        self.per_rank_gen = per_rank_gen
        self.r = lora_r if self.per_rank_gen else 1
        n_output_queries = num_layers * (num_modules * self.r + num_extra_modules)
        self.layer_to_layer = layer_to_layer_ctx_encoder
        if self.layer_to_layer:
            n_output_queries = num_modules * self.r + num_extra_modules
        self.config = Idefics2PerceiverConfig(
            input_size=feature_size,
            num_blocks=kwargs["num_blocks"],
            num_self_attn_per_block=kwargs["num_self_attn_per_block"],
            shared_weights=kwargs["shared_weights"],
            n_latents=n_latent_queries,
            intermediate_size_factor=4,
            hidden_size=output_size,
            attn_implementation=perceiver_attn_implementation,
            activation_checkpointing=perceiver_activation_checkpointing,
            modality_projection_chunk_size=(
                perceiver_modality_projection_chunk_size
            ),
        )
        self.decoder_config = Idefics2PerceiverConfig(
            input_size=output_size,
            num_blocks=1,
            num_self_attn_per_block=0,
            shared_weights=False,
            n_latents=n_output_queries,
            intermediate_size_factor=4,
            hidden_size=output_size,
            attn_implementation=perceiver_attn_implementation,
            activation_checkpointing=perceiver_activation_checkpointing,
            modality_projection_chunk_size=0,
        )
        self.perceiver = Idefics2Perceiver(self.config, self.decoder_config)
        self.iterative_mode = False

    def enable_iterative_mode(self, x: bool):
        self.iterative_mode = x

    def forward(
        self,
        ctx_features: Float[Tensor, "bs seq_len feature_dim"]
        | Float[Tensor, "bs x seq_len feature_dim"],
        ctx_attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "bs seq_len"] | None = None,
    ):
        if self.layer_to_layer and not self.iterative_mode:
            if ctx_attn_mask is not None:
                ctx_attn_mask = repeat(
                    ctx_attn_mask,
                    "bs seq_len -> (num_layers bs) seq_len",
                    num_layers=self.num_layers,
                )
                ctx_features = rearrange(
                    ctx_features,
                    "bs num_layers seq_len feature_dim -> (num_layers bs) seq_len feature_dim",
                )
            if ctx_position_ids is not None:
                ctx_position_ids = repeat(
                    ctx_position_ids,
                    "1 seq_len -> 1 (num_layers seq_len)",
                    num_layers=self.num_layers,
                )
                ctx_features = rearrange(
                    ctx_features,
                    "1 num_layers seq_len feature_dim -> 1 (num_layers seq_len) feature_dim",
                )

        x = self.perceiver(ctx_features, ctx_attn_mask, ctx_position_ids)

        if self.layer_to_layer and self.iterative_mode:
            lora_x = rearrange(
                x,
                "bs (n_modules r) d -> bs n_modules r d",
                n_modules=self.num_modules,
                r=self.r,
            )
            return lora_x, None

        if self.layer_to_layer:
            per_layer_size = self.num_modules * self.r + self.num_extra_modules
            x = rearrange(
                x,
                "(num_layers bs) (per_layer_sz) d -> bs (num_layers per_layer_sz) d",
                num_layers=self.num_layers,
                per_layer_sz=per_layer_size,
            )
        lora_x, extra_x = unpack(
            x,
            [
                [self.num_layers * self.num_modules * self.r],
                [self.num_layers * self.num_extra_modules],
            ],
            "bs * feature_dim",
        )
        lora_x = rearrange(
            lora_x,
            "bs (n_layers n_modules r) d -> bs n_layers n_modules r d",
            n_modules=self.num_modules,
            n_layers=self.num_layers,
            r=self.r,
        )
        if not self.per_rank_gen:
            lora_x = lora_x.squeeze(3)

        extra_x = rearrange(
            extra_x,
            "bs (n_layers n_extra_modules) d -> bs n_layers n_extra_modules d",
            n_extra_modules=self.num_extra_modules,
            n_layers=self.num_layers,
        )

        return lora_x, extra_x


AGGREGATOR_CLS = {
    AGGREGATOR_TYPE.PERCEIVER: Perceiver,
}
