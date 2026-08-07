import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from math import sqrt
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint
from einops import unpack
from einops.layers.torch import EinMix as Mix
from jaxtyping import Float, Integer
from peft import (
    LoraConfig,
    LoraRuntimeConfig,
    PeftConfig,
    PeftModel,
)
from peft.tuners._buffer_dict import BufferDict
from peft.utils import PeftType, TaskType
from torch import Tensor, nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.models.modernbert.modeling_modernbert import ModernBertModel

from ctx_to_lora.configs import (
    AggregatorArguments,
    CtxEncoderArguments,
    HypernetArguments,
    RepositoryMergerArguments,
)
from ctx_to_lora.data.processing import tokenize_ctx_text
from ctx_to_lora.model_loading import (
    get_model,
    get_tokenizer,
)
from ctx_to_lora.modeling.aggregator import (
    AGGREGATOR_CLS,
    AggregatorConfig,
    get_aggregator_config,
)
from ctx_to_lora.modeling.ctx_encoder import (
    CTX_ENCODER_CLS,
    CTX_ENCODER_TYPE,
    _attention_implementation,
    _gemma4_flex_kernel_options,
)
from ctx_to_lora.modeling.lora_layer import (
    apply_lora_to_layers,
    lora_forward,
    lora_forward_packed,
)
from ctx_to_lora.modeling.lora_merger import combine_lora
from ctx_to_lora.modeling.repo_lora_merger import (
    RepoMergeMethod,
    RepositoryLoRAMerger,
    RepositoryMergerConfig,
)
from ctx_to_lora.utils import (
    get_layers,
    get_num_layers,
    get_peft_in_out_features,
    get_peft_modules,
    layer_index_from_module_name,
    peft_module_target_type,
)

logger = logging.getLogger()


def text_config_attr(config, name: str):
    if hasattr(config, name):
        return getattr(config, name)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, name):
        return getattr(text_config, name)
    raise AttributeError(f"{type(config).__name__} has no {name!r} or text_config.{name}")


@dataclass
class HypernetConfig:
    latent_size: int
    use_light_weight_lora: bool
    light_weight_latent_size: int
    per_rank_gen: bool
    use_per_rank_bias: bool
    use_bias: bool
    per_layer_processing: bool
    use_token_mixing: bool
    num_pre_head_layers: int
    dropout_rate: float

    lora_config: LoraConfig
    extra_modules: list[str] | None
    base_hidden_size: int

    layer_indices: Iterable[int]
    feature_sizes: tuple[dict[str, int], dict[str, int]]
    aggregator_config: AggregatorConfig
    repo_merger_config: RepositoryMergerConfig


def get_hypernet_config(
    model: PreTrainedModel,
    ctx_encoder_model_config: PretrainedConfig,
    hypernet_args: HypernetArguments,
    aggregator_args: AggregatorArguments,
    ctx_encoder_args: CtxEncoderArguments,
    repo_merger_args: RepositoryMergerArguments,
):
    ctx_text_config = getattr(ctx_encoder_model_config, "text_config", ctx_encoder_model_config)
    num_modules = 0
    lora_config = getattr(model, "peft_config", None)
    if lora_config is not None:
        lora_config = lora_config["default"]
        feature_sizes = get_peft_in_out_features(model, peft_config=lora_config)
        num_modules += len(feature_sizes[0])
    else:
        feature_sizes = (None, None)
    num_extra_modules = len(hypernet_args.extra_modules or [])
    if lora_config is not None and getattr(lora_config, "target_module_shapes", None):
        indices = sorted(
            {
                layer_index_from_module_name(module_info["name"])
                for module_info in get_peft_modules(model, lora_config)
            }
        )
        if not indices:
            raise ValueError(
                "target_module_shapes did not match any PEFT modules: "
                f"{lora_config.target_module_shapes}"
            )
        indices = torch.tensor(indices, device=model.device, dtype=torch.long)
    else:
        indices = torch.arange(get_num_layers(model), device=model.device)
    hypernet_config_values = vars(hypernet_args).copy()
    # Execution/compilation policy belongs to train.py, not the serialized
    # mathematical HypernetConfig.
    hypernet_config_values.pop("compile_hypernet", None)
    hypernet_config_values.pop("compile_base_model", None)
    return HypernetConfig(
        **hypernet_config_values,
        base_hidden_size=text_config_attr(model.config, "hidden_size"),
        lora_config=lora_config,
        layer_indices=indices,
        feature_sizes=feature_sizes,
        aggregator_config=get_aggregator_config(
            model,
            ctx_text_config,
            ctx_encoder_args.ctx_encoder_type == CTX_ENCODER_TYPE.PER_LAYER_ACTIVATIONS,
            hypernet_args.latent_size,
            num_modules,
            num_extra_modules,
            lora_config.r,
            hypernet_args.per_rank_gen,
            aggregator_args,
            num_layers=len(indices),
        ),
        repo_merger_config=RepositoryMergerConfig(
            method=repo_merger_args.repo_merge_method,
            output_rank=repo_merger_args.repo_output_rank,
            ties_keep_fraction=repo_merger_args.ties_keep_fraction,
            ties_sign_method=repo_merger_args.ties_sign_method,
            ties_merge_type=repo_merger_args.ties_merge_type,
            ties_merge_scale=repo_merger_args.ties_merge_scale,
            knots_concat_across_output=repo_merger_args.knots_concat_across_output,
            knots_singular_value_epsilon=(
                repo_merger_args.knots_singular_value_epsilon
            ),
            retrieval_top_k=repo_merger_args.retrieval_top_k,
            fusion_num_blocks=repo_merger_args.repo_fusion_num_blocks,
            fusion_num_heads=repo_merger_args.repo_fusion_num_heads,
            svd_oversample=repo_merger_args.repo_svd_oversample,
            svd_power_iterations=repo_merger_args.repo_svd_power_iterations,
            svd_exact_max_dim=repo_merger_args.repo_svd_exact_max_dim,
            svd_singular_value_epsilon=(
                repo_merger_args.repo_svd_singular_value_epsilon
            ),
            svd_seed=repo_merger_args.repo_svd_seed,
        ),
    )


def get_init_peft_weights(model: PeftModel, peft_config: PeftConfig = None):
    if peft_config is None:
        peft_config = model.peft_config["default"]
    peft_weights = {}
    adapter_name = "default"
    for module_info in get_peft_modules(model, peft_config):
        module_name = module_info["name"]
        module = module_info["module"]
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        assert isinstance(module.base_layer, nn.Linear), (
            "all modules should be a leave module that is Linear layer"
        )

        name = peft_module_target_type(module_name, module)
        peft_weights.setdefault(name, {})

        for submodule_name, submodule in module.named_modules():
            if not isinstance(submodule, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue

            if adapter_name not in submodule:
                continue

            if submodule_name not in peft_weights[name]:
                peft_weights[name][submodule_name] = submodule[adapter_name]
            else:
                smod1 = peft_weights[name][submodule_name]
                smod2 = submodule[adapter_name]
                assert type(smod1) == type(smod2)

    return peft_weights


class ResMLPBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_rate: float = 0,
    ):
        super().__init__()
        layers = []
        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(x)


class ResMLPBlockPerLayer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        layers = [
            nn.LayerNorm(input_size),
            Mix(
                "bs n_layers n_modules r d_in -> bs n_layers n_modules r d_hid",
                weight_shape="n_layers d_in d_hid",
                bias_shape="n_layers d_hid",
                n_layers=n_layers,
                d_in=input_size,
                d_hid=hidden_size,
            ),
            nn.SiLU(),
            Mix(
                "bs n_layers n_modules r d_hid -> bs n_layers n_modules r d_out",
                weight_shape="n_layers d_hid d_out",
                bias_shape="n_layers d_out",
                n_layers=n_layers,
                d_hid=hidden_size,
                d_out=output_size,
            ),
            nn.LayerNorm(output_size),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)


class HyperLoRA(nn.Module):
    def __init__(self, config: HypernetConfig):
        super().__init__()

        # aggregator output [bs, n_layers, n_modules, feature_dim]
        # by mixing the pooled features with layer embs and module embs (for pooling)
        # or via a perceiver w/ bottleneck size = n_modules * n_layers
        self.config = config
        logger.debug(f"HyperLoRA config: {self.config}")
        self.iterative_mode = False
        self._init_model()

    def _init_model(self):
        # Checkpoints produced before repository composition was introduced use
        # Doc-to-LoRA's original rank concatenation semantics.
        if getattr(self.config, "repo_merger_config", None) is None:
            self.config.repo_merger_config = RepositoryMergerConfig()
        self.agg_config = self.config.aggregator_config
        self.aggregator = AGGREGATOR_CLS[self.agg_config.aggregator_type](
            **vars(self.agg_config)
        )

        self.lora_config = self.config.lora_config
        self.r = self.lora_config.r

        self.target_modules = (
            tuple(sorted(self.config.feature_sizes[0])) if self.lora_config else None
        )
        self.num_modules = len(self.target_modules) if self.target_modules else 0
        self.extra_modules = (
            self.config.extra_modules if self.config.extra_modules else None
        )
        self.num_extra_modules = len(self.extra_modules) if self.extra_modules else 0
        self.layer_indices = self.config.layer_indices
        self.n_layers = len(self.layer_indices)

        self.d_in, self.d_out = self.config.feature_sizes
        self.d_latent = self.config.latent_size
        self.repo_merger = RepositoryLoRAMerger(
            self.config.repo_merger_config,
            latent_size=self.d_latent,
        )

        if self.target_modules:
            if self.config.per_layer_processing:
                layers = [
                    ResMLPBlockPerLayer(
                        self.n_layers,
                        self.d_latent,
                        self.d_latent * 4,
                        self.d_latent,
                    )
                    for _ in range(self.config.num_pre_head_layers)
                ]
            else:
                layers = [
                    ResMLPBlock(
                        input_size=self.config.latent_size,
                        hidden_size=self.config.latent_size * 4,
                        output_size=self.config.latent_size,
                        dropout_rate=getattr(self.config, "dropout_rate", 0),
                    )
                    for _ in range(self.config.num_pre_head_layers)
                ]

            self.layers = nn.Sequential(*layers)

            self.d_lora = max(self.d_in[m] + self.d_out[m] for m in self.target_modules)

            self.bias_A = nn.ParameterDict(
                {
                    m: nn.Parameter(
                        torch.normal(
                            0,
                            0.2 / (self.d_in[m] * self.r) ** 0.5,
                            (self.n_layers, self.r, self.d_in[m]),
                        )
                    )
                    for m in self.target_modules
                }
            )
            self.bias_B = nn.ParameterDict(
                {
                    m: nn.Parameter(torch.zeros((self.n_layers, self.r, self.d_out[m])))
                    for m in self.target_modules
                }
            )

            self.scaler_A = nn.ParameterDict(
                {
                    m: nn.Parameter(torch.ones((1, self.n_layers, self.r, 1)))
                    for m in self.target_modules
                }
            )
            self.scaler_B = nn.ParameterDict(
                {
                    m: nn.Parameter(torch.zeros((1, self.n_layers, self.r, 1)))
                    for m in self.target_modules
                }
            )

            n_modules = len(self.target_modules)
            # have to do this otherwise doesnt work with adamw_torch_fused
            # has something to do with the bias shape (n_modules r d_lora)
            # when n_modules == 1, adamw_torch_fused complains about device/layout
            # but when n_modules > 1, it works fine
            if n_modules == 1:
                self.head = Mix(
                    "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
                    weight_shape="n_layers d_latent d_lora",
                    bias_shape=None,  # no bias
                    n_layers=len(self.layer_indices),
                    d_latent=self.config.latent_size,
                    r=self.config.lora_config.r,
                    d_lora=self.d_lora,
                )
            else:
                self.head = Mix(
                    "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
                    weight_shape="n_layers n_modules d_latent d_lora",
                    bias_shape=None,  # no bias
                    n_layers=len(self.layer_indices),
                    n_modules=n_modules,
                    d_latent=self.config.latent_size,
                    r=self.config.lora_config.r,
                    d_lora=self.d_lora,
                )

    def get_head_bias(self):
        bias_dict = dict()
        for module in self.target_modules:
            bias_A = self.bias_A[module]
            bias_B = self.bias_B[module]

            bias_dict[module] = dict(A=bias_A, B=bias_B)
        return bias_dict

    def _to_lora_dict(
        self, flat_loras: Float[Tensor, "bs n_layers n_modules r max_io_dim"]
    ) -> dict[str, dict[str, Float[Tensor, "bs n_layers r _"]]]:
        if self.target_modules is None:
            return None
        # list of [bs, n_layers, r, in_d_outim]
        # and in_d_outim might vary across modules
        loras = unpack(
            flat_loras,
            [[] for _ in range(len(self.target_modules))],
            "bs n_layers * r max_io_dim",
        )

        # dict of {module:
        #   {A: [bs, n_layers, r, d_inim],
        #    B: [bs, n_layers, r, d_outim]}}
        lora_dict = dict()
        for module, lora in zip(self.target_modules, loras):
            A, B = unpack(
                lora[..., : self.d_in[module] + self.d_out[module]],
                [[self.d_in[module]], [self.d_out[module]]],
                "bs n_layers r *",
            )

            # The learned repository fusion emits a fixed repository rank,
            # while the ordinary Doc-to-LoRA decoder emits the base chunk
            # rank. Reuse the same learned per-rank scalers cyclically so this
            # remains the same decoder rather than a second adapter head.
            scaler_A = self._scaler_for_rank(self.scaler_A[module], A.shape[2])
            scaler_B = self._scaler_for_rank(self.scaler_B[module], B.shape[2])
            # apparently doing A * scaler_A is slow due to broadcasting
            A = torch.einsum("ijkl,ijkl->ijkl", A, scaler_A)
            B = torch.einsum("ijkl,ijkl->ijkl", B, scaler_B)

            lora_dict[module] = dict(A=A, B=B)

        return lora_dict

    @staticmethod
    def _scaler_for_rank(scaler: Tensor, rank: int) -> Tensor:
        base_rank = scaler.shape[2]
        if rank == base_rank:
            return scaler
        repetitions = (rank + base_rank - 1) // base_rank
        return scaler.repeat(1, 1, repetitions, 1)[:, :, :rank]

    def _to_layernorm_dict(
        self, flat_layernorms: Float[Tensor, "bs n_layers n_modules hidden_size"]
    ) -> dict[str, Float[Tensor, "bs n_layers hidden_size"]]:
        if self.extra_modules is None:
            return None
        layernorms = unpack(
            flat_layernorms,
            [[] for _ in range(len(self.extra_modules))],
            "bs n_layers * hidden_size",
        )
        return {k: v for k, v in zip(self.extra_modules, layernorms)}

    def enable_iterative_mode(self, x: bool):
        self.iterative_mode = x
        self.aggregator.enable_iterative_mode(x)

    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
    ):
        lora_emb = self.generate_chunk_latents(features, attn_mask, position_ids)

        # [bs, n_layers, n_modules, r, max_in_d_outim]
        flat_loras = self.decode_chunk_latents(lora_emb)
        flat_layernorms = None

        return flat_loras, flat_layernorms

    def generate_chunk_latents(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"] | tuple[Tensor, ...],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
    ) -> Tensor:
        """Run the unchanged Doc-to-LoRA aggregator and pre-head blocks."""

        # [bs, n_layers, n_total_modules, r, feature_dim]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.aggregator.layer_to_layer and self.iterative_mode:
                # iterative inference
                # Tensor features are [bs, num_layers, seq_len, feature_dim].
                # The memory-bounded path instead supplies one CPU tensor per
                # selected layer so checkpointing retains CPU inputs, not the
                # complete long-context activation stack on the GPU.
                cpu_layer_features = isinstance(features, (tuple, list))
                n_layers = len(features) if cpu_layer_features else features.shape[1]
                per_layer_embeddings = []
                for i in range(n_layers):
                    layer_features = features[i] if cpu_layer_features else features[:, i]
                    if self.training:
                        def aggregate_one(value):
                            if value.device.type == "cpu":
                                value = value.to(
                                    device=next(self.aggregator.parameters()).device,
                                    non_blocking=False,
                                )
                            return self.aggregator(
                                value, attn_mask, position_ids
                            )[0]

                        layer_embedding = checkpoint(
                            aggregate_one,
                            layer_features,
                            use_reentrant=False,
                        )
                    else:
                        if layer_features.device.type == "cpu":
                            layer_features = layer_features.to(
                                device=next(self.aggregator.parameters()).device,
                                non_blocking=False,
                            )
                        layer_embedding, _ = self.aggregator(
                            layer_features, attn_mask, position_ids
                        )
                    per_layer_embeddings.append(layer_embedding)
                # Stacking instead of assigning into a preallocated tensor keeps
                # the autograd connection to the shared Perceiver parameters.
                lora_emb = torch.stack(per_layer_embeddings, dim=1)

            else:
                # batched inference
                lora_emb, _ = self.aggregator(features, attn_mask, position_ids)

        if self.target_modules:
            if lora_emb.dim() == 4:
                lora_emb = lora_emb.unsqueeze(3).expand(
                    -1, -1, -1, self.config.lora_config.r, -1
                )
            lora_emb = self.layers(lora_emb)
            norm = torch.norm(lora_emb, dim=-1, keepdim=True)
            lora_emb = lora_emb / norm
        return lora_emb

    def decode_chunk_latents(self, lora_emb: Tensor) -> Tensor | None:
        if self.target_modules is None:
            return None
        return self.head(lora_emb)

    def generate_weights(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
        return_latents: bool = False,
    ):
        chunk_latents = self.generate_chunk_latents(features, attn_mask, position_ids)
        flat_loras = self.decode_chunk_latents(chunk_latents)
        result = (self._to_lora_dict(flat_loras), None)
        if return_latents:
            return (*result, chunk_latents)
        return result

    def merge_repository_loras(
        self,
        chunk_loras: dict[str, dict[str, Tensor]],
        n_ctx_chunks: Tensor,
        chunk_latents: Tensor | None = None,
        scalers: Tensor | None = None,
        bias_scaler: float | None = None,
    ) -> dict[str, dict[str, Tensor]]:
        bias = self.get_head_bias() if self.config.use_bias else None
        return self.repo_merger(
            chunk_loras,
            n_ctx_chunks,
            lora_bias=bias,
            chunk_latents=chunk_latents,
            latent_decoder=lambda latents: self._to_lora_dict(
                self.decode_chunk_latents(latents)
            ),
            scalers=scalers,
            bias_scaler=bias_scaler,
        )


class ModulatedPretrainedModel(nn.Module):
    def __init__(
        self,
        base_model: PeftModel,
        hypernet_config: HypernetConfig,
        ctx_encoder_args: CtxEncoderArguments,
        use_base_input_as_ctx: bool = False,
        # need non-packed inputs for generation
        use_sequence_packing: bool = True,
        user_defined_scaling: float = 1,
        inp_compressor=None,
    ):
        assert not use_base_input_as_ctx
        super().__init__()
        self.device = base_model.device
        self.peft_config = base_model.peft_config["default"]
        self.hypernet_config = hypernet_config
        self.ctx_encoder_args = ctx_encoder_args
        self.use_base_input_as_ctx = use_base_input_as_ctx
        self.use_sequence_packing = use_sequence_packing
        self.user_defined_scaling = user_defined_scaling
        self.inp_compressor = inp_compressor
        self.model_accepts_loss_kwargs = True
        self.generated_loras = None
        self.generated_chunk_latents = None

        self.register_module("base_model", base_model)
        self._init_model()
        self._bias_hyper_init()

    @property
    def supports_gradient_checkpointing(self):
        return True

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                return self.base_model.gradient_checkpointing_enable()
            return self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        base = self.base_model.get_base_model()
        if hasattr(base, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                return base.gradient_checkpointing_enable()
            return base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        raise AttributeError("base model does not support gradient checkpointing")

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            return self.base_model.gradient_checkpointing_disable()
        base = self.base_model.get_base_model()
        if hasattr(base, "gradient_checkpointing_disable"):
            return base.gradient_checkpointing_disable()
        raise AttributeError("base model does not support gradient checkpointing")

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        train: bool = True,
        base_model_kwargs: dict = None,
        use_flash_attn: bool = True,
        **kwargs: Any,
    ):
        lora_config = state_dict["hypernet_config"].lora_config
        print(f"lora_config: {lora_config}")
        model_name_or_path = state_dict["base_model_name_or_path"]
        base_model = get_model(
            model_name_or_path,
            train=train,
            requires_grad=False,
            peft_config=lora_config,
            model_kwargs=base_model_kwargs,
            use_flash_attn=use_flash_attn,
        )
        hypernet_config = state_dict["hypernet_config"]
        if getattr(hypernet_config, "num_pre_head_layers", None) is None:
            hypernet_config.num_pre_head_layers = 4
        if getattr(hypernet_config, "use_per_rank_bias", None) is None:
            hypernet_config.use_per_rank_bias = False
        if getattr(hypernet_config, "use_bias", None) is None:
            hypernet_config.use_bias = True
        if getattr(hypernet_config, "repo_merger_config", None) is None:
            hypernet_config.repo_merger_config = RepositoryMergerConfig()
        ctx_encoder_args = state_dict["ctx_encoder_args"]
        model = cls(base_model, hypernet_config, ctx_encoder_args, **kwargs)
        model.load_state_dict(state_dict)
        return model

    def patch_lora_forward(self):
        lora_forward_fn = (
            lora_forward_packed if self.use_sequence_packing else lora_forward
        )
        allowed_layers = {int(i) for i in self.hypernet.layer_indices}
        for module_info in get_peft_modules(self.base_model, self.peft_config):
            name = module_info["name"]
            if layer_index_from_module_name(name) not in allowed_layers:
                continue
            module = module_info["module"]
            if getattr(module, "patched_forward", False):
                continue
            logger.debug(f"Applying LoRA forward to {name}")
            module.forward_orig = module.forward
            module.patched_forward = True
            module.generated_lora_forward = partial(
                lora_forward_fn,
                self=module,
                lora_dropout_p=self.peft_config.lora_dropout,
                scaling=self.peft_config.lora_alpha,
            )
            module.forward = module.generated_lora_forward

    def _init_model(self):
        # disable adapter of the base model
        # this only works with LoRA(?)
        # we disable to avoid peft lora computation
        self.base_model.disable_adapter_layers()

        self.hypernet = (
            HyperLoRA(self.hypernet_config).to(self.device).to(torch.float32)
        )

        self.patch_lora_forward()

        ctx_model_name = self.ctx_encoder_args.ctx_encoder_model_name_or_path
        if ctx_model_name is None:
            ctx_model_name = self.base_model.config.name_or_path
        # use an explicit copy of the base model
        # for using with "modules_to_save"
        base_model_attn_impl = self.base_model.config._attn_implementation
        logger.debug(f"ctx_model_name: {ctx_model_name}")
        logger.debug(f"base_model.config._attn_implementation: {base_model_attn_impl}")
        context_attention_implementation = (
            "ctx_flex_attention"
            if int(
                getattr(
                    self.ctx_encoder_args,
                    "ctx_encoder_flex_query_chunk_size",
                    0,
                )
            )
            else "flex_attention"
        )
        encoder_model_kwargs = {
            "attn_implementation": context_attention_implementation
        }
        if getattr(self.ctx_encoder_args, "ctx_encoder_revision", None):
            encoder_model_kwargs["revision"] = (
                self.ctx_encoder_args.ctx_encoder_revision
            )
        encoder_model = get_model(
            ctx_model_name,
            train=self.base_model.training,
            requires_grad=False,
            use_flash_attn=base_model_attn_impl
            in {"flash_attention_2", "flex_attention"}
            or "gemma-4" in ctx_model_name.lower(),
            use_q_lora=self.ctx_encoder_args.quantize_ctx_encoder,
            model_kwargs=encoder_model_kwargs,
        )
        self.ctx_encoder = CTX_ENCODER_CLS[self.ctx_encoder_args.ctx_encoder_type](
            encoder_model, self.ctx_encoder_args
        )
        if (
            self.ctx_encoder_args.ctx_encoder_type
            == CTX_ENCODER_TYPE.PER_LAYER_ACTIVATIONS
            and self.hypernet.aggregator.layer_to_layer
        ):
            self.ctx_encoder.select_layer_inputs(self.hypernet.layer_indices)
        self.hypernet.enable_iterative_mode(
            bool(
                getattr(
                    self.ctx_encoder_args,
                    "sequential_ctx_layer_aggregation",
                    False,
                )
            )
        )

    # delegate to base_model
    @property
    def config(self):
        return self.base_model.config

    @property
    def generation_config(self):
        return self.base_model.generation_config

    @property
    def vocab_size(self):
        if hasattr(self.base_model, "vocab_size"):
            return self.base_model.vocab_size
        config = self.base_model.config
        vocab_size = getattr(config, "vocab_size", None)
        if vocab_size is not None:
            return vocab_size
        return getattr(config.text_config, "vocab_size")

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    @torch.no_grad()
    def _bias_hyper_init(self):
        if self.hypernet.extra_modules:
            self.hypernet.extra_head.weight.data[:] = 0
            self.hypernet.extra_head.bias.data[:] = 0
        if self.hypernet.target_modules:
            peft_weights = get_init_peft_weights(
                self.base_model, self.hypernet.lora_config
            )
            logger.debug(f"peft_weights: {peft_weights}")
            r = self.hypernet_config.lora_config.r
            nn.init.normal_(
                self.hypernet.head.weight,
                mean=0,
                std=0.5
                / sqrt(self.hypernet.config.latent_size + self.hypernet.d_lora * r),
                # the head outputs per rank lora --> divide by r to scale down grad
            )

    def state_dict(self, *args, **kwargs):
        # we assume ctx_encoder and base model is frozen here
        if len([p for p in self.ctx_encoder.parameters() if p.requires_grad]):
            raise ValueError("ctx_encoder contains trainable parameters")
        if len([p for p in self.base_model.parameters() if p.requires_grad]):
            raise ValueError("base model contains trainable parameters")

        state_dict = self.hypernet.state_dict(*args, **kwargs)
        state_dict["base_model_name_or_path"] = self.base_model.name_or_path
        state_dict["hypernet_config"] = self.hypernet_config
        state_dict["ctx_encoder_args"] = self.ctx_encoder_args
        return state_dict

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self.base_model_name_or_path = state_dict.pop("base_model_name_or_path")
        current_perceiver_attn = getattr(
            self.hypernet_config.aggregator_config,
            "perceiver_attn_implementation",
            "eager",
        )
        current_perceiver_checkpointing = getattr(
            self.hypernet_config.aggregator_config,
            "perceiver_activation_checkpointing",
            False,
        )
        current_perceiver_projection_chunk_size = getattr(
            self.hypernet_config.aggregator_config,
            "perceiver_modality_projection_chunk_size",
            0,
        )
        self.hypernet_config = state_dict.pop("hypernet_config")
        # Like context-layer CPU staging, the Perceiver attention kernel is an
        # execution policy with no trainable parameters.  Preserve the current
        # launch choice when resuming checkpoints written by the eager path.
        self.hypernet_config.aggregator_config.perceiver_attn_implementation = (
            current_perceiver_attn
        )
        self.hypernet_config.aggregator_config.perceiver_activation_checkpointing = (
            current_perceiver_checkpointing
        )
        self.hypernet_config.aggregator_config.perceiver_modality_projection_chunk_size = (
            current_perceiver_projection_chunk_size
        )
        checkpoint_ctx_encoder_args = state_dict.pop("ctx_encoder_args")
        # These flags change execution/memory policy, not model weights or the
        # mathematical objective. Preserve the current launch configuration so
        # an older checkpoint can resume with a safer implementation.
        for name in (
            "ctx_chunk_microbatch_size",
            "sequential_ctx_layer_aggregation",
            "checkpoint_ctx_to_lora_chunks",
            "offload_ctx_layer_inputs_to_cpu",
            "ctx_encoder_mlp_chunk_size",
            "ctx_encoder_flex_query_chunk_size",
            "offload_optimizer_state_during_context",
        ):
            if hasattr(self.ctx_encoder_args, name):
                setattr(
                    checkpoint_ctx_encoder_args,
                    name,
                    getattr(self.ctx_encoder_args, name),
                )
        self.ctx_encoder_args = checkpoint_ctx_encoder_args
        if self.base_model_name_or_path != self.base_model.name_or_path:
            raise ValueError(
                f"Base model name or path mismatch. "
                f"The base model given is: {self.base_model.name_or_path}, "
                f"but the loaded name is: {self.base_model_name_or_path}"
            )
        self._init_model()

        def remove_compile_prefix(sd: dict[str, Tensor]) -> dict[str, Tensor]:
            COMPILED_PREFIX = "_orig_mod."
            for k in list(sd.keys()):
                if k.startswith(COMPILED_PREFIX):
                    sd[k[len(COMPILED_PREFIX) :]] = sd.pop(k)
            return sd

        load_result = self.hypernet.load_state_dict(
            remove_compile_prefix(state_dict),
            strict=True,  # , *args, **kwargs
        )
        logger.info(f"load result: {load_result}")
        return load_result

    def _generate_weights_one_context_batch(
        self,
        ctx_ids: Integer[Tensor, "bs ctx_len"],
        ctx_attn_mask: Integer[Tensor, "bs ctx_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "bs ctx_len"] | None = None,
        return_latents: bool = False,
        **kwargs: Any,
    ):
        with torch.no_grad():
            ctx_encoder_kwargs = dict(
                input_ids=ctx_ids,
                attention_mask=ctx_attn_mask,
                position_ids=ctx_position_ids,
            )
            if isinstance(self.ctx_encoder.base_model, ModernBertModel):
                position_ids = ctx_position_ids.flatten()
                indices = torch.arange(
                    position_ids.size(0), device=position_ids.device, dtype=torch.int32
                )
                # [bsz + 1]
                cu_seqlens = torch.cat(
                    (
                        indices[position_ids == 0],
                        torch.tensor(
                            position_ids.size(),
                            device=position_ids.device,
                            dtype=torch.int32,
                        ),
                    )
                )
                ctx_encoder_kwargs = dict(
                    input_ids=ctx_ids.squeeze(0),
                    cu_seqlens=cu_seqlens,
                    max_seqlen=position_ids.max() + 1,
                    attention_mask=-1,
                    seq_len=-1,
                    batch_size=-1,
                )

            ctx_features = self.ctx_encoder(**ctx_encoder_kwargs, **kwargs)

        if isinstance(self.ctx_encoder.base_model, ModernBertModel):
            ctx_features = ctx_features.unsqueeze(0)
        if (
            self.hypernet.aggregator.layer_to_layer
            and isinstance(ctx_features, torch.Tensor)
            and ctx_features.dim() == 4
        ):
            layer_indices = torch.as_tensor(
                self.hypernet.layer_indices,
                dtype=torch.long,
                device=ctx_features.device,
            )
            if ctx_features.shape[1] != layer_indices.numel():
                ctx_features = ctx_features.index_select(1, layer_indices)
        elif isinstance(ctx_features, (tuple, list)):
            if len(ctx_features) != len(self.hypernet.layer_indices):
                raise ValueError(
                    "Offloaded context-layer input count does not match "
                    "the selected hypernetwork layers"
                )
        return self.hypernet.generate_weights(
            ctx_features,
            ctx_attn_mask,
            ctx_position_ids,
            return_latents=return_latents,
        )

    def generate_weights(
        self,
        ctx_ids: Integer[Tensor, "bs ctx_len"],
        ctx_attn_mask: Integer[Tensor, "bs ctx_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "bs ctx_len"] | None = None,
        return_latents: bool = False,
        **kwargs: Any,
    ):
        microbatch_size = getattr(
            self.ctx_encoder_args, "ctx_chunk_microbatch_size", 0
        )
        if microbatch_size not in {0, 1}:
            raise ValueError(
                "ctx_chunk_microbatch_size currently supports only 0 (packed) "
                "or 1 (one complete canonical chunk at a time)"
            )
        starts = None
        if (
            microbatch_size == 1
            and ctx_position_ids is not None
            and ctx_ids.ndim == 2
            and ctx_ids.shape[0] == 1
        ):
            starts = torch.where(ctx_position_ids[0] == 0)[0]

        if starts is not None and starts.numel() > 1:
            boundaries = [int(index) for index in starts.detach().cpu().tolist()]
            boundaries.append(ctx_ids.shape[1])
            per_chunk = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk_mask = (
                    ctx_attn_mask[:, start:end]
                    if ctx_attn_mask is not None
                    else None
                )
                generate_one = self._generate_weights_one_context_batch
                if (
                    self.training
                    and getattr(
                        self.ctx_encoder_args,
                        "checkpoint_ctx_to_lora_chunks",
                        False,
                    )
                ):
                    generate_one = self._checkpointed_chunk_weights
                per_chunk.append(
                    generate_one(
                        ctx_ids[:, start:end],
                        chunk_mask,
                        ctx_position_ids[:, start:end],
                        return_latents=return_latents,
                        **kwargs,
                    )
                )
            lora_dict = {
                module: {
                    key: torch.cat(
                        [result[0][module][key] for result in per_chunk], dim=0
                    )
                    for key in ("A", "B")
                }
                for module in per_chunk[0][0]
            }
            generated = (lora_dict, None)
            if return_latents:
                generated = (
                    *generated,
                    torch.cat([result[2] for result in per_chunk], dim=0),
                )
        else:
            generate_one = self._generate_weights_one_context_batch
            if (
                starts is not None
                and self.training
                and getattr(
                    self.ctx_encoder_args,
                    "checkpoint_ctx_to_lora_chunks",
                    False,
                )
            ):
                generate_one = self._checkpointed_chunk_weights
            generated = generate_one(
                ctx_ids,
                ctx_attn_mask,
                ctx_position_ids,
                return_latents=return_latents,
                **kwargs,
            )

        if self.user_defined_scaling == 1:
            return generated
        lora_dict = generated[0]
        for module in lora_dict:
            lora_dict[module]["A"] = lora_dict[module]["A"] * self.user_defined_scaling
            lora_dict[module]["B"] = lora_dict[module]["B"] * self.user_defined_scaling
        return (lora_dict, *generated[1:])

    def _checkpointed_chunk_weights(
        self,
        ctx_ids: Tensor,
        ctx_attn_mask: Tensor | None,
        ctx_position_ids: Tensor,
        return_latents: bool,
        **kwargs: Any,
    ):
        """Checkpoint one complete chunk across frozen Gemma and HyperLoRA."""

        modules = tuple(self.hypernet.target_modules)

        def run(ids, attention_mask, position_ids):
            result = self._generate_weights_one_context_batch(
                ids,
                attention_mask,
                position_ids,
                return_latents=return_latents,
                **kwargs,
            )
            flat = []
            for module in modules:
                flat.extend((result[0][module]["A"], result[0][module]["B"]))
            if return_latents:
                flat.append(result[2])
            return tuple(flat)

        flat = checkpoint(
            run,
            ctx_ids,
            ctx_attn_mask,
            ctx_position_ids,
            use_reentrant=False,
        )
        loras = {}
        offset = 0
        for module in modules:
            loras[module] = {"A": flat[offset], "B": flat[offset + 1]}
            offset += 2
        result = (loras, None)
        if return_latents:
            result = (*result, flat[offset])
        return result

    def enable_iterative_mode(self, x: bool):
        self.hypernet.enable_iterative_mode(x)

    def forward(
        self,
        ctx_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_attn_mask: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
        n_queries: Integer[Tensor, "n_ctx"] | None = None,
        return_generated_lora: bool | None = False,
        generate_lora_only: bool = False,
        generated_loras_override: dict[str, dict[str, Tensor]] | None = None,
        *model_inputs_args: Any,
        **model_inputs_kwargs: dict[str, Any],
    ) -> tuple | ModelOutput:
        """Forward pass of the modulated model."""
        generated_loras = generated_loras_override
        generated_layernorms = None
        chunk_latents = None
        packed_answer_boundaries = None
        if generated_loras_override is not None and ctx_ids is not None:
            raise ValueError("A generated-LoRA override cannot also encode context")
        if generated_loras_override is not None:
            pass
        elif ctx_ids is None and not self.use_base_input_as_ctx:
            logger.warning(
                (
                    "*" * 100,
                    "\n\nNo ctx_features provided, using the base model for forward pass\n\n",
                    "*" * 100,
                )
            )

        else:
            if self.use_base_input_as_ctx:
                ctx_ids = (
                    model_inputs_kwargs["input_ids"]
                    if "input_ids" in model_inputs_kwargs
                    else model_inputs_args[0]
                )
                ctx_attn_mask = (
                    model_inputs_kwargs["attention_mask"]
                    if "attention_mask" in model_inputs_kwargs
                    else None
                )
                ctx_position_ids = (
                    model_inputs_kwargs["position_ids"]
                    if "position_ids" in model_inputs_kwargs
                    else None
                )
            need_latents = self.hypernet.repo_merger.requires_chunk_latents
            generated = self.generate_weights(
                ctx_ids,
                ctx_attn_mask,
                ctx_position_ids,
                return_latents=need_latents,
            )
            generated_loras, generated_layernorms = generated[:2]
            chunk_latents = generated[2] if need_latents else None

        if generated_loras is not None:
            if n_ctx_chunks is None:
                lora_device = generated_loras[next(iter(generated_loras))][
                    "A"
                ].device
                n_ctx_chunks = torch.ones(
                    generated_loras[next(iter(generated_loras))]["A"].shape[0],
                    dtype=torch.int32,
                    device=lora_device,
                )
            if generated_loras_override is None:
                generated_loras = self.hypernet.merge_repository_loras(
                    generated_loras,
                    n_ctx_chunks,
                    chunk_latents=chunk_latents,
                )
            if generate_lora_only:
                return generated_loras

            # input_ids in model_inputs_kwargs contains only
            # prompt + response (for hypernet training)
            position_ids = (
                model_inputs_kwargs["position_ids"]
                if "position_ids" in model_inputs_kwargs
                else None
            )

            if n_queries is None:
                if ctx_position_ids is None:
                    n_queries = torch.ones(
                        ctx_ids.shape[0], dtype=torch.int32, device=self.device
                    )
                else:
                    # quite redundant (we do cu_seqlens many places)
                    # TODO: compute cu_seqlens here and propagate that
                    n_queries = torch.ones(
                        (ctx_position_ids == 0).sum(),
                        dtype=torch.int32,
                        device=self.device,
                    )

            if (
                position_ids is not None
                and not model_inputs_args
                and position_ids.ndim == 2
                and position_ids.shape[0] == 1
            ):
                starts = torch.where(position_ids[0] == 0)[0]
                if starts.numel() > 1:
                    packed_answer_boundaries = [
                        int(value) for value in starts.detach().cpu().tolist()
                    ] + [position_ids.shape[1]]
            if packed_answer_boundaries is None:
                apply_lora_to_layers(
                    self.base_model,
                    self.hypernet.layer_indices,
                    generated_loras,
                    n_queries,
                    position_ids,
                )
        if _attention_implementation(self.base_model) == "flex_attention":
            # The same exact FlexAttention path is used for the packed answer
            # model.  Pin the regular Triton kernel here as well: PyTorch
            # 2.11's short-query flex-decoding heuristic has no valid Gemma 4
            # GQA choice on H100 for some packed QA shapes.
            model_inputs_kwargs.setdefault(
                "kernel_options", _gemma4_flex_kernel_options()
            )
        if packed_answer_boundaries is None:
            model_outputs = self.base_model(*model_inputs_args, **model_inputs_kwargs)
        else:
            # Execute each logical QA as an independent model sequence while
            # reusing the repository adapter generated above. This is the
            # reference-exact answer path: it prevents cross-QA attention and
            # avoids shape-dependent bf16 drift between a flattened packed
            # forward and literal separate examples. Context encoding and
            # adapter generation still happen once per physical context group.
            context_for_query = torch.arange(
                n_queries.numel(), device=n_queries.device
            ).repeat_interleave(n_queries.to(torch.long))
            if context_for_query.numel() != len(packed_answer_boundaries) - 1:
                raise ValueError("n_queries does not match packed QA boundaries")
            outputs = []
            total_length = position_ids.shape[1]
            for query_index, (start, end) in enumerate(
                zip(packed_answer_boundaries[:-1], packed_answer_boundaries[1:])
            ):
                context_index = int(context_for_query[query_index].item())
                query_loras = {
                    module: {
                        name: value[context_index : context_index + 1]
                        for name, value in factors.items()
                    }
                    for module, factors in generated_loras.items()
                }
                one_query = torch.ones(
                    1, dtype=n_queries.dtype, device=n_queries.device
                )
                query_kwargs = {}
                for name, value in model_inputs_kwargs.items():
                    if (
                        isinstance(value, torch.Tensor)
                        and value.ndim >= 2
                        and value.shape[0] == 1
                        and value.shape[1] == total_length
                    ):
                        query_kwargs[name] = value[:, start:end]
                    else:
                        query_kwargs[name] = value
                apply_lora_to_layers(
                    self.base_model,
                    self.hypernet.layer_indices,
                    query_loras,
                    one_query,
                    query_kwargs.get("position_ids"),
                )
                outputs.append(self.base_model(**query_kwargs))
            model_outputs = outputs[0]
            model_outputs.logits = torch.cat(
                [output.logits for output in outputs], dim=1
            )

        if return_generated_lora:
            return model_outputs, (generated_loras, generated_layernorms)
        else:
            return model_outputs

    def combine_lora(self, *args, **kwargs):
        # for timing
        return combine_lora(*args, **kwargs)

    def merge_repository_loras(self, *args, **kwargs):
        # for timing and evaluation instrumentation
        return self.hypernet.merge_repository_loras(*args, **kwargs)

    def apply_lora_to_layers(self, *args, **kwargs):
        # for timing
        return apply_lora_to_layers(*args, **kwargs)

    # for simple api usage
    def internalize(self, ctx_str: str):
        ctx_tokenizer = get_tokenizer(self.ctx_encoder.base_model.name_or_path)
        ctx_ids = tokenize_ctx_text(dict(context=[ctx_str]), ctx_tokenizer)["ctx_ids"]
        return self._internalize_from_ids(torch.tensor(ctx_ids, device=self.device))

    def _internalize_from_ids(
        self,
        ctx_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_attn_mask: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
    ):
        self.patch_lora_forward()
        if ctx_attn_mask is None and ctx_position_ids is None:
            assert ctx_ids.shape[0] == 1
            ctx_attn_mask = torch.ones_like(ctx_ids)
        need_latents = self.hypernet.repo_merger.requires_chunk_latents
        generated = self.generate_weights(
            ctx_ids,
            ctx_attn_mask,
            ctx_position_ids,
            return_latents=need_latents,
        )
        generated_loras, generated_layernorms = generated[:2]
        self.generated_loras = generated_loras
        self.generated_chunk_latents = generated[2] if need_latents else None

    def reset(self):
        self.generated_loras = None
        self.generated_chunk_latents = None
        layers = get_layers(self.base_model)
        for layer_idx in self.hypernet.layer_indices:
            for module_info in get_peft_modules(layers[layer_idx], self.peft_config):
                name = module_info["name"]
                module = module_info["module"]
                logger.debug(f"Resetting forward for {name}")
                module.forward = module.forward_orig
                module.patched_forward = False

    @torch.inference_mode()
    def generate(
        self,
        ctx_ids: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        ctx_attn_mask: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        ctx_position_ids: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
        n_queries: Integer[Tensor, "n_ctx"] | None = None,
        scalers: Float[Tensor, "n_ctx"] | None = None,
        bias_scaler: float | None = None,
        *model_inputs_args: Any,
        **model_inputs_kwargs: dict[str, Any],
    ):
        generated_loras = None
        generated_layernorms = None
        chunk_latents = None
        if (
            ctx_ids is None
            and not self.generated_loras
            and not self.use_base_input_as_ctx
        ):
            print(
                "*" * 100
                + "\n\nNo ctx_ids provided, using the base model for generation\n\n"
                + "*" * 100
            )
        elif ctx_ids is None and self.generated_loras:
            generated_loras = self.generated_loras
            chunk_latents = self.generated_chunk_latents
            if n_ctx_chunks is None:
                n_ctx_chunks = torch.tensor((1,), device=self.device)
            print(
                "*" * 100
                + "\n\nUsing internalized LoRAs for generation\n\n"
                + "*" * 100
            )
        else:
            if self.use_base_input_as_ctx:
                ctx_ids = (
                    model_inputs_kwargs["input_ids"]
                    if "input_ids" in model_inputs_kwargs
                    else model_inputs_args[0]
                )
                ctx_attn_mask = (
                    model_inputs_kwargs["attention_mask"]
                    if "attention_mask" in model_inputs_kwargs
                    else None
                )
                ctx_position_ids = (
                    model_inputs_kwargs["position_ids"]
                    if "position_ids" in model_inputs_kwargs
                    else None
                )
            need_latents = self.hypernet.repo_merger.requires_chunk_latents
            generated = self.generate_weights(
                ctx_ids,
                ctx_attn_mask,
                ctx_position_ids,
                return_latents=need_latents,
            )
            generated_loras, generated_layernorms = generated[:2]
            chunk_latents = generated[2] if need_latents else None

        if generated_loras is not None:
            if n_ctx_chunks is None:
                n_ctx_chunks = torch.ones(
                    generated_loras[next(iter(generated_loras))]["A"].shape[0],
                    dtype=torch.int32,
                    device=ctx_ids.device if ctx_ids is not None else self.device,
                )
            generated_loras = self.merge_repository_loras(
                generated_loras,
                n_ctx_chunks,
                chunk_latents=chunk_latents,
                scalers=scalers,
                bias_scaler=bias_scaler,
            )

            # apply lora hook to the base model
            # TODO: we dont this position_ids for generation?
            position_ids = (
                model_inputs_kwargs["position_ids"]
                if "position_ids" in model_inputs_kwargs
                else None
            )
            if n_queries is None:
                if ctx_position_ids is None:
                    n_queries = torch.ones(
                        model_inputs_kwargs["input_ids"].shape[0],
                        dtype=torch.int32,
                        device=self.device,
                    )
                else:
                    # quite redundant (we do cu_seqlens many places)
                    # TODO: compute cu_seqlens here and propagate that
                    n_queries = torch.ones(
                        (ctx_position_ids == 0).sum(),
                        dtype=torch.int32,
                        device=self.device,
                    )

            apply_lora_to_layers(
                self.base_model,
                self.hypernet.layer_indices,
                generated_loras,
                n_queries,
                position_ids,
            )

        model_outputs = self.base_model.generate(
            *model_inputs_args, **model_inputs_kwargs
        )
        return model_outputs


# needed for loading model from checkpoint
# see https://github.com/huggingface/transformers/pull/34632
torch.serialization.add_safe_globals(
    [
        AggregatorConfig,
        LoraConfig,
        HypernetConfig,
        RepositoryMergerConfig,
        RepoMergeMethod,
        PeftType,
        TaskType,
        LoraRuntimeConfig,
        set,  # for real?
    ]
)
