# taken from github.com/sakanaai/text-to-lora
import argparse
import logging
import math
import os
from copy import deepcopy
from math import sqrt
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from peft import PeftConfig, PeftModel, get_peft_config, load_peft_weights
from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


def add_full_stop(s):
    s = s.strip()
    # check if s ends with . or .*
    if s[-1].isalpha():
        s += "."
    return s


def get_layers(model):
    if hasattr(model, "model"):
        return get_layers(model.model)
    return model.layers


def get_pooling_fn(pooling_type: Literal["last_token", "cls"]):
    if pooling_type == "last_token":
        return last_token_pool
    elif pooling_type == "cls":
        return cls_pool
    else:
        raise ValueError(f"Invalid pooling type: {pooling_type}")


def cls_pool(
    outputs: dict[str, torch.Tensor], attention_mask: torch.Tensor
) -> torch.Tensor:
    right_padding = attention_mask[:, 0].sum() == attention_mask.shape[0]
    assert right_padding, f'tokenizer.padding_side should be "right"'
    return outputs["last_hidden_state"][:, 0].detach()


def last_token_pool(
    outputs: dict[str, torch.Tensor], attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden_states = (
        outputs["hidden_states"][-1].detach()
        if "hidden_states" in outputs
        else outputs["last_hidden_state"].detach()
    )
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def lora_state_dict_to_tensor_dict(lora_sd, target_modules, layer_indices, device):
    A, B = (
        {
            target_module: [None for _ in range(len(layer_indices))]
            for target_module in target_modules
        },
        {
            target_module: [None for _ in range(len(layer_indices))]
            for target_module in target_modules
        },
    )

    for k, v in lora_sd.items():
        for target_module in target_modules:
            if target_module in k:
                layer_idx = int(k.split("layers.")[-1].split(".")[0])
                if layer_idx in layer_indices:
                    if "lora_A" in k:
                        A[target_module][layer_idx] = v.to(device)
                    elif "lora_B" in k:
                        B[target_module][layer_idx] = v.to(device)

    for target_module in target_modules:
        A[target_module] = torch.stack(A[target_module], dim=0)
        B[target_module] = torch.stack(B[target_module], dim=0)

    return dict(A=A, B=B)


def get_emb_model_and_fns(emb_model_name, device):
    emb_model = AutoModel.from_pretrained(
        emb_model_name,
        device_map=device,
        torch_dtype=torch.float32 if "gte" in emb_model_name else torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
    if emb_tokenizer.pad_token_id is None:
        emb_tokenizer.pad_token_id = emb_tokenizer.eos_token_id
    task_desc_format_fn = add_full_stop
    pooling_fn = get_pooling_fn("cls")
    return emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn


def get_model_and_tokenizer(
    model_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    tokenizer_kwargs=None,
    device="cuda:0",
    dtype=torch.bfloat16,
):
    model = get_model(
        model_path,
        train,
        requires_grad,
        use_flash_attn,
        peft_config,
        model_kwargs,
        device,
        dtype,
    )
    tokenizer = get_tokenizer(model_path, tokenizer_kwargs, peft_config, train)
    return model, tokenizer


def get_tokenizer(model_path, tokenizer_kwargs=None, peft_config=None, train=False):
    # NOTE: lora models don't have tokenizer config in the folder

    padding_side = "left" if not train else "right"
    if peft_config:
        model_path = peft_config.base_model_name_or_path

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side=padding_side, **tokenizer_kwargs
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    template_path = f"chat_templates/{model_path}.jinja"
    assert os.path.exists(template_path), (
        f"Chat template not found for {model_path}.\n"
        "We assume a specfic form of chat template for consistency between models. "
        "Please use the templates provided."
    )
    print(f"Loading chat template from {template_path}")
    chat_template = open(template_path).read()
    chat_template = chat_template.replace("    ", "").replace("\n", "")
    tokenizer.chat_template = chat_template

    tokenizer.add_eos_token = False
    tokenizer.truncation_side = "left"
    return tokenizer


def get_model(
    model_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    device="cuda:0",
    dtype=torch.bfloat16,
):
    model_init_kwargs = dict(
        pretrained_model_name_or_path=model_path,
        device_map=device,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if model_kwargs is not None:
        model_init_kwargs.update(model_kwargs)
    if use_flash_attn:
        model_init_kwargs["attn_implementation"] = "flash_attention_2"
    if train:
        # for training disable cache
        model_init_kwargs["use_cache"] = False
    logger.debug(f"Model init kwargs: {model_init_kwargs}")
    model = AutoModelForCausalLM.from_pretrained(**model_init_kwargs)
    if peft_config is not None:
        model = PeftModel(model, peft_config)
    model.train(train)
    for param in model.parameters():
        param.requires_grad = requires_grad
    return model


def get_lora_module_names(model, target_modules, layer_indices):
    module_names = {
        target_module: [[] for _ in range(len(layer_indices))]
        for target_module in target_modules
    }
    for k in get_peft_model_state_dict(model):
        if ("lora" not in k) and ("vera_lambda" not in k):
            continue
        layer_idx = int(k.split("layers.")[-1].split(".")[0])
        if layer_idx in layer_indices:
            for target_module in target_modules:
                if target_module in k:
                    if "vera_lambda" in k:
                        # replace the name to match the lora naming convention
                        k = k.replace("vera_lambda_d", "lora_A.weight")
                        k = k.replace("vera_lambda_b", "lora_B.weight")
                    module_names[target_module][layer_idx].append(k)
                    break
    return module_names


logger = logging.getLogger()


# taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    x = (x * cos) + (rotate_half(x) * sin)
    return x


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
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


class TaskEncoder(nn.Module):
    def __init__(self, task_emb_size: int, encoded_task_emb_size: int):
        super().__init__()
        self.encoded_task_emb_size = encoded_task_emb_size
        self.mlp = nn.Sequential(
            nn.Linear(task_emb_size, encoded_task_emb_size),
            nn.LayerNorm(encoded_task_emb_size),
        )

    def get_one_hot_task_emb(
        self, num_tasks: int, task_idx: torch.Tensor
    ) -> torch.Tensor:
        return torch.eye(num_tasks, device=task_idx.device)[task_idx]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"encoded_task_emb": self.mlp(x)}


class DiscreteOneHotTaskEncoder(nn.Module):
    def __init__(self, task_emb_size: int, n_classes: int, n_embs: int):
        super().__init__()
        self.n_classes = n_classes
        self.n_embs = n_embs
        self.encoded_task_emb_size = n_classes * n_embs
        self.mlp = nn.Sequential(
            nn.Linear(task_emb_size, self.encoded_task_emb_size * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(self.encoded_task_emb_size * 4, self.encoded_task_emb_size),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.mlp(x)
        logits = logits.reshape(*logits.shape[:-1], self.n_embs, self.n_classes)
        if self.training:
            dist = torch.distributions.one_hot_categorical.OneHotCategorical(
                logits=logits
            )
            one_hot = dist.sample() + dist.probs - dist.probs.detach()
        else:
            one_hot = torch.eye(self.n_classes, device=x.device)[
                torch.argmax(logits, dim=-1)
            ]
        return {"encoded_task_emb": torch.flatten(one_hot, start_dim=-2, end_dim=-1)}


# similar to https://github.com/karpathy/deep-vector-quantization/blob/c3c026a1ccea369bc892ad6dde5e6d6cd5a508a4/dvq/model/quantize.py#L77
class SoftmaxTaskEncoder(nn.Module):
    def __init__(self, task_emb_size: int, code_dim: int, n_embs: int):
        super().__init__()
        self.n_embs = n_embs
        self.code_dim = code_dim
        self.mlp = nn.Sequential(
            nn.Linear(task_emb_size, code_dim * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(code_dim * 4, n_embs),
        )
        self.embed = nn.Parameter(torch.randn(n_embs, code_dim), requires_grad=True)
        self.ln = nn.LayerNorm(code_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.mlp(x).reshape(-1, self.n_embs)  # [bs, n_embs]
        probs = F.softmax(logits, dim=-1)  # [bs, n_embs]
        emb = probs.unsqueeze(-1) * self.embed.unsqueeze(0)  # [bs, n_embs, code_dim]
        emb = emb.sum(1)  # [bs, code_dim]
        emb = self.ln(emb)
        return {"encoded_task_emb": emb}


# based on https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
class VQTaskEncoder(nn.Module):
    def __init__(
        self, task_emb_size: int, code_dim: int, n_embs: int, decay: float = 0.99
    ):
        super().__init__()
        self.n_embs = n_embs
        self.code_dim = code_dim
        self.decay = decay
        self.mlp = nn.Sequential(
            nn.Linear(task_emb_size, code_dim * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(code_dim * 4, code_dim),
        )
        self.ln = nn.LayerNorm(code_dim)
        self.register_buffer("embed", torch.randn(code_dim, n_embs))
        self.register_buffer("cluster_size", torch.zeros(n_embs))
        self.register_buffer("embed_avg", self.embed.clone())

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_e = self.mlp(x).reshape(-1, self.code_dim)  # [bs, code_dim]

        dist = (
            z_e.pow(2).sum(1, keepdim=True)
            - 2 * z_e @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        indices = torch.argmin(dist, dim=-1)  # [bs]
        one_hot = F.one_hot(indices, self.n_embs).to(z_e.dtype)  # [bs, n_embs]
        indices = indices.view(*x.shape[:-1])  # [bs]
        quantize = self.embed_code(indices)
        if self.training:
            # exponential moving average update for the codebook
            embed_onehot_sum = one_hot.sum(0)  # [n_embs]
            embed_sum = z_e.transpose(0, 1) @ one_hot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + 1e-5) / (n + self.n_embs * 1e-5) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        commitment_cost = 0.01
        diff = commitment_cost * (quantize.detach() - z_e).pow(2).mean()
        quantize = self.ln(z_e + (quantize - z_e).detach())
        return {"encoded_task_emb": quantize, "loss": diff}


class MLPResidualBlock(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, pre_layer_norm, post_dropout
    ):
        super().__init__()
        layers = []
        if pre_layer_norm:
            layers.append(nn.LayerNorm(input_size))
        layers += [
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size, output_size),
            nn.SiLU(),
        ]
        if post_dropout:
            layers.append(nn.Dropout(0.05))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(x)


def zero_lora_param_dict(target_modules, n_layers, rank, in_features, out_features):
    return nn.ParameterDict(
        {
            "A": nn.ParameterDict(
                {
                    m: nn.Parameter(
                        torch.zeros(n_layers, rank, in_features[m]), requires_grad=False
                    )
                    for m in target_modules
                }
            ),
            "B": nn.ParameterDict(
                {
                    m: nn.Parameter(
                        torch.zeros(n_layers, out_features[m], rank),
                        requires_grad=False,
                    )
                    for m in target_modules
                }
            ),
        }
    )


def lora_tensor_dict_to_param_dict(lora_tensor_dict, requires_grad):
    return nn.ParameterDict(
        {
            "A": nn.ParameterDict(
                {
                    k: nn.Parameter(v, requires_grad)
                    for k, v in lora_tensor_dict["A"].items()
                }
            ),
            "B": nn.ParameterDict(
                {
                    k: nn.Parameter(v, requires_grad)
                    for k, v in lora_tensor_dict["B"].items()
                }
            ),
        }
    )


class HyperModulator(nn.Module):
    def __init__(
        self,
        model: PeftModel,
        output_space: str,
        module_names: list[str],
        training_task: Literal["sft", "recon"] = "sft",
        pred_z_score: bool | None = None,
        mean_recon_target: dict[str, torch.Tensor] | None = None,
        std_recon_target: dict[str, torch.Tensor] | None = None,
        match_lora_init: bool = False,
        task_emb_size: int | None = None,
        shared_AB_head: bool = False,
        autoreg_gen: bool = False,
        learnable_pos_emb: bool = False,
        encoder_type: Literal["linear", "discrete", "vq", "softmax"] = "linear",
        AB_offset: dict[str, dict[str, torch.Tensor]] | None = None,
        learnable_AB_offset: bool = False,
        zero_init_head: bool = False,
        latent_size: int = 128,
        head_in_size: int = 512,
        head_use_bias: bool = False,
        factorized: bool = False,
        delta_w_scaling: float = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        assert output_space == "lora", f"Invalid output space: {output_space}"
        assert (not shared_AB_head) or output_space == "lora", (
            "shared_AB_head is only supported for lora output space"
        )
        assert (not autoreg_gen) or output_space == "lora", (
            "autoreg_gen is only supported for shared_AB_head and lora output space"
        )
        assert (not learnable_pos_emb) or autoreg_gen, (
            "learnable_pos_emb is only supported when autoreg_gen is True"
        )
        assert not (training_task == "recon" and AB_offset), (
            "AB_offset is not supported for recon training"
        )

        super().__init__()
        self.model_config = model_config = model.config
        self.peft_config = peft_config = model.peft_config["default"]
        self.training_task = training_task
        self.pred_z_score = pred_z_score
        self.module_names = module_names
        self.shared_AB_head = shared_AB_head
        self.autoreg_gen = autoreg_gen
        self.learnable_pos_emb = learnable_pos_emb
        self.scaling = (
            peft_config.lora_alpha / peft_config.r if output_space == "lora" else 1
        )
        if getattr(peft_config, "use_rslora", False):
            self.scaling *= math.sqrt(peft_config.r)
        self.output_space = output_space
        self.max_num_layers = model_config.num_hidden_layers

        self.target_modules = peft_config.target_modules
        self.module_to_int = {m: i for i, m in enumerate(self.target_modules)}
        self.task_to_int = dict()

        self.factorized = factorized
        self.delta_w_scaling = delta_w_scaling
        self.device = device = model.device
        self.dtype = dtype

        self.in_features, self.out_features = get_in_out_features(model, peft_config)
        n_layers = self.max_num_layers
        self.AB_offset = zero_lora_param_dict(
            self.target_modules,
            n_layers,
            peft_config.r,
            self.in_features,
            self.out_features,
        )
        self.mean_recon_target = zero_lora_param_dict(
            self.target_modules,
            n_layers,
            peft_config.r,
            self.in_features,
            self.out_features,
        )
        self.std_recon_target = zero_lora_param_dict(
            self.target_modules,
            n_layers,
            peft_config.r,
            self.in_features,
            self.out_features,
        )
        if AB_offset is not None:
            self.AB_offset = lora_tensor_dict_to_param_dict(
                AB_offset, requires_grad=learnable_AB_offset
            )
        if mean_recon_target is not None:
            self.mean_recon_target = lora_tensor_dict_to_param_dict(
                mean_recon_target, requires_grad=False
            )
        if std_recon_target is not None:
            self.std_recon_target = lora_tensor_dict_to_param_dict(
                std_recon_target, requires_grad=False
            )

        # the input and output features for each target module (i.e. layer type)

        encoded_task_emb_size = 0

        if task_emb_size is not None:
            encoded_task_emb_size = latent_size // 2
            self.task_encoder = {
                "linear": TaskEncoder(task_emb_size, encoded_task_emb_size),
                "discrete": DiscreteOneHotTaskEncoder(
                    task_emb_size, n_classes=64, n_embs=1
                ),
                "vq": VQTaskEncoder(
                    task_emb_size, code_dim=encoded_task_emb_size, n_embs=32
                ),
                "softmax": SoftmaxTaskEncoder(
                    task_emb_size, code_dim=encoded_task_emb_size, n_embs=32
                ),
            }[encoder_type].to(device)
        if encoder_type == "discrete":
            depth_emb_size = self.max_num_layers
            type_emb_size = len(self.target_modules)
            self.layer_depth_encoder = lambda x: F.one_hot(
                x, num_classes=self.max_num_layers
            ).to(dtype)
            self.layer_type_encoder = lambda x: F.one_hot(
                x, num_classes=len(self.target_modules)
            ).to(dtype)
        else:
            depth_emb_size = latent_size // 4
            type_emb_size = latent_size // 4
            self.layer_depth_encoder = nn.Sequential(
                nn.Embedding(self.max_num_layers, depth_emb_size),
                nn.LayerNorm(depth_emb_size),
            )
            self.layer_type_encoder = nn.Sequential(
                nn.Embedding(len(self.target_modules), type_emb_size),
                nn.LayerNorm(type_emb_size),
            )

        mlp_inp_size = depth_emb_size + type_emb_size + encoded_task_emb_size

        self.mixer = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size, mlp_inp_size * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size * 4, mlp_inp_size),
            nn.SiLU(),
            nn.Dropout(0.05),
        )
        self.mlp1 = MLPResidualBlock(
            mlp_inp_size,
            mlp_inp_size * 4,
            mlp_inp_size,
            pre_layer_norm=True,
            post_dropout=True,
        )
        self.mlp2 = MLPResidualBlock(
            mlp_inp_size,
            mlp_inp_size * 4,
            mlp_inp_size,
            pre_layer_norm=True,
            post_dropout=True,
        )

        self.mlp3 = nn.Sequential(
            nn.LayerNorm(mlp_inp_size),
            nn.Linear(mlp_inp_size, mlp_inp_size * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size * 4, head_in_size),
            nn.SiLU(),
        )

        if autoreg_gen:
            if learnable_pos_emb:
                self.pos_emb = nn.Sequential(
                    nn.Embedding(peft_config.r, mlp_inp_size),
                    nn.LayerNorm(mlp_inp_size),
                )
            else:
                self.rotary_emb = LlamaRotaryEmbedding(
                    dim=mlp_inp_size,
                    max_position_embeddings=64,  # model_config.hidden_size,
                    device=device,
                )

        # multiple heads, each for a different target module
        if shared_AB_head:
            self.out_emb = nn.ParameterDict(
                (
                    m,
                    nn.ParameterDict(
                        dict(
                            # same init as the embedding layer
                            A=nn.Parameter(
                                torch.normal(0, 1, size=(mlp_inp_size,)),
                                requires_grad=True,
                            ),
                            B=nn.Parameter(
                                torch.normal(0, 1, size=(mlp_inp_size,)),
                                requires_grad=True,
                            ),
                        )
                    ),
                )
                for m in self.target_modules
            )
            self.out_emb_norm = nn.LayerNorm(mlp_inp_size)

        heads = []
        for module in self.target_modules:
            in_features = self.in_features[module]
            out_features = self.out_features[module]
            if not shared_AB_head:
                output_size = (
                    peft_config.r + out_features
                    if output_space == "vera"
                    else (peft_config.r * (in_features + out_features))
                )
                if autoreg_gen:
                    output_size = in_features + out_features

            else:
                output_size = peft_config.r * max(in_features, out_features)
                if autoreg_gen:
                    output_size = max(in_features, out_features)

            logger.debug(f"head_in_size: {head_in_size}, output_size: {output_size}")
            logger.debug(f"head_use_bias: {head_use_bias}")
            layer = nn.Linear(
                head_in_size, output_size, bias=head_use_bias, device=device
            )
            if zero_init_head:
                logger.debug(f"zeroing out head weights for {module}")
                nn.init.zeros_(layer.weight)

            heads.append((module, layer))
        self.heads = nn.ModuleDict(heads)
        peft_weights = get_init_peft_weights(model, peft_config)
        logger.debug(f"peft_weights:\n{peft_weights}")

        self.split_shapes = dict()
        # we use the trainable peft weights of the base peft model
        # to initialize the bias of the heads
        for module_name, head in self.heads.items():
            if output_space == "lora":
                A = peft_weights[module_name]["lora_A"].weight.clone().flatten()
                B = peft_weights[module_name]["lora_B"].weight.clone().flatten()
                init_bias = [A, B]

            elif output_space == "vera":
                # d init with a constant value (0.1) and b with zeros
                # for vera, we're using the outputs of the hypernet as masking + scaling values
                d = peft_weights[module_name]["vera_lambda_d"].clone().flatten()  # 0.1
                b = peft_weights[module_name]["vera_lambda_b"].clone().flatten()  # 0
                init_bias = [d, b]

                # we just need one copy of A and B
                vera_A_B_learnable = True
                self.vera_A = nn.Parameter(
                    peft_weights[module_name]["vera_A"],
                    requires_grad=vera_A_B_learnable,
                )
                self.vera_B = nn.Parameter(
                    peft_weights[module_name]["vera_B"],
                    requires_grad=vera_A_B_learnable,
                )

            with torch.no_grad():
                if match_lora_init and head.bias is not None:
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
                    if (not shared_AB_head) and (not autoreg_gen):
                        # based on https://proceedings.mlr.press/v205/beck23a/beck23a.pdf (bias-HyperInit)
                        # this encourages the hnet to initially use some shared values of the output
                        # also an easy way to match initialization of the corresponding method!
                        head.bias.copy_(torch.cat(init_bias))

                    # match the gradient scale to the full architecture
                    elif shared_AB_head:
                        if not autoreg_gen:
                            head.bias.copy_(init_bias[0] / sqrt(2))
                        else:
                            size = head.bias.shape[0]
                            head.bias.copy_(
                                init_bias[0][:size] / sqrt(2) / sqrt(peft_config.r)
                            )
                    else:
                        size = head.bias.shape[0]
                        head.bias.copy_(init_bias[0][:size] / sqrt(peft_config.r))

                # NOTE: how we slice the weights have to be consistent with the forward pass
                self.split_shapes[module_name] = [len(x) for x in init_bias]
                logger.debug(f"split_shapes: {self.split_shapes}")

        self.heads.to(device).to(dtype)
        del peft_weights

    def _embed_layer_depth(
        self, depth_indices: list[int] | int | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(depth_indices, int):
            depth_indices = torch.tensor(
                [depth_indices], dtype=torch.long, device=self.device
            )
        elif isinstance(depth_indices, list):
            depth_indices = torch.tensor(
                depth_indices, dtype=torch.long, device=self.device
            )

        return self.layer_depth_encoder(depth_indices)  # [bs (or 1), max_num_layers]

    def _embed_layer_type(self, layer_type: str) -> torch.Tensor:
        module_idx = self.module_to_int[layer_type]
        module_idx = torch.tensor([module_idx], dtype=torch.long, device=self.device)
        # we only forward one layer type at at time
        # so the shape is always [1, num_target_modules]
        return self.layer_type_encoder(module_idx)

    def get_one_hot_task_emb(
        self, num_tasks: int, task_idx: torch.Tensor
    ) -> torch.Tensor:
        return self.task_encoder.get_one_hot_task_emb(num_tasks, task_idx)

    def _hypernet_forward(self, layer_indices, layer_type, encoded_task_emb):
        # forward one layer type at a time

        bs = len(layer_indices)
        depth_emb = self._embed_layer_depth(layer_indices)  # [bs, depth_emb_size]
        layer_type_emb = self._embed_layer_type(layer_type)  # [1, layer_emb_size]
        layer_type_emb = layer_type_emb.expand(bs, -1)  # [bs, layer_emb_size]
        if encoded_task_emb is None:
            encoded_task_emb = torch.empty(0, device=self.device)

        cat_emb = torch.cat([encoded_task_emb, depth_emb, layer_type_emb], dim=-1)
        mlp_inp = self.mixer(cat_emb)

        mlp_out = self.mlp1(mlp_inp)
        head = self.heads[layer_type]
        if not self.shared_AB_head:
            # head outputs both A and B
            if not self.autoreg_gen:
                # head outputs all the ranks at once
                head_out = head(self.mlp3(self.mlp2(mlp_out)))
                splitted_out = torch.split(
                    head_out, self.split_shapes[layer_type], dim=-1
                )
            else:
                # head outputs one rank at a time
                head_in = mlp_out.unsqueeze(1).expand(-1, self.peft_config.r, -1)
                if self.learnable_pos_emb:
                    head_in = self.mlp2(
                        head_in
                        + self.pos_emb(
                            torch.arange(self.peft_config.r, device=self.device)
                        ).unsqueeze(0)
                    )
                else:
                    cos, sin = self.rotary_emb(
                        head_in,
                        torch.arange(self.peft_config.r, device=self.device).unsqueeze(
                            0
                        ),
                    )  # [1, r, 2048]
                    head_in = self.mlp2(apply_rotary_pos_emb(head_in, cos, sin))
                head_out = head(self.mlp3(head_in))

                # [bs, self.peft_config.r, self.in_features[layer_type]]
                # [bs, self.peft_config.r, self.out_features[layer_type]]
                A, B = torch.split(
                    head_out,
                    (self.in_features[layer_type], self.out_features[layer_type]),
                    dim=-1,
                )
                A = A.reshape(bs, -1)
                B = B.reshape(bs, -1)
                splitted_out = [A, B]

        else:
            # head outputs either A or B
            splitted_out = []
            for out_emb, num_features in zip(
                self.out_emb[layer_type].values(),
                [self.in_features[layer_type], self.out_features[layer_type]],
            ):
                head_in = self.mlp2(mlp_out + self.out_emb_norm(out_emb))
                if not self.autoreg_gen:
                    # head outputs all the ranks at once
                    head_out = head(self.mlp3(head_in))
                    head_out = head_out.view(bs, self.peft_config.r, -1)
                    head_out = head_out[..., :num_features]
                    head_out = head_out.reshape(bs, self.peft_config.r * num_features)
                else:
                    # head outputs one rank at a time
                    head_in = head_in.unsqueeze(1).expand(-1, self.peft_config.r, -1)
                    if self.learnable_pos_emb:
                        pos_emb = self.mlp2(
                            self.pos_emb(
                                torch.arange(self.peft_config.r, device=self.device)
                            ).unsqueeze(0)
                        )
                        head_in = head_in + pos_emb
                    else:
                        cos, sin = self.rotary_emb(
                            head_in,
                            torch.arange(
                                self.peft_config.r, device=self.device
                            ).unsqueeze(0),
                        )  # [1, r, 2048]
                        head_in = self.mlp2(apply_rotary_pos_emb(head_in, cos, sin))

                    head_out = head(self.mlp3(head_in))
                    head_out = head_out[..., :num_features]
                    head_out = head_out.reshape(bs, self.peft_config.r * num_features)

                splitted_out.append(head_out)

        return splitted_out

    def get_delta_weights(
        self,
        layer_indices: torch.Tensor,
        layer_type: str,
        encoded_task_emb: torch.Tensor = None,
        factorized: bool | None = None,
    ) -> torch.Tensor:
        if factorized is None:
            factorized = self.factorized
        bs = len(layer_indices)

        splitted_out = self._hypernet_forward(
            layer_indices,
            layer_type,
            encoded_task_emb,
        )

        if self.output_space == "lora":
            A, B = splitted_out
            A = A.reshape(
                bs, self.peft_config.r, self.in_features[layer_type]
            )  # .transpose(-1, -2)
            B = B.reshape(
                bs, self.peft_config.r, self.out_features[layer_type]
            ).transpose(-1, -2)
            if self.training_task == "sft":
                A = A + self.AB_offset["A"][layer_type][layer_indices]
                B = B + self.AB_offset["B"][layer_type][layer_indices]
            if factorized:
                return A, B
            deltaW = torch.bmm(B, A)

        elif self.output_space == "vera":
            raise NotImplementedError("Vera output space is deprecated")
        # the deltaW should have shape [bs, (len(layer_indices)), out_features, in_features]
        return deltaW

    @torch.no_grad()
    def gen_lora(self, layer_indices, encoded_task_emb):
        assert encoded_task_emb.shape[0] == 1, (
            "Only one task at a time is supported for now"
        )
        lora_A, lora_B = dict(), dict()
        for target_module in self.target_modules:
            factorized_delta_w = self.get_delta_weights(
                layer_indices,
                target_module,
                encoded_task_emb.expand(layer_indices.shape[0], -1),
                factorized=True,
            )
            if self.output_space == "lora":
                lora_A[target_module], lora_B[target_module] = factorized_delta_w

        # save deltaW to lora state dict format
        lora_state_dict = dict()
        for target_module in self.target_modules:
            for layer_idx in layer_indices:
                for module_name in self.module_names[target_module][layer_idx]:
                    if "lora_A" in module_name:
                        lora_state_dict[module_name] = (
                            lora_A[target_module][layer_idx].cpu().contiguous()
                        )
                    elif "lora_B" in module_name:
                        lora_state_dict[module_name] = (
                            lora_B[target_module][layer_idx].cpu().contiguous()
                        )
                    else:
                        raise ValueError(f"Unexpected module name: {module_name}")
        if self.training_task == "recon":
            lora_state_dict = self.convert_to_raw_scale(lora_state_dict, layer_indices)
        return lora_state_dict


def get_in_out_features(
    model: PeftModel,
    peft_config: PeftConfig = None,
) -> tuple[dict[str, int], dict[str, int]]:
    if peft_config is None:
        peft_config = model.peft_config["default"]
    in_features = dict()
    out_features = dict()
    for module_name, module in model.named_modules():
        if not check_target_module_exists(peft_config, module_name):
            continue
        if not isinstance(module, BaseTunerLayer):
            continue
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        assert isinstance(module.base_layer, nn.Linear), (
            "all modules should be a leave module that is Linear layer"
        )

        # this should always pass
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules, (
            f"Module {name} not in target modules"
        )

        if name not in in_features:
            in_features[name] = module.in_features
            out_features[name] = module.out_features
        else:
            # assumes each module has the same input and output features
            assert in_features[name] == module.in_features
            assert out_features[name] == module.out_features

    return in_features, out_features


def get_init_peft_weights(model: PeftModel, peft_config: PeftConfig = None):
    if peft_config is None:
        peft_config = model.peft_config["default"]
    peft_weights = {module_name: dict() for module_name in peft_config.target_modules}
    adapter_name = "default"
    for module_name, module in model.named_modules():
        if not check_target_module_exists(peft_config, module_name):
            continue
        if not isinstance(module, BaseTunerLayer):
            continue
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        assert isinstance(module.base_layer, nn.Linear), (
            "all modules should be a leave module that is Linear layer"
        )

        # this should always pass
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules

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


def save_lora(lora_state_dict, adapter_config, lora_dir):
    logger.debug(f"lora_dir: {lora_dir}")
    os.makedirs(lora_dir, exist_ok=True)
    adapter_config.save_pretrained(lora_dir)
    save_file(lora_state_dict, f"{lora_dir}/adapter_model.safetensors")


def create_hypermod(
    args, peft_type, device, model, layer_indices, task_emb_size, from_scratch=True
):
    assert args.training_task in ["sft", "recon"], (
        f"Invalid training task: {args.training_task}"
    )
    module_names = get_lora_module_names(model, args.target_modules, layer_indices)

    mt_lora_sd = mt_lora_td = mean_recon_target = std_recon_target = None
    if from_scratch:
        if args.mt_lora_path:
            mt_lora_sd = load_peft_weights(args.mt_lora_path)
            mt_lora_td = lora_state_dict_to_tensor_dict(
                mt_lora_sd, args.target_modules, layer_indices, device=device
            )

    hypermod = HyperModulator(
        model,
        training_task=args.training_task,
        pred_z_score=args.pred_z_score,
        mean_recon_target=mean_recon_target,
        std_recon_target=std_recon_target,
        output_space=peft_type,
        module_names=module_names,
        match_lora_init=args.mt_lora_path is None and args.training_task == "sft",
        task_emb_size=len(args.train_ds_names)
        if args.use_one_hot_task_emb
        else task_emb_size,
        shared_AB_head=args.shared_AB_head,
        autoreg_gen=args.autoreg_gen,
        learnable_pos_emb=args.learnable_pos_emb,
        AB_offset=mt_lora_td,
        learnable_AB_offset=args.learnable_AB_offset,
        zero_init_head=args.training_task == "sft",
        latent_size=args.hypernet_latent_size,
        head_in_size=args.head_in_size,
        head_use_bias=(args.training_task == "sft" and args.mt_lora_path is None)
        or (args.training_task == "recon" and not args.pred_z_score),
        factorized=getattr(args, "factorized", False),
        delta_w_scaling=getattr(args, "delta_w_scaling", 10000),
    ).to(device)

    return hypermod


def save_hypermod_checkpoint(save_dir, hypermod, curstep):
    save_path = f"{save_dir}/checkpoints/it_{curstep}/hypermod.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(hypermod.state_dict(), save_path)
    return save_path


def load_hypermod_checkpoint(checkpoint_path, device):
    base_hypermod_dir = os.path.dirname(checkpoint_path)
    if "checkpoint" in base_hypermod_dir:
        base_hypermod_dir = base_hypermod_dir.split("checkpoint")[0]

    args = argparse.Namespace(**yaml.safe_load(open(f"{base_hypermod_dir}/args.yaml")))
    peft_config = get_peft_config(
        PeftConfig.from_json_file(f"{base_hypermod_dir}/adapter_config.json")
    )
    peft_type = peft_config.peft_type.lower()
    state_dict = torch.load(checkpoint_path, map_location=device)

    model, tokenizer = get_model_and_tokenizer(
        args.model_dir,
        train=False,
        requires_grad=False,
        peft_config=peft_config,
        model_kwargs={"output_hidden_states": True, "output_attentions": False},
        device=device,
    )
    # train to output delta_w for all layers
    layer_indices = torch.tensor(
        range(len(get_layers(model))), dtype=torch.long, device=device
    )

    task_emb_size = emb_model = emb_tokenizer = task_desc_format_fn = pooling_fn = None
    if not args.use_one_hot_task_emb:
        emb_tokenizer = deepcopy(tokenizer)
        task_desc_format_fn = add_full_stop
        pooling_fn = get_pooling_fn("last_token")

        if args.emb_model:
            emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn = (
                get_emb_model_and_fns(args.emb_model, device)
            )
            logger.debug(f"emb_model: {emb_model}")
        emb_model.eval()
        task_emb_size = emb_model.config.hidden_size

    hypermod = create_hypermod(
        args, peft_type, device, model, layer_indices, task_emb_size, from_scratch=False
    )
    info = hypermod.load_state_dict(state_dict, strict=False)
    print(f"Loaded hypermod state dict: {info}")
    hypermod.eval().to(device)
    return (
        args,
        hypermod,
        model,
        tokenizer,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
    )


def load_hypermod(hypermod_dir, device):
    checkpoint_path = f"{hypermod_dir}/hypermod.pt"
    (
        args,
        hypermod,
        model,
        tokenizer,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
    ) = load_hypermod_checkpoint(checkpoint_path, device)
    return (
        args,
        hypermod,
        model,
        tokenizer,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
    )


def embed_texts(
    texts,
    emb_model,
    emb_tokenizer,
    task_desc_format_fn,
    pooling_fn,
    device,
    batch_size=None,
):
    formatted_descs = list(map(task_desc_format_fn, texts))
    tokenized_ds_descs = emb_tokenizer(
        formatted_descs,
        truncation=True,
        padding=True,
        max_length=2**13,
        return_tensors="pt",
    )
    return embed_tokens(tokenized_ds_descs, emb_model, pooling_fn, device, batch_size)


def embed_tokens(tokenized_texts, emb_model, pooling_fn, device, batch_size=None):
    if batch_size is None:
        # Process all at once if no batch size specified
        tokenized_texts = {k: v.to(device) for k, v in tokenized_texts.items()}
        return _embed_tokens_single_batch(tokenized_texts, emb_model, pooling_fn)

    # Process in batches
    n_samples = tokenized_texts["input_ids"].shape[0]
    embeddings = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = {k: v[start_idx:end_idx].to(device) for k, v in tokenized_texts.items()}
        batch_embeddings = _embed_tokens_single_batch(batch, emb_model, pooling_fn)
        embeddings.append(batch_embeddings)

    return torch.cat(embeddings, dim=0)


def _embed_tokens_single_batch(tokenized_texts, emb_model, pooling_fn):
    outputs = emb_model(**tokenized_texts, output_hidden_states=True)
    task_embs = pooling_fn(outputs, tokenized_texts["attention_mask"]).to(torch.float32)
    return torch.nn.functional.normalize(task_embs) * sqrt(task_embs.shape[-1])


if __name__ == "__main__":
    import random
    import string
    import sys
    import time

    hypermod_dir = sys.argv[1]
    task_desc = sys.argv[2].strip("\"' ")

    print(f"\nGenerating LoRA for description:\n\n{task_desc}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load metadata
    args = argparse.Namespace(**yaml.safe_load(open(f"{hypermod_dir}/args.yaml")))
    peft_config = get_peft_config(
        PeftConfig.from_json_file(f"{hypermod_dir}/adapter_config.json")
    )
    curtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    uuid = "".join(
        [random.choice(string.ascii_letters + string.digits) for _ in range(8)]
    )
    (
        args,
        hypermod,
        model,
        tokenizer,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
    ) = load_hypermod(hypermod_dir, device)
    layer_indices = range(len(get_layers(model)))
    layer_indices = torch.tensor(layer_indices, dtype=torch.long, device=device)
    emb_size = emb_model.config.hidden_size

    # generate loras
    task_emb = embed_texts(
        [task_desc], emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device
    )
    encoder_out = hypermod.task_encoder(task_emb)
    encoded_task_emb = encoder_out["encoded_task_emb"].detach()
    lora_sd = hypermod.gen_lora(layer_indices, encoded_task_emb)
    lora_dir = f"{hypermod_dir}/extras/user_generated/{curtime}_{uuid}/"
    save_lora(lora_sd, peft_config, lora_dir)
    with open(f"{lora_dir}/task_desc.txt", "w") as f:
        f.write(task_desc)
    print(f"Saved lora to {lora_dir}")
