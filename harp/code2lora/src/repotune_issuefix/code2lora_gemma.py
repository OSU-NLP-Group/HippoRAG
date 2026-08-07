#!/usr/bin/env python3
"""Gemma-aware Code2LoRA primitives for static issue-fixing training.

This module intentionally mirrors the useful parts of Code2LoRA's
``code2lora_core.py`` while adapting module discovery for Gemma 4:

* target discovery can be scoped to ``model.language_model.layers.*`` so audio
  and vision towers are ignored;
* layer-index parsing understands both ``model.layers.N`` and
  ``model.language_model.layers.N``;
* target types can be shape-aware, e.g. ``q_proj__in1536__out2048``.
"""

from __future__ import annotations

import importlib.util
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_GEMMA4_TEXT_REGEX = r"^model\.language_model\.layers\."
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "gate_proj",
    "down_proj",
]


class LoRA(nn.Module):
    """Frozen linear layer plus externally generated low-rank delta."""

    def __init__(self, base: nn.Linear, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(max(1, rank))
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None

    def set_lora_weights(self, A: torch.Tensor, B: torch.Tensor) -> None:
        self.A = A
        self.B = B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.A is None or self.B is None:
            return y
        x_f32 = x.detach().to(torch.float32)
        delta = F.linear(F.linear(x_f32, self.A.to(torch.float32)), self.B.to(torch.float32))
        return y + (delta * self.scaling).to(dtype=y.dtype)


@dataclass
class ModuleSpec:
    full_name: str
    layer_idx: int
    type: str
    base_type: str
    in_features: int
    out_features: int


def shape_aware_type(base_type: str, in_features: int, out_features: int) -> str:
    return f"{base_type}__in{int(in_features)}__out{int(out_features)}"


def _layer_index(name: str) -> int:
    match = re.search(r"\bmodel\.(?:language_model\.)?layers\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def get_module_specs(
    model: nn.Module,
    target_module_types: Iterable[str] = DEFAULT_TARGET_MODULES,
    *,
    module_name_regex: str = DEFAULT_GEMMA4_TEXT_REGEX,
    shape_aware_types: bool = True,
) -> List[ModuleSpec]:
    """Discover target ``nn.Linear`` modules.

    ``shape_aware_types=True`` is the important Gemma 4 adaptation. Bare
    projection names have multiple shapes in Gemma 4, so the hypernetwork must
    generate one A/B pair per projection-shape group.
    """
    target_types = list(target_module_types)
    name_re = re.compile(module_name_regex) if module_name_regex else None
    specs: List[ModuleSpec] = []
    for name, module in model.named_modules():
        if name_re is not None and not name_re.search(name):
            continue
        base_type = next((t for t in target_types if t in name), None)
        if base_type is None or not isinstance(module, nn.Linear):
            continue
        type_name = (
            shape_aware_type(base_type, module.in_features, module.out_features)
            if shape_aware_types
            else base_type
        )
        specs.append(ModuleSpec(
            full_name=name,
            layer_idx=_layer_index(name),
            type=type_name,
            base_type=base_type,
            in_features=int(module.in_features),
            out_features=int(module.out_features),
        ))
    specs.sort(key=lambda sp: (sp.layer_idx, sp.full_name))
    return specs


def discover_module_types_and_dims(specs: Iterable[ModuleSpec]) -> Dict[str, Tuple[int, int]]:
    type_dims: Dict[str, Tuple[int, int]] = {}
    for sp in specs:
        dims = (int(sp.in_features), int(sp.out_features))
        if sp.type in type_dims and type_dims[sp.type] != dims:
            raise ValueError(f"type {sp.type} appears with inconsistent dims: {type_dims[sp.type]} vs {dims}")
        type_dims[sp.type] = dims
    return type_dims


def summarize_specs(specs: Iterable[ModuleSpec]) -> Dict[str, Any]:
    counts_by_type: Dict[str, int] = {}
    dims_by_base_type: Dict[str, set[Tuple[int, int]]] = {}
    for sp in specs:
        counts_by_type[sp.type] = counts_by_type.get(sp.type, 0) + 1
        dims_by_base_type.setdefault(sp.base_type, set()).add((sp.in_features, sp.out_features))
    return {
        "n_specs": sum(counts_by_type.values()),
        "n_types": len(counts_by_type),
        "counts_by_type": dict(sorted(counts_by_type.items())),
        "dims_by_base_type": {
            key: [list(dims) for dims in sorted(value)]
            for key, value in sorted(dims_by_base_type.items())
        },
    }


def replace_with_lora(model: nn.Module, specs: Iterable[ModuleSpec], *, rank: int, alpha: float) -> None:
    named = dict(model.named_modules())
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for sp in specs:
        parent_name, attr = sp.full_name.rsplit(".", 1)
        original = getattr(named[parent_name], attr)
        if isinstance(original, LoRA):
            continue
        if not isinstance(original, nn.Linear):
            raise TypeError(f"{sp.full_name} is not nn.Linear: {type(original)!r}")
        wrapped = LoRA(original, sp.in_features, sp.out_features, rank, alpha).to(device=device, dtype=dtype)
        setattr(named[parent_name], attr, wrapped)


def inject_lora_weights(
    model: nn.Module,
    specs: Iterable[ModuleSpec],
    head_out: Dict[str, Dict[str, torch.Tensor]],
    *,
    batch_index: int = 0,
) -> None:
    named = dict(model.named_modules())
    A_by_type = head_out["A"]
    B_by_type = head_out["B"]
    for sp in specs:
        named[sp.full_name].set_lora_weights(
            A_by_type[sp.type][batch_index],
            B_by_type[sp.type][batch_index],
        )


class Code2LoRAHead(nn.Module):
    """Map one repo-state embedding to generated LoRA A/B matrices."""

    def __init__(
        self,
        input_dim: int,
        type_dims: Dict[str, Tuple[int, int]],
        *,
        hidden_dim: int = 1024,
        rank: int = 8,
        init_log_scale: float = -3.5,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.type_dims = dict(type_dims)
        self.types = sorted(self.type_dims)
        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.heads_A = nn.ModuleDict({
            t: nn.Linear(self.hidden_dim, self.rank * self.type_dims[t][0])
            for t in self.types
        })
        self.heads_B = nn.ModuleDict({
            t: nn.Linear(self.hidden_dim, self.type_dims[t][1] * self.rank)
            for t in self.types
        })
        self.log_scale_A = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(init_log_scale)) for t in self.types
        })
        self.log_scale_B = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(init_log_scale)) for t in self.types
        })

    def forward(self, ctx: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        if ctx.dim() == 3:
            ctx = torch.max(ctx, dim=1).values
        h = self.trunk(ctx.float())
        h = F.normalize(h, p=2, dim=-1) * math.sqrt(self.hidden_dim)
        A_out: Dict[str, torch.Tensor] = {}
        B_out: Dict[str, torch.Tensor] = {}
        for type_name in self.types:
            in_f, out_f = self.type_dims[type_name]
            A_raw = self.heads_A[type_name](h).view(-1, self.rank, in_f)
            B_raw = self.heads_B[type_name](h).view(-1, out_f, self.rank)
            scale_A = torch.exp(self.log_scale_A[type_name]).clamp(1e-5, 0.3)
            scale_B = torch.exp(self.log_scale_B[type_name]).clamp(1e-5, 0.3)
            A_out[type_name] = torch.tanh(A_raw) * scale_A
            B_out[type_name] = torch.tanh(B_raw) * scale_B
        return {"A": A_out, "B": B_out}

    def config_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "rank": self.rank,
            "types": self.types,
            "type_dims": {key: list(value) for key, value in self.type_dims.items()},
        }


def count_head_parameters(input_dim: int, type_dims: Dict[str, Tuple[int, int]], *, hidden_dim: int, rank: int) -> int:
    trunk = input_dim * hidden_dim + hidden_dim + hidden_dim * hidden_dim + hidden_dim
    heads = 0
    for in_f, out_f in type_dims.values():
        heads += hidden_dim * (rank * in_f) + (rank * in_f)
        heads += hidden_dim * (out_f * rank) + (out_f * rank)
        heads += 2
    return int(trunk + heads)


def torch_dtype(name: str) -> Any:
    key = name.lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    if key == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {name}")


def model_load_kwargs(*, dtype: Any, device: str, local_files_only: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "torch_dtype": dtype,
    }
    if importlib.util.find_spec("accelerate") is not None:
        kwargs["low_cpu_mem_usage"] = True
        kwargs["device_map"] = {"": device}
    return kwargs


def load_gemma4_model(model_name: str, *, dtype: Any, device: str, local_files_only: bool):
    import transformers as hf_transformers

    auto_cls = getattr(hf_transformers, "AutoModelForMultimodalLM", None)
    if auto_cls is None:
        raise RuntimeError("This Transformers build lacks AutoModelForMultimodalLM; use a Gemma-4-capable env.")
    model = auto_cls.from_pretrained(
        model_name,
        **model_load_kwargs(dtype=dtype, device=device, local_files_only=local_files_only),
    )
    if next(model.parameters()).device.type != device:
        model = model.to(device)
    return model

