from collections.abc import Iterable
from functools import partial

import torch
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float, Integer
from torch import Tensor

from peft.tuners.tuners_utils import BaseTunerLayer

from ctx_to_lora.utils import get_layers, peft_module_target_type


def lora_forward(
    x: Float[Tensor, "tot_q seq_len d_in"],
    n_qs: Integer[Tensor, "n_ctx"],
    tot_q: int,
    A: Float[Tensor, "n_ctx r d_in"],
    B: Float[Tensor, "n_ctx r d_out"],
    lora_dropout_p: float,
    scaling: float,
    self,
    *args,
    **kwargs,
) -> Float[Tensor, "tot_q seq_len d_out"]:
    # A: [n_ctx, r, d_in] -> [tot_q, r, d_in]
    A = A.repeat_interleave(n_qs, dim=0, output_size=tot_q)
    # B: [n_ctx, d_out, r] -> [tot_q, d_out, r]
    B = B.repeat_interleave(n_qs, dim=0, output_size=tot_q)

    base_out = torch.nn.Linear.forward(self, x, *args, **kwargs)
    x = x.to(A.dtype)
    delta_x = F.dropout(x, p=lora_dropout_p, training=self.training)
    delta_x = einsum(A, delta_x, "tot_q r d_in, tot_q s_len d_in -> tot_q s_len r")
    delta_x = einsum(B, delta_x, "tot_q r d_out, tot_q s_len r -> tot_q s_len d_out")
    delta_x = delta_x * scaling
    return (base_out + delta_x).to(base_out.dtype)


def lora_forward_packed(
    x: Float[Tensor, "1 tot_len d_in"],
    n_qs: Integer[Tensor, "n_ctx"],
    tot_q: int,
    seq_lens: Integer[Tensor, "tot_q"],
    tot_len: int,
    A: Float[Tensor, "n_ctx r d_in"],
    B: Float[Tensor, "n_ctx r d_out"],
    lora_dropout_p: float,
    scaling: float,
    self,
    *args,
    **kwargs,
) -> Float[Tensor, "1 tot_len d_out"]:
    # bs of x should be 1 in this case
    base_out = torch.nn.Linear.forward(self, x, *args, **kwargs)
    x = x.to(A.dtype)
    dropped_x = F.dropout(x, p=lora_dropout_p, training=self.training)
    query_A = A.repeat_interleave(n_qs, dim=0, output_size=tot_q)
    query_B = B.repeat_interleave(n_qs, dim=0, output_size=tot_q)

    delta_x = base_out.new_zeros(base_out.shape, dtype=dropped_x.dtype)
    start = 0
    for query_idx, seq_len in enumerate(seq_lens.tolist()):
        end = start + seq_len
        delta_rank = einsum(
            query_A[query_idx],
            dropped_x[:, start:end],
            "r d_in, bs seq_len d_in -> bs seq_len r",
        )
        delta_x[:, start:end] = einsum(
            query_B[query_idx],
            delta_rank,
            "r d_out, bs seq_len r -> bs seq_len d_out",
        )
        start = end
    delta_x = delta_x * scaling

    return (base_out + delta_x).to(base_out.dtype)


def apply_lora_to_layers(
    model: torch.nn.Module,
    layer_indices: Iterable[int],
    generated_loras: dict[str, dict[str, Float[Tensor, "n_ctx n_layers r _"]]],
    n_qs: Integer[Tensor, "n_ctx"],
    position_ids: Integer[Tensor, "bs seq_len"] = None,
) -> None:
    layers = get_layers(model)
    if position_ids is not None:
        position_ids = position_ids.squeeze(0)
        seq_lens = position_ids[torch.where(position_ids == 0)[0][1:] - 1]
        seq_lens = torch.cat(
            [seq_lens, torch.tensor([position_ids[-1]], device=seq_lens.device)]
        )
        seq_lens += 1
        tot_len = seq_lens.sum().item()
    tot_q = n_qs.sum().item()
    for generated_idx, layer_idx in enumerate(layer_indices):
        layer_idx = int(layer_idx)
        layer = layers[layer_idx]

        for module_name, module in layer.named_modules():
            if not isinstance(module, BaseTunerLayer):
                continue
            mname = peft_module_target_type(module_name, module)
            if mname not in generated_loras:
                continue
            A = generated_loras[mname]["A"][:, generated_idx]
            B = generated_loras[mname]["B"][:, generated_idx]
            generated_forward = getattr(
                module, "generated_lora_forward", module.forward
            )
            module.forward = partial(
                generated_forward, n_qs=n_qs, tot_q=tot_q, A=A, B=B
            )
            if position_ids is not None:
                module.forward = partial(
                    module.forward, seq_lens=seq_lens, tot_len=tot_len
                )


def clear_lora_from_layers(
    model: torch.nn.Module,
    layer_indices: Iterable[int],
) -> None:
    """Drop dynamically bound generated-LoRA tensor references.

    ``apply_lora_to_layers`` installs each adapter through a ``partial`` whose
    closure owns its A/B tensors.  Streaming training has already extracted
    gradients with respect to those leaf tensors before this cleanup runs, so
    restoring the stable patched forward releases the last adapter without
    changing any completed forward or backward computation.
    """

    layers = get_layers(model)
    for layer_idx in layer_indices:
        layer = layers[int(layer_idx)]
        for module in layer.modules():
            if not isinstance(module, BaseTunerLayer):
                continue
            generated_forward = getattr(module, "generated_lora_forward", None)
            if generated_forward is not None:
                module.forward = generated_forward
