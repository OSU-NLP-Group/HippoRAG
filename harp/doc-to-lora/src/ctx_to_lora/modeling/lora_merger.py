"""
Utilities for merging / aggregating LoRA adapters coming from multiple chunks.
"""

import torch
from einops import rearrange
from jaxtyping import Float, Integer
from torch import Tensor


def compute_rank(n_lora, rank):
    return (n_lora + 1) * rank


def combine_lora(
    generated_loras: dict[str, dict[str, Tensor]],
    n_chunks: Integer[Tensor, "n_ctx"],
    lora_bias: dict[str, dict[str, Tensor]] | None = None,
    scalers: Float[Tensor, "n_ctx"] | None = None,
    bias_scaler: float | None = None,
) -> dict[str, dict[str, Tensor]]:
    total_chunks = int(n_chunks.sum())
    if bias_scaler is None:
        bias_scaler = 1
    # Assume all modules share same base rank r
    first_module = next(iter(generated_loras))
    sampled_lora = generated_loras[first_module]["A"]
    base_rank = sampled_lora.shape[-2]
    device = sampled_lora.device
    dtype = sampled_lora.dtype
    max_rank_needed = int(compute_rank(n_chunks.max(), base_rank))

    combined_loras: dict[str, dict[str, Tensor]] = {
        module: {"A": None, "B": None} for module in generated_loras.keys()
    }
    rank_dim = 2
    num_groups = len(n_chunks)
    rank_per_group = (n_chunks * base_rank).tolist()
    bias_tensor = None
    for module_name, module_loras in generated_loras.items():
        for matrix_key in ("A", "B"):
            if lora_bias is not None:
                bias_tensor = lora_bias[module_name][matrix_key]
            loras = module_loras[matrix_key]
            if (scalers is not None) and (matrix_key == "A"):
                loras = loras * scalers[:, None, None, None]

            flat_loras = rearrange(
                loras, "tot_chunks n_layers r dim -> 1 n_layers (tot_chunks r) dim"
            )
            per_group_deltas = flat_loras.split(rank_per_group, dim=rank_dim)

            combined_shape = [num_groups, *per_group_deltas[0].shape[1:]]
            combined_shape[rank_dim] = max_rank_needed

            combined = torch.zeros(*combined_shape, device=device, dtype=dtype)

            for g, deltas in enumerate(per_group_deltas):
                combined_rank = deltas.shape[rank_dim]

                # Build slice pattern, slice up to combined_rank.
                # slice_pattern = [g, slice(None), slice(None), slice(None)]
                # slice_pattern[rank_dim] = slice(combined_rank)

                combined[g, :, :combined_rank, :] = deltas

                if bias_tensor is not None:
                    # bias_slice_pattern = [g, slice(None), slice(None), slice(None)]
                    # bias_slice_pattern[rank_dim] = slice(
                    #     combined_rank, combined_rank + base_rank
                    # )
                    combined[g, :, combined_rank : combined_rank + base_rank, :] = (
                        bias_tensor * bias_scaler
                    )

            combined_loras[module_name][matrix_key] = combined

    return combined_loras
