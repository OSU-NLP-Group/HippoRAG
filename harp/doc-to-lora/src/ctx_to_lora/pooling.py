from enum import Enum

import torch
from jaxtyping import Float, Integer
from torch import Tensor

POOL_FN = Enum("POOL_FN", ["MEAN", "MAX", "LAST_TOKEN"])


def inv_bool_mask(m: Integer[Tensor, "bs seq_len"]) -> Integer[Tensor, "bs seq_len 1"]:
    return (m - 1).bool().unsqueeze(-1)


def get_pooling_fn(pooling_type: str):
    if pooling_type == POOL_FN.MEAN:
        return mean_pool
    elif pooling_type == POOL_FN.MAX:
        return max_pool
    elif pooling_type == POOL_FN.LAST_TOKEN:
        return last_token_pool


def mean_pool(
    features: Float[Tensor, "bs seq_len feature_dim"],
    attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
) -> Float[Tensor, "bs 1 feature_dim"]:
    if attn_mask is not None:
        features = features.masked_fill(inv_bool_mask(attn_mask), 0)
    return features.sum(dim=1) / attn_mask.sum(dim=1).unsqueeze(1)


def max_pool(
    features: Float[Tensor, "bs seq_len feature_dim"],
    attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
) -> Float[Tensor, "bs 1 feature_dim"]:
    if attn_mask is not None:
        features = features.masked_fill(inv_bool_mask(attn_mask), -float("inf"))
    return torch.max(features, dim=1)


def last_token_pool(
    features: Float[Tensor, "bs seq_len feature_dim"],
    attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
) -> Float[Tensor, "bs feature_dim"]:
    left_padding = attn_mask[:, -1].sum() == attn_mask.shape[0]
    if left_padding:
        return features[:, -1]
    else:
        sequence_lengths = attn_mask.sum(dim=1) - 1
        batch_size = features.shape[0]
        return features[
            torch.arange(batch_size, device=features.device),
            sequence_lengths,
        ]
