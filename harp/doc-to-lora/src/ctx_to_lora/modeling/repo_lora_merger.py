"""Repository-level composition for Doc-to-LoRA generated adapters.

The tensor mathematics follows the public TIES-Merging and KnOTS reference
implementations. The references are optional test dependencies and are not
vendored into this repository.

This module ports their tensor mathematics into differentiable PyTorch code.  It
deliberately merges dense ``B @ A`` updates rather than the non-identifiable
LoRA factors.  Optimized paths must remain forward-equivalent to the reference
operations before the final, explicitly reported low-rank reconstruction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ctx_to_lora.modeling.lora_merger import combine_lora


LoRADict = dict[str, dict[str, Tensor]]


class RepoMergeMethod(str, Enum):
    CONCAT = "concat"
    LEARNED_FUSION = "learned_fusion"
    TIES = "ties"
    STREAMING_TIES_EXACT = "streaming_ties_exact"
    KNOTS_TIES = "knots_ties"
    BM25_TOPK_TIES = "bm25_topk_ties"


@dataclass
class RepositoryMergerConfig:
    method: RepoMergeMethod = RepoMergeMethod.CONCAT
    output_rank: int = 64
    ties_keep_fraction: float = 0.2
    ties_sign_method: str = "sum"
    ties_merge_type: str = "mean"
    ties_merge_scale: float = 1.0
    knots_concat_across_output: bool = True
    knots_singular_value_epsilon: float = 1e-5
    retrieval_top_k: int = 8
    fusion_num_blocks: int = 2
    fusion_num_heads: int = 8
    svd_oversample: int = 8
    svd_power_iterations: int = 1
    svd_exact_max_dim: int = 512
    svd_seed: int = 17
    svd_singular_value_epsilon: float = 1e-7

    def __post_init__(self) -> None:
        self.method = RepoMergeMethod(self.method)
        if self.output_rank <= 0:
            raise ValueError("output_rank must be positive")
        if not 0 < self.ties_keep_fraction <= 1:
            raise ValueError("ties_keep_fraction must be in (0, 1]")
        if self.ties_sign_method not in {"sum", "sum_of_values", "sum_of_signs"}:
            raise ValueError(f"Unsupported TIES sign method: {self.ties_sign_method}")
        if self.ties_merge_type not in {"mean", "sum"}:
            raise ValueError(f"Unsupported TIES merge type: {self.ties_merge_type}")
        if self.method == RepoMergeMethod.STREAMING_TIES_EXACT:
            if self.ties_sign_method != "sum_of_signs":
                raise ValueError(
                    "streaming_ties_exact uses final sign-count voting; set "
                    "ties_sign_method=sum_of_signs"
                )
            if self.ties_merge_type != "mean":
                raise ValueError(
                    "streaming_ties_exact divides the selected sum by the total "
                    "chunk count; set ties_merge_type=mean"
                )
        if not self.knots_concat_across_output:
            raise ValueError(
                "The primary KnOTS implementation requires "
                "knots_concat_across_output=True"
            )
        if self.svd_singular_value_epsilon < 0:
            raise ValueError("svd_singular_value_epsilon must be non-negative")


def _group_slices(n_ctx_chunks: Tensor) -> list[slice]:
    counts = [int(x) for x in n_ctx_chunks.detach().cpu().tolist()]
    if not counts or any(x <= 0 for x in counts):
        raise ValueError(f"n_ctx_chunks must contain positive counts, got {counts}")
    out: list[slice] = []
    start = 0
    for count in counts:
        out.append(slice(start, start + count))
        start += count
    return out


def _validate_loras(loras: LoRADict, n_ctx_chunks: Tensor) -> None:
    if not loras:
        raise ValueError("At least one generated LoRA target is required")
    expected_chunks = int(n_ctx_chunks.sum().item())
    reference_shape = None
    for module, factors in loras.items():
        if set(factors) != {"A", "B"}:
            raise ValueError(f"{module} must contain exactly A and B factors")
        A, B = factors["A"], factors["B"]
        if A.ndim != 4 or B.ndim != 4:
            raise ValueError(
                f"Expected [chunks, layers, rank, dim] factors for {module}; "
                f"got A={tuple(A.shape)}, B={tuple(B.shape)}"
            )
        if A.shape[:3] != B.shape[:3] or A.shape[0] != expected_chunks:
            raise ValueError(
                f"Incompatible factors/counts for {module}: "
                f"A={tuple(A.shape)}, B={tuple(B.shape)}, chunks={expected_chunks}"
            )
        if reference_shape is None:
            reference_shape = A.shape[:3]
        elif A.shape[:2] != reference_shape[:2]:
            raise ValueError("All targets must use the same chunk and layer axes")


def _pad_rank(A: Tensor, B: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    current = A.shape[-2]
    if current > rank:
        raise ValueError(f"Cannot pad rank {current} into smaller rank {rank}")
    if current == rank:
        return A, B
    pad = rank - current
    return (
        torch.nn.functional.pad(A, (0, 0, 0, pad)),
        torch.nn.functional.pad(B, (0, 0, 0, pad)),
    )


def append_lora_bias(
    loras: LoRADict,
    lora_bias: LoRADict | None,
    bias_scaler: float | None = None,
) -> LoRADict:
    """Append the context-independent Doc-to-LoRA bias exactly once."""

    if lora_bias is None:
        return loras
    if bias_scaler is None:
        bias_scaler = 1.0
    out: LoRADict = {}
    for module, factors in loras.items():
        if module not in lora_bias:
            raise KeyError(f"Missing Doc-to-LoRA bias for target {module}")
        A, B = factors["A"], factors["B"]
        bias_A = lora_bias[module]["A"].to(device=A.device, dtype=A.dtype)
        bias_B = lora_bias[module]["B"].to(device=B.device, dtype=B.dtype)
        # Match combine_lora: its bias_scaler is applied to both stored
        # factors, hence to the dense update quadratically.
        bias_A = bias_A * bias_scaler
        bias_B = bias_B * bias_scaler
        bias_A = bias_A.unsqueeze(0).expand(A.shape[0], -1, -1, -1)
        bias_B = bias_B.unsqueeze(0).expand(B.shape[0], -1, -1, -1)
        out[module] = {
            "A": torch.cat((A, bias_A), dim=2),
            "B": torch.cat((B, bias_B), dim=2),
        }
    return out


def dense_lora_update(A: Tensor, B: Tensor) -> Tensor:
    """Return dense updates for factors shaped ``[..., rank, input/output]``."""

    return torch.einsum("...ro,...ri->...oi", B, A)


def _reference_topk_threshold(vector: Tensor, keep_fraction: float) -> Tensor:
    """Match the threshold convention in the supplied TIES repositories."""

    if keep_fraction >= 1:
        return vector.new_tensor(float("-inf"))
    flat = vector.abs().reshape(-1)
    # The reference uses kthvalue(d - int(d * K)), including its boundary
    # convention (which may retain one extra coordinate in the absence of ties).
    kth = max(1, flat.numel() - int(flat.numel() * keep_fraction))
    return flat.kthvalue(kth).values


def ties_merge_vectors(
    vectors: Tensor,
    keep_fraction: float,
    sign_method: str = "sum",
    merge_type: str = "mean",
    merge_scale: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Reference-faithful TIES over ``[num_updates, num_parameters]`` vectors."""

    if vectors.ndim != 2:
        raise ValueError(f"Expected a matrix of task vectors, got {vectors.shape}")
    if keep_fraction >= 1:
        trim_mask = torch.ones_like(vectors, dtype=torch.bool)
    else:
        thresholds = torch.stack(
            [_reference_topk_threshold(row.detach(), keep_fraction) for row in vectors]
        )
        trim_mask = vectors.abs() >= thresholds[:, None]
    trimmed = vectors * trim_mask

    if sign_method in {"sum", "sum_of_values"}:
        elected_sign = torch.sign(trimmed.sum(dim=0))
    elif sign_method == "sum_of_signs":
        elected_sign = torch.sign(torch.sign(trimmed).sum(dim=0))
    else:
        raise ValueError(f"Unsupported sign method: {sign_method}")

    majority_sign = torch.sign(elected_sign.sum())
    elected_sign = torch.where(
        elected_sign == 0, majority_sign.to(elected_sign.dtype), elected_sign
    )
    elected_sign = elected_sign.detach()
    agree_mask = torch.where(
        elected_sign.unsqueeze(0) > 0, trimmed > 0, trimmed < 0
    )
    selected = trimmed * agree_mask
    if merge_type == "mean":
        count = (selected != 0).sum(dim=0).clamp_min(1)
        merged = selected.sum(dim=0) / count
    elif merge_type == "sum":
        merged = selected.sum(dim=0)
    else:
        raise ValueError(f"Unsupported merge type: {merge_type}")
    merged = merged * merge_scale
    diagnostics = {
        "retained_density": trim_mask.float().mean().detach(),
        "sign_agreement_density": agree_mask.float().mean().detach(),
        "sign_conflict_rate": (trim_mask & ~agree_mask).float().mean().detach(),
    }
    return merged, diagnostics


def exact_streaming_ties_merge(
    updates: Tensor,
    thresholds: Tensor | None = None,
    sign_method: str = "sum_of_signs",
    merge_scale: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Merge dense updates using final-sign, order-independent accumulators.

    This is the exact streaming variant discussed for repository composition:
    each update contributes to (1) an election accumulator, (2) a positive
    value sum, and (3) a negative value sum.  The final election chooses one
    of the two value sums independently at every parameter.  The selected sum
    is divided by the total number of chunk LoRAs, including chunks that were
    trimmed or disagreed at that parameter.

    ``updates`` is accepted here as a tensor to make the mathematical primitive
    independently testable.  The production factor path below computes one
    dense ``B @ A`` update at a time and never constructs this stacked tensor.
    """

    if updates.ndim < 2 or updates.shape[0] == 0:
        raise ValueError(
            "Expected one or more dense updates shaped [chunks, ...], got "
            f"{tuple(updates.shape)}"
        )
    if sign_method not in {"sum", "sum_of_values", "sum_of_signs"}:
        raise ValueError(f"Unsupported sign method: {sign_method}")
    if thresholds is None:
        thresholds = updates.new_full((updates.shape[0],), float("-inf"))
    if thresholds.shape != (updates.shape[0],):
        raise ValueError(
            "thresholds must contain one scalar per chunk; got "
            f"{tuple(thresholds.shape)} for {updates.shape[0]} chunks"
        )

    accumulator_dtype = (
        torch.float32
        if updates.dtype in (torch.float16, torch.bfloat16)
        else updates.dtype
    )
    shape = updates.shape[1:]
    election = torch.zeros(shape, device=updates.device, dtype=accumulator_dtype)
    positive_sum = torch.zeros_like(election)
    negative_sum = torch.zeros_like(election)
    retained_entries = updates.new_zeros((), dtype=torch.float32)

    for chunk in range(updates.shape[0]):
        update = updates[chunk].to(accumulator_dtype)
        trim_mask = update.abs() >= thresholds[chunk].to(accumulator_dtype)
        trimmed = update * trim_mask
        if sign_method == "sum_of_signs":
            election = election + torch.sign(trimmed).detach()
        else:
            election = election + trimmed.detach()
        positive_sum = positive_sum + torch.clamp_min(trimmed, 0)
        negative_sum = negative_sum + torch.clamp_max(trimmed, 0)
        retained_entries = retained_entries + trim_mask.float().sum()

    final_sign = torch.sign(election).detach()
    # The exact forward value at a tied coordinate is zero.  A literal constant
    # zero branch would also make an all-zero LoRA initialization impossible to
    # leave, because sign election is discrete.  Preserve the exact forward
    # while using the signed update sum as a straight-through tie gradient.
    signed_sum = positive_sum + negative_sum
    tie_zero_with_gradient = signed_sum - signed_sum.detach()
    selected_sum = torch.where(
        final_sign > 0,
        positive_sum,
        torch.where(final_sign < 0, negative_sum, tie_zero_with_gradient),
    )
    merged = selected_sum / updates.shape[0]

    # Diagnostics are deliberately recomputed without retaining another dense
    # count matrix.  This keeps the persistent algorithmic state to the three
    # matrices above while preserving the existing reporting contract.
    agreeing_entries = retained_entries.new_zeros(())
    conflicting_entries = retained_entries.new_zeros(())
    with torch.no_grad():
        for chunk in range(updates.shape[0]):
            update = updates[chunk].to(accumulator_dtype)
            trim_mask = update.abs() >= thresholds[chunk].to(accumulator_dtype)
            trimmed = update * trim_mask
            agree = torch.where(
                final_sign > 0,
                trimmed > 0,
                torch.where(final_sign < 0, trimmed < 0, torch.zeros_like(trim_mask)),
            )
            agreeing_entries = agreeing_entries + agree.float().sum()
            conflicting_entries = conflicting_entries + (
                trim_mask & ~agree
            ).float().sum()

    total_entries = updates.shape[0] * updates[0].numel()
    diagnostics = {
        "retained_density": (retained_entries / total_entries).detach(),
        "sign_agreement_density": (agreeing_entries / total_entries).detach(),
        "sign_conflict_rate": (conflicting_entries / total_entries).detach(),
        "final_sign_tie_density": (final_sign == 0).float().mean().detach(),
    }
    return (merged * merge_scale).to(updates.dtype), diagnostics


def _randomized_svd(
    matrix: Tensor,
    rank: int,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    q = min(rank + oversample, *matrix.shape)
    generator = torch.Generator(device=matrix.device)
    generator.manual_seed(seed)
    omega = torch.randn(
        matrix.shape[1], q, device=matrix.device, dtype=matrix.dtype, generator=generator
    )
    sample = matrix @ omega
    for _ in range(power_iterations):
        sample = matrix @ (matrix.mT @ sample)
    Q, _ = torch.linalg.qr(sample, mode="reduced")
    small = Q.mT @ matrix
    small_U, singular_values, Vh = torch.linalg.svd(small, full_matrices=False)
    return Q @ small_U, singular_values, Vh


def truncated_svd_lora(
    matrix: Tensor,
    rank: int,
    config: RepositoryMergerConfig,
    zero_gradient_basis: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Factor a dense update into stored ``A[r, in]`` and ``B[r, out]``."""

    rank = min(rank, *matrix.shape)
    output_dtype = matrix.dtype
    # CUDA QR/SVD do not implement fp16/bf16. Autocast would otherwise undo
    # the explicit promotion, so disable it for this small factorization only.
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        factor_matrix = (
            matrix.float()
            if matrix.dtype in (torch.float16, torch.bfloat16)
            else matrix
        )
        if min(factor_matrix.shape) <= config.svd_exact_max_dim:
            U, singular_values, Vh = torch.linalg.svd(
                factor_matrix, full_matrices=False
            )
        else:
            U, singular_values, Vh = _randomized_svd(
                factor_matrix,
                rank,
                config.svd_oversample,
                config.svd_power_iterations,
                config.svd_seed,
            )
        U = U[:, :rank]
        singular_values = singular_values[:rank]
        Vh = Vh[:rank]
        # Zero/repeated trailing singular directions are both uninformative and
        # make d(sqrt(s))/ds undefined at s=0. Drop only those numerical-null
        # directions, then let the caller pad the representation to fixed rank.
        keep = singular_values > config.svd_singular_value_epsilon
        if not bool(keep.any()):
            if zero_gradient_basis:
                # A literal pair of zero factors has zero derivative through
                # B.T @ A and permanently traps a zero-initialized streaming
                # merge. Keep the forward update exactly zero, but expose a
                # deterministic rank-r projection of d(matrix) through B.
                generator = torch.Generator(device=factor_matrix.device)
                generator.manual_seed(config.svd_seed)
                random_basis = torch.randn(
                    factor_matrix.shape[1],
                    rank,
                    device=factor_matrix.device,
                    dtype=factor_matrix.dtype,
                    generator=generator,
                )
                basis, _ = torch.linalg.qr(random_basis, mode="reduced")
                A = basis.mT
                projected_B = (factor_matrix @ basis).mT
                B = projected_B - projected_B.detach()
                return A.to(output_dtype), B.to(output_dtype), matrix.new_zeros(())
            # Avoid entering SVD backward for a numerically zero matrix: its
            # singular vectors are arbitrary and their repeated-zero gradient
            # is undefined. These zero factors retain a finite zero-gradient
            # connection to the dense TIES result.
            A = factor_matrix[:rank, :] * 0
            B = factor_matrix[:, :rank].mT * 0
            return A.to(output_dtype), B.to(output_dtype), matrix.new_zeros(())
        U = U[:, keep]
        singular_values = singular_values[keep]
        Vh = Vh[keep]
        sqrt_s = singular_values.sqrt()
        A = sqrt_s[:, None] * Vh
        B = (U * sqrt_s[None, :]).mT
        reconstruction = B.mT @ A
        denominator = factor_matrix.norm().clamp_min(
            torch.finfo(factor_matrix.dtype).eps
        )
        error = (factor_matrix - reconstruction).norm() / denominator
    return A.to(output_dtype), B.to(output_dtype), error.detach()


class RepositoryFusionBlock(nn.Module):
    def __init__(self, latent_size: int, num_heads: int):
        super().__init__()
        if latent_size % num_heads:
            raise ValueError("Fusion latent size must be divisible by num_heads")
        self.cross_norm = nn.LayerNorm(latent_size)
        self.cross_attention = nn.MultiheadAttention(
            latent_size, num_heads, batch_first=True
        )
        self.self_norm = nn.LayerNorm(latent_size)
        self.self_attention = nn.MultiheadAttention(
            latent_size, num_heads, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(latent_size)
        self.ff = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.SiLU(),
            nn.Linear(latent_size * 4, latent_size),
        )

    def forward(self, queries: Tensor, values: Tensor) -> Tensor:
        q = self.cross_norm(queries)
        queries = queries + self.cross_attention(q, values, values, need_weights=False)[0]
        q = self.self_norm(queries)
        queries = queries + self.self_attention(q, q, q, need_weights=False)[0]
        return queries + self.ff(self.ff_norm(queries))


class LearnedRepositoryFusion(nn.Module):
    """Set-Perceiver fusion over compact per-chunk, per-rank latents."""

    def __init__(self, latent_size: int, output_rank: int, num_blocks: int, num_heads: int):
        super().__init__()
        self.output_rank = output_rank
        self.queries = nn.Parameter(
            torch.randn(output_rank, latent_size) / math.sqrt(latent_size)
        )
        self.blocks = nn.ModuleList(
            [RepositoryFusionBlock(latent_size, num_heads) for _ in range(num_blocks)]
        )

    def forward(self, chunk_latents: Tensor, n_ctx_chunks: Tensor) -> Tensor:
        # chunk_latents: [chunks, layers, modules, chunk_rank, latent]
        if chunk_latents.ndim != 5:
            raise ValueError(f"Unexpected chunk latent shape: {chunk_latents.shape}")
        fused_repositories = []
        for group in _group_slices(n_ctx_chunks):
            values = chunk_latents[group]
            count, layers, modules, chunk_rank, latent = values.shape
            values = values.permute(1, 2, 0, 3, 4).reshape(
                layers * modules, count * chunk_rank, latent
            )
            queries = self.queries.unsqueeze(0).expand(layers * modules, -1, -1)
            for block in self.blocks:
                queries = block(queries, values)
            # The reused Doc-to-LoRA head is trained on unit-norm pre-head
            # latents, so keep that decoder contract after repository fusion.
            queries = queries / queries.norm(dim=-1, keepdim=True).clamp_min(
                torch.finfo(queries.dtype).eps
            )
            fused_repositories.append(
                queries.reshape(layers, modules, self.output_rank, latent)
            )
        return torch.stack(fused_repositories, dim=0)


class RepositoryLoRAMerger(nn.Module):
    def __init__(self, config: RepositoryMergerConfig, latent_size: int | None = None):
        super().__init__()
        self.config = config
        self.last_diagnostics: dict[str, Tensor | float | str] = {}
        self.latent_fusion = None
        if config.method == RepoMergeMethod.LEARNED_FUSION:
            if latent_size is None:
                raise ValueError("learned_fusion requires latent_size")
            self.latent_fusion = LearnedRepositoryFusion(
                latent_size,
                config.output_rank,
                config.fusion_num_blocks,
                config.fusion_num_heads,
            )

    @property
    def requires_chunk_latents(self) -> bool:
        return self.config.method == RepoMergeMethod.LEARNED_FUSION

    def forward(
        self,
        chunk_loras: LoRADict,
        n_ctx_chunks: Tensor,
        lora_bias: LoRADict | None = None,
        chunk_latents: Tensor | None = None,
        latent_decoder: Callable[[Tensor], LoRADict] | None = None,
        scalers: Tensor | None = None,
        bias_scaler: float | None = None,
    ) -> LoRADict:
        _validate_loras(chunk_loras, n_ctx_chunks)
        method = self.config.method
        if method == RepoMergeMethod.CONCAT:
            return combine_lora(
                chunk_loras,
                n_ctx_chunks,
                lora_bias=lora_bias,
                scalers=scalers,
                bias_scaler=bias_scaler,
            )
        # For one-chunk examples use the literal original Doc-to-LoRA operation
        # (base content rank plus bias), not merely a numerically equivalent
        # rank-64 zero-padded adapter.
        if bool(torch.all(n_ctx_chunks == 1)):
            self.last_diagnostics = {
                "method": method.value,
                "svd_reconstruction_error": 0.0,
                "retained_density": 1.0,
                "sign_agreement_density": 1.0,
                "sign_conflict_rate": 0.0,
            }
            return combine_lora(
                chunk_loras,
                n_ctx_chunks,
                lora_bias=lora_bias,
                scalers=scalers,
                bias_scaler=bias_scaler,
            )
        if scalers is not None:
            if scalers.numel() != int(n_ctx_chunks.sum().item()):
                raise ValueError("scalers must contain one value per context chunk")
            if method == RepoMergeMethod.LEARNED_FUSION:
                raise ValueError(
                    "Per-chunk inference scalers are undefined for learned fusion"
                )
            chunk_loras = {
                module: {
                    "A": factors["A"] * scalers[:, None, None, None],
                    "B": factors["B"],
                }
                for module, factors in chunk_loras.items()
            }
        if method == RepoMergeMethod.LEARNED_FUSION:
            if chunk_latents is None or latent_decoder is None or self.latent_fusion is None:
                raise ValueError("learned_fusion requires chunk latents and a decoder")
            fused = self.latent_fusion(chunk_latents, n_ctx_chunks)
            merged = latent_decoder(fused)
            # Preserve the exact ordinary Doc-to-LoRA path for one-chunk groups.
            single_chunk_groups = {
                repo_index: group
                for repo_index, group in enumerate(_group_slices(n_ctx_chunks))
                if group.stop - group.start == 1
            }
            for module in merged:
                rows_A, rows_B = [], []
                for repo_index in range(len(n_ctx_chunks)):
                    group = single_chunk_groups.get(repo_index)
                    if group is None:
                        rows_A.append(merged[module]["A"][repo_index])
                        rows_B.append(merged[module]["B"][repo_index])
                        continue
                    A, B = _pad_rank(
                        chunk_loras[module]["A"][group],
                        chunk_loras[module]["B"][group],
                        self.config.output_rank,
                    )
                    rows_A.append(A[0])
                    rows_B.append(B[0])
                merged[module] = {
                    "A": torch.stack(rows_A),
                    "B": torch.stack(rows_B),
                }
            return append_lora_bias(merged, lora_bias, bias_scaler)
        if method in {RepoMergeMethod.TIES, RepoMergeMethod.BM25_TOPK_TIES}:
            if method == RepoMergeMethod.BM25_TOPK_TIES:
                largest = int(n_ctx_chunks.max().item())
                if largest > self.config.retrieval_top_k:
                    raise ValueError(
                        "BM25 selection must happen before context encoding: "
                        f"received {largest} chunks with retrieval_top_k="
                        f"{self.config.retrieval_top_k}"
                    )
            merged = self._merge_ties(chunk_loras, n_ctx_chunks)
            return append_lora_bias(merged, lora_bias, bias_scaler)
        if method == RepoMergeMethod.STREAMING_TIES_EXACT:
            merged = self._merge_streaming_ties_exact(chunk_loras, n_ctx_chunks)
            return append_lora_bias(merged, lora_bias, bias_scaler)
        if method == RepoMergeMethod.KNOTS_TIES:
            merged = self._merge_knots_ties(chunk_loras, n_ctx_chunks)
            return append_lora_bias(merged, lora_bias, bias_scaler)
        raise ValueError(f"Unsupported repository merger: {method}")

    def _single_chunk_fixed_rank(self, repo_loras: LoRADict) -> LoRADict:
        out: LoRADict = {}
        for module, factors in repo_loras.items():
            A, B = _pad_rank(
                factors["A"], factors["B"], self.config.output_rank
            )
            out[module] = {"A": A, "B": B}
        return out

    @torch.no_grad()
    def _global_ties_thresholds(self, repo_loras: LoRADict) -> Tensor:
        num_chunks = next(iter(repo_loras.values()))["A"].shape[0]
        thresholds = []
        for chunk in range(num_chunks):
            values = []
            for module in sorted(repo_loras):
                A = repo_loras[module]["A"][chunk]
                B = repo_loras[module]["B"][chunk]
                for layer in range(A.shape[0]):
                    values.append(
                        dense_lora_update(A[layer].float(), B[layer].float())
                        .abs()
                        .reshape(-1)
                    )
            thresholds.append(
                _reference_topk_threshold(
                    torch.cat(values), self.config.ties_keep_fraction
                )
            )
        return torch.stack(thresholds)

    @torch.no_grad()
    def _global_ties_majority(
        self, repo_loras: LoRADict, thresholds: Tensor
    ) -> Tensor:
        sign_mass = thresholds.new_zeros(())
        for module in sorted(repo_loras):
            A, B = repo_loras[module]["A"], repo_loras[module]["B"]
            for layer in range(A.shape[1]):
                updates = dense_lora_update(
                    A[:, layer].float(), B[:, layer].float()
                )
                trimmed = updates * (updates.abs() >= thresholds[:, None, None])
                if self.config.ties_sign_method == "sum_of_signs":
                    signs = torch.sign(torch.sign(trimmed).sum(dim=0))
                else:
                    signs = torch.sign(trimmed.sum(dim=0))
                sign_mass = sign_mass + signs.sum()
        return torch.sign(sign_mass)

    def _merge_dense_layer(
        self, updates: Tensor, thresholds: Tensor, majority_sign: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        trim_mask = updates.abs() >= thresholds[:, None, None]
        trimmed = updates * trim_mask
        if self.config.ties_sign_method == "sum_of_signs":
            signs = torch.sign(torch.sign(trimmed).sum(dim=0))
        else:
            signs = torch.sign(trimmed.sum(dim=0))
        signs = torch.where(signs == 0, majority_sign, signs).detach()
        agree = torch.where(signs.unsqueeze(0) > 0, trimmed > 0, trimmed < 0)
        selected = trimmed * agree
        if self.config.ties_merge_type == "mean":
            merged = selected.sum(dim=0) / (selected != 0).sum(dim=0).clamp_min(1)
        else:
            merged = selected.sum(dim=0)
        diagnostics = {
            "retained_density": trim_mask.float().mean().detach(),
            "sign_agreement_density": agree.float().mean().detach(),
            "sign_conflict_rate": (trim_mask & ~agree).float().mean().detach(),
        }
        return merged * self.config.ties_merge_scale, diagnostics

    def _merge_ties_group(
        self, repo_loras: LoRADict
    ) -> tuple[LoRADict, list[Tensor], list[dict[str, Tensor]]]:
        num_chunks = next(iter(repo_loras.values()))["A"].shape[0]
        if num_chunks == 1:
            return self._single_chunk_fixed_rank(repo_loras), [], []
        thresholds = self._global_ties_thresholds(repo_loras)
        majority = self._global_ties_majority(repo_loras, thresholds)
        out: LoRADict = {}
        errors = []
        layer_diagnostics = []
        for module in sorted(repo_loras):
            A, B = repo_loras[module]["A"], repo_loras[module]["B"]
            merged_A, merged_B = [], []
            for layer in range(A.shape[1]):
                def merge_and_factor(layer_A: Tensor, layer_B: Tensor):
                    updates = dense_lora_update(
                        layer_A.float(), layer_B.float()
                    )
                    merged_dense, diagnostics = self._merge_dense_layer(
                        updates, thresholds, majority
                    )
                    out_A, out_B, error = truncated_svd_lora(
                        merged_dense, self.config.output_rank, self.config
                    )
                    return (
                        out_A,
                        out_B,
                        error,
                        diagnostics["retained_density"],
                        diagnostics["sign_agreement_density"],
                        diagnostics["sign_conflict_rate"],
                    )

                layer_A = A[:, layer]
                layer_B = B[:, layer]
                if torch.is_grad_enabled() and (
                    layer_A.requires_grad or layer_B.requires_grad
                ):
                    values = checkpoint(
                        merge_and_factor,
                        layer_A,
                        layer_B,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                else:
                    values = merge_and_factor(layer_A, layer_B)
                out_A, out_B, error = values[:3]
                diagnostics = {
                    "retained_density": values[3],
                    "sign_agreement_density": values[4],
                    "sign_conflict_rate": values[5],
                }
                layer_diagnostics.append(diagnostics)
                out_A = out_A.to(A.dtype)
                out_B = out_B.to(B.dtype)
                out_A, out_B = _pad_rank(out_A, out_B, self.config.output_rank)
                merged_A.append(out_A)
                merged_B.append(out_B)
                errors.append(error)
            out[module] = {
                "A": torch.stack(merged_A).unsqueeze(0),
                "B": torch.stack(merged_B).unsqueeze(0),
            }
        return out, errors, layer_diagnostics

    def _merge_ties(self, chunk_loras: LoRADict, n_ctx_chunks: Tensor) -> LoRADict:
        grouped: dict[str, dict[str, list[Tensor]]] = {
            module: {"A": [], "B": []} for module in chunk_loras
        }
        all_errors = []
        all_diagnostics: list[dict[str, Tensor]] = []
        for group in _group_slices(n_ctx_chunks):
            repo_loras = {
                module: {key: value[key][group] for key in ("A", "B")}
                for module, value in chunk_loras.items()
            }
            merged, errors, diagnostics = self._merge_ties_group(repo_loras)
            all_errors.extend(errors)
            all_diagnostics.extend(diagnostics)
            for module in grouped:
                grouped[module]["A"].append(merged[module]["A"][0])
                grouped[module]["B"].append(merged[module]["B"][0])
        result = {
            module: {
                "A": torch.stack(factors["A"]),
                "B": torch.stack(factors["B"]),
            }
            for module, factors in grouped.items()
        }
        self.last_diagnostics = {
            "method": self.config.method.value,
            "svd_reconstruction_error": (
                torch.stack(all_errors).mean() if all_errors else 0.0
            ),
        }
        if all_diagnostics:
            for key in all_diagnostics[0]:
                self.last_diagnostics[key] = torch.stack(
                    [diagnostics[key] for diagnostics in all_diagnostics]
                ).mean()
        return result

    @torch.no_grad()
    def _bounded_global_ties_thresholds(self, repo_loras: LoRADict) -> Tensor:
        """Compute exact per-chunk global trim thresholds with bounded memory.

        The reference convention retains the largest ``int(d * keep) + 1``
        coordinates (plus any boundary ties).  Keeping only those candidates
        while visiting one dense layer at a time produces the same threshold as
        concatenating all model updates, without constructing that full vector.
        """

        num_chunks = next(iter(repo_loras.values()))["A"].shape[0]
        device = next(iter(repo_loras.values()))["A"].device
        if self.config.ties_keep_fraction >= 1:
            return torch.full(
                (num_chunks,), float("-inf"), device=device, dtype=torch.float32
            )

        total_parameters = 0
        for factors in repo_loras.values():
            A, B = factors["A"], factors["B"]
            total_parameters += A.shape[1] * A.shape[-1] * B.shape[-1]
        candidate_count = min(
            total_parameters,
            int(total_parameters * self.config.ties_keep_fraction) + 1,
        )

        thresholds = []
        for chunk in range(num_chunks):
            candidates = None
            for module in sorted(repo_loras):
                A = repo_loras[module]["A"][chunk]
                B = repo_loras[module]["B"][chunk]
                for layer in range(A.shape[0]):
                    values = dense_lora_update(
                        A[layer].float(), B[layer].float()
                    ).abs().reshape(-1)
                    if values.numel() > candidate_count:
                        values = torch.topk(
                            values, candidate_count, sorted=False
                        ).values
                    candidates = (
                        values if candidates is None else torch.cat((candidates, values))
                    )
                    if candidates.numel() > candidate_count:
                        candidates = torch.topk(
                            candidates, candidate_count, sorted=False
                        ).values
            if candidates is None:
                raise ValueError("Cannot compute a TIES threshold without LoRA targets")
            thresholds.append(candidates.min())
        return torch.stack(thresholds)

    def _merge_streaming_ties_exact_layer(
        self,
        layer_A: Tensor,
        layer_B: Tensor,
        thresholds: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Merge one layer without materializing ``[chunks, out, in]``."""

        num_chunks = layer_A.shape[0]
        if num_chunks > torch.iinfo(torch.int16).max:
            raise ValueError(
                "streaming_ties_exact currently supports at most 32767 chunks "
                "per repository"
            )
        shape = (layer_B.shape[-1], layer_A.shape[-1])
        positive_sum = torch.zeros(shape, device=layer_A.device, dtype=torch.float32)
        negative_sum = torch.zeros_like(positive_sum)
        sign_votes = torch.zeros(shape, device=layer_A.device, dtype=torch.int16)
        retained_entries = positive_sum.new_zeros(())

        for chunk in range(num_chunks):
            update = dense_lora_update(
                layer_A[chunk].float(), layer_B[chunk].float()
            )
            trim_mask = update.abs() >= thresholds[chunk]
            trimmed = update * trim_mask
            with torch.no_grad():
                sign_votes.add_(torch.sign(trimmed).to(torch.int16))
                retained_entries.add_(trim_mask.float().sum())
            positive_sum = positive_sum + torch.clamp_min(trimmed, 0)
            negative_sum = negative_sum + torch.clamp_max(trimmed, 0)

        final_sign = torch.sign(sign_votes).detach()
        # Keep exact ties at zero in the forward pass, but avoid a dead
        # all-zero initialization by passing the signed-sum gradient through
        # that nondifferentiable branch.
        signed_sum = positive_sum + negative_sum
        tie_zero_with_gradient = signed_sum - signed_sum.detach()
        merged = torch.where(
            final_sign > 0,
            positive_sum,
            torch.where(
                final_sign < 0, negative_sum, tie_zero_with_gradient
            ),
        )
        merged = merged * (self.config.ties_merge_scale / num_chunks)

        # Revisit one temporary dense update at a time for diagnostics rather
        # than retaining positive/negative count matrices throughout the merge.
        agreeing_entries = retained_entries.new_zeros(())
        conflicting_entries = retained_entries.new_zeros(())
        with torch.no_grad():
            for chunk in range(num_chunks):
                update = dense_lora_update(
                    layer_A[chunk].detach().float(),
                    layer_B[chunk].detach().float(),
                )
                trim_mask = update.abs() >= thresholds[chunk]
                trimmed = update * trim_mask
                agree = torch.where(
                    final_sign > 0,
                    trimmed > 0,
                    torch.where(
                        final_sign < 0,
                        trimmed < 0,
                        torch.zeros_like(trim_mask),
                    ),
                )
                agreeing_entries.add_(agree.float().sum())
                conflicting_entries.add_((trim_mask & ~agree).float().sum())

        total_entries = num_chunks * math.prod(shape)
        diagnostics = {
            "retained_density": (retained_entries / total_entries).detach(),
            "sign_agreement_density": (
                agreeing_entries / total_entries
            ).detach(),
            "sign_conflict_rate": (
                conflicting_entries / total_entries
            ).detach(),
            "final_sign_tie_density": (
                final_sign == 0
            ).float().mean().detach(),
        }
        return merged, diagnostics

    def _merge_streaming_ties_exact_group(
        self, repo_loras: LoRADict
    ) -> tuple[LoRADict, list[Tensor], list[dict[str, Tensor]]]:
        num_chunks = next(iter(repo_loras.values()))["A"].shape[0]
        if num_chunks == 1:
            return self._single_chunk_fixed_rank(repo_loras), [], []

        thresholds = self._bounded_global_ties_thresholds(repo_loras)
        out: LoRADict = {}
        errors: list[Tensor] = []
        layer_diagnostics: list[dict[str, Tensor]] = []
        for module in sorted(repo_loras):
            A, B = repo_loras[module]["A"], repo_loras[module]["B"]
            merged_A, merged_B = [], []
            for layer in range(A.shape[1]):
                def merge_and_factor(layer_A: Tensor, layer_B: Tensor):
                    merged_dense, diagnostics = (
                        self._merge_streaming_ties_exact_layer(
                            layer_A, layer_B, thresholds
                        )
                    )
                    out_A, out_B, error = truncated_svd_lora(
                        merged_dense,
                        self.config.output_rank,
                        self.config,
                        zero_gradient_basis=True,
                    )
                    return (
                        out_A,
                        out_B,
                        error,
                        diagnostics["retained_density"],
                        diagnostics["sign_agreement_density"],
                        diagnostics["sign_conflict_rate"],
                        diagnostics["final_sign_tie_density"],
                    )

                layer_A = A[:, layer]
                layer_B = B[:, layer]
                if torch.is_grad_enabled() and (
                    layer_A.requires_grad or layer_B.requires_grad
                ):
                    values = checkpoint(
                        merge_and_factor,
                        layer_A,
                        layer_B,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                else:
                    values = merge_and_factor(layer_A, layer_B)
                out_A, out_B, error = values[:3]
                layer_diagnostics.append(
                    {
                        "retained_density": values[3],
                        "sign_agreement_density": values[4],
                        "sign_conflict_rate": values[5],
                        "final_sign_tie_density": values[6],
                    }
                )
                out_A = out_A.to(A.dtype)
                out_B = out_B.to(B.dtype)
                out_A, out_B = _pad_rank(
                    out_A, out_B, self.config.output_rank
                )
                merged_A.append(out_A)
                merged_B.append(out_B)
                errors.append(error)
            out[module] = {
                "A": torch.stack(merged_A).unsqueeze(0),
                "B": torch.stack(merged_B).unsqueeze(0),
            }
        return out, errors, layer_diagnostics

    def _merge_streaming_ties_exact(
        self, chunk_loras: LoRADict, n_ctx_chunks: Tensor
    ) -> LoRADict:
        grouped: dict[str, dict[str, list[Tensor]]] = {
            module: {"A": [], "B": []} for module in chunk_loras
        }
        all_errors: list[Tensor] = []
        all_diagnostics: list[dict[str, Tensor]] = []
        for group in _group_slices(n_ctx_chunks):
            repo_loras = {
                module: {key: value[key][group] for key in ("A", "B")}
                for module, value in chunk_loras.items()
            }
            merged, errors, diagnostics = (
                self._merge_streaming_ties_exact_group(repo_loras)
            )
            all_errors.extend(errors)
            all_diagnostics.extend(diagnostics)
            for module in grouped:
                grouped[module]["A"].append(merged[module]["A"][0])
                grouped[module]["B"].append(merged[module]["B"][0])

        result = {
            module: {
                "A": torch.stack(factors["A"]),
                "B": torch.stack(factors["B"]),
            }
            for module, factors in grouped.items()
        }
        self.last_diagnostics = {
            "method": self.config.method.value,
            "streaming_algorithm": "exact_final_sign",
            "streaming_normalization": "total_chunks",
            "svd_reconstruction_error": (
                torch.stack(all_errors).mean() if all_errors else 0.0
            ),
        }
        if all_diagnostics:
            for key in all_diagnostics[0]:
                self.last_diagnostics[key] = torch.stack(
                    [diagnostics[key] for diagnostics in all_diagnostics]
                ).mean()
        return result

    def _knots_align_layer(self, A: Tensor, B: Tensor) -> tuple[Tensor, list[Tensor]]:
        # A: [chunks, rank, input], B: [chunks, rank, output].  For column-wise
        # KnOTS, W_cat W_cat^T is represented exactly by concatenating
        # B_i R_i^T, where A_i^T = Q_i R_i.
        # The supplied KnOTS code computes the shared SVD in float64, then
        # stores U/s/V in float32 before merging the aligned representations.
        A = A.float()
        B = B.float()
        left_parts = []
        for chunk in range(A.shape[0]):
            _, R = torch.linalg.qr(A[chunk].mT, mode="reduced")
            left_parts.append(B[chunk].mT @ R.mT)
        low_rank_left = torch.cat(left_parts, dim=1)
        U, singular_values, _ = torch.linalg.svd(
            low_rank_left.double(), full_matrices=False
        )
        keep = singular_values > self.config.knots_singular_value_epsilon
        U = U[:, keep].float()
        if U.shape[1] == 0:
            # Preserve a differentiable zero path for degenerate updates.
            U = low_rank_left[:, :1] * 0
        aligned = [(U.mT @ B[i].mT) @ A[i] for i in range(A.shape[0])]
        return U, aligned

    def _merge_knots_group(self, repo_loras: LoRADict) -> tuple[LoRADict, list[Tensor]]:
        num_chunks = next(iter(repo_loras.values()))["A"].shape[0]
        if num_chunks == 1:
            return self._single_chunk_fixed_rank(repo_loras), []

        records = []
        aligned_by_chunk: list[list[Tensor]] = [[] for _ in range(num_chunks)]
        for module in sorted(repo_loras):
            A, B = repo_loras[module]["A"], repo_loras[module]["B"]
            for layer in range(A.shape[1]):
                U, aligned = self._knots_align_layer(A[:, layer], B[:, layer])
                records.append((module, layer, U, aligned[0].shape))
                for chunk in range(num_chunks):
                    aligned_by_chunk[chunk].append(aligned[chunk].reshape(-1))

        vectors = torch.stack([torch.cat(parts) for parts in aligned_by_chunk])
        merged_vector, diagnostics = ties_merge_vectors(
            vectors,
            self.config.ties_keep_fraction,
            self.config.ties_sign_method,
            self.config.ties_merge_type,
            self.config.ties_merge_scale,
        )

        per_module: dict[str, dict[str, list[Tensor]]] = {
            module: {"A": [], "B": []} for module in repo_loras
        }
        errors = []
        offset = 0
        for module, _layer, U, shape in records:
            size = math.prod(shape)
            merged_sV = merged_vector[offset : offset + size].reshape(shape)
            offset += size
            shared_rank = merged_sV.shape[0]
            if shared_rank <= self.config.output_rank:
                out_A, out_B = _pad_rank(
                    merged_sV, U.mT, self.config.output_rank
                )
                error = merged_sV.new_zeros(())
            else:
                small_A, small_B, error = truncated_svd_lora(
                    merged_sV, self.config.output_rank, self.config
                )
                # merged_sV = small_B.T @ small_A; U @ merged_sV therefore
                # has stored B factor small_B @ U.T.
                out_A = small_A
                out_B = small_B @ U.mT
            per_module[module]["A"].append(out_A)
            per_module[module]["B"].append(out_B)
            errors.append(error)
        if offset != merged_vector.numel():
            raise AssertionError("KnOTS aligned-vector split did not consume all values")

        out = {
            module: {
                "A": torch.stack(factors["A"]).unsqueeze(0),
                "B": torch.stack(factors["B"]).unsqueeze(0),
            }
            for module, factors in per_module.items()
        }
        self.last_diagnostics = {
            "method": self.config.method.value,
            **diagnostics,
            "knots_shared_rank": torch.tensor(
                [record[2].shape[1] for record in records],
                dtype=torch.float32,
                device=merged_vector.device,
            ).mean().detach(),
        }
        return out, errors

    def _merge_knots_ties(
        self, chunk_loras: LoRADict, n_ctx_chunks: Tensor
    ) -> LoRADict:
        grouped: dict[str, dict[str, list[Tensor]]] = {
            module: {"A": [], "B": []} for module in chunk_loras
        }
        all_errors = []
        for group in _group_slices(n_ctx_chunks):
            repo_loras = {
                module: {key: value[key][group] for key in ("A", "B")}
                for module, value in chunk_loras.items()
            }
            merged, errors = self._merge_knots_group(repo_loras)
            all_errors.extend(errors)
            for module in grouped:
                grouped[module]["A"].append(merged[module]["A"][0])
                grouped[module]["B"].append(merged[module]["B"][0])
        result = {
            module: {
                "A": torch.stack(factors["A"]),
                "B": torch.stack(factors["B"]),
            }
            for module, factors in grouped.items()
        }
        if all_errors:
            self.last_diagnostics["svd_reconstruction_error"] = torch.stack(
                all_errors
            ).mean()
        return result
