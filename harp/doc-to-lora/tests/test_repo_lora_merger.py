import importlib.util
import os
from pathlib import Path

import pytest
import torch

from ctx_to_lora.modeling.lora_merger import combine_lora
from ctx_to_lora.modeling.repo_lora_merger import (
    RepoMergeMethod,
    RepositoryLoRAMerger,
    RepositoryMergerConfig,
    dense_lora_update,
    exact_streaming_ties_merge,
    ties_merge_vectors,
)


def factors(chunks=3, layers=2, rank=2, d_in=5, d_out=4, seed=11):
    generator = torch.Generator().manual_seed(seed)
    return {
        "proj": {
            "A": torch.randn(chunks, layers, rank, d_in, generator=generator),
            "B": torch.randn(chunks, layers, rank, d_out, generator=generator),
        }
    }


def dense_from_result(result, module="proj"):
    return dense_lora_update(result[module]["A"], result[module]["B"])


def load_knots_merging_functions():
    configured = os.environ.get("KNOTS_REFERENCE_PATH")
    if not configured:
        pytest.skip("set KNOTS_REFERENCE_PATH to run reference-equivalence tests")
    path = Path(configured)
    if not path.is_file():
        pytest.skip(f"KnOTS reference file not found: {path}")
    spec = importlib.util.spec_from_file_location("knots_reference_merging", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_concat_is_the_original_doc_to_lora_operation():
    chunk_loras = factors(chunks=3)
    counts = torch.tensor([1, 2])
    bias = factors(chunks=1)["proj"]
    bias = {"proj": {key: value[0] for key, value in bias.items()}}
    expected = combine_lora(chunk_loras, counts, lora_bias=bias)
    merger = RepositoryLoRAMerger(
        RepositoryMergerConfig(method=RepoMergeMethod.CONCAT)
    )
    actual = merger(chunk_loras, counts, lora_bias=bias)
    for key in ("A", "B"):
        torch.testing.assert_close(actual["proj"][key], expected["proj"][key])


def test_concat_dense_update_is_exact_sum_for_required_k_and_dtypes():
    for dtype in (torch.float32, torch.bfloat16):
        for chunks in (1, 2, 4, 8):
            chunk_loras = factors(chunks=chunks, layers=1, rank=8).copy()
            chunk_loras = {
                module: {key: value.to(dtype) for key, value in values.items()}
                for module, values in chunk_loras.items()
            }
            actual = RepositoryLoRAMerger(
                RepositoryMergerConfig(method=RepoMergeMethod.CONCAT)
            )(chunk_loras, torch.tensor([chunks]))
            dense = dense_from_result(actual).float()[0, 0]
            expected = dense_lora_update(
                chunk_loras["proj"]["A"][:, 0].float(),
                chunk_loras["proj"]["B"][:, 0].float(),
            ).sum(dim=0)
            tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
            torch.testing.assert_close(dense, expected, atol=tolerance, rtol=tolerance)


def test_ties_vector_math_matches_supplied_knots_reference():
    reference = load_knots_merging_functions()
    vectors = torch.tensor(
        [
            [1.0, -4.0, 0.2, 8.0, -3.0, 0.5],
            [2.0, 3.0, -0.1, -7.0, -5.0, 0.4],
            [-1.0, 5.0, 0.3, 6.0, 4.0, -0.6],
        ]
    )
    expected, _, _ = reference.ties_merging(
        list(vectors), topK=50, merging_type="mean"
    )
    actual, _ = ties_merge_vectors(vectors, 0.5, "sum", "mean", 1.0)
    torch.testing.assert_close(actual, expected)


def test_direct_ties_merges_dense_updates_and_is_order_invariant():
    chunk_loras = factors(chunks=3, layers=1, d_in=5, d_out=4)
    config = RepositoryMergerConfig(
        method=RepoMergeMethod.TIES,
        output_rank=4,
        ties_keep_fraction=0.5,
        svd_exact_max_dim=32,
    )
    merger = RepositoryLoRAMerger(config)
    counts = torch.tensor([3])
    actual = merger(chunk_loras, counts)

    vectors = dense_lora_update(
        chunk_loras["proj"]["A"][:, 0], chunk_loras["proj"]["B"][:, 0]
    ).reshape(3, -1)
    expected, _ = ties_merge_vectors(vectors, 0.5, "sum", "mean", 1.0)
    torch.testing.assert_close(
        dense_from_result(actual)[0, 0], expected.reshape(4, 5), atol=2e-5, rtol=2e-5
    )

    permutation = torch.tensor([2, 0, 1])
    permuted = {
        "proj": {
            key: value[permutation]
            for key, value in chunk_loras["proj"].items()
        }
    }
    reordered = merger(permuted, counts)
    torch.testing.assert_close(
        dense_from_result(actual), dense_from_result(reordered), atol=2e-5, rtol=2e-5
    )


def test_exact_streaming_ties_matches_the_final_sign_example():
    updates = torch.tensor([[2.0], [-4.0], [1.0], [-5.0], [-6.0]])
    actual, diagnostics = exact_streaming_ties_merge(updates)
    # Three negative votes win. The selected negative sum is -15 and the
    # exact streaming normalization divides by all five chunk LoRAs.
    torch.testing.assert_close(actual, torch.tensor([-3.0]))
    torch.testing.assert_close(
        diagnostics["sign_agreement_density"], torch.tensor(3 / 5)
    )
    torch.testing.assert_close(
        diagnostics["sign_conflict_rate"], torch.tensor(2 / 5)
    )


def test_exact_streaming_ties_matches_explicit_batch_math_and_is_order_invariant():
    updates = torch.tensor(
        [
            [[2.0, -1.0, 8.0], [-4.0, 3.0, 1.0]],
            [[-4.0, -2.0, -7.0], [-5.0, 4.0, 2.0]],
            [[1.0, 5.0, 6.0], [-6.0, -8.0, 3.0]],
        ]
    )
    thresholds = torch.tensor([1.5, 3.5, 2.5])
    trim_mask = updates.abs() >= thresholds[:, None, None]
    trimmed = updates * trim_mask
    final_sign = torch.sign(torch.sign(trimmed).sum(dim=0))
    selected = torch.where(
        final_sign > 0,
        torch.clamp_min(trimmed, 0),
        torch.where(
            final_sign < 0,
            torch.clamp_max(trimmed, 0),
            torch.zeros_like(trimmed),
        ),
    )
    expected = selected.sum(dim=0) / updates.shape[0]

    actual, _ = exact_streaming_ties_merge(updates, thresholds)
    torch.testing.assert_close(actual, expected)

    permutation = torch.tensor([2, 0, 1])
    reordered, _ = exact_streaming_ties_merge(
        updates[permutation], thresholds[permutation]
    )
    torch.testing.assert_close(reordered, actual)


def test_exact_streaming_tie_is_forward_zero_with_nonzero_surrogate_gradient():
    updates = torch.tensor([[2.0], [-2.0]], requires_grad=True)
    merged, diagnostics = exact_streaming_ties_merge(updates)
    torch.testing.assert_close(merged, torch.zeros_like(merged))
    torch.testing.assert_close(
        diagnostics["final_sign_tie_density"], torch.tensor(1.0)
    )

    merged.sum().backward()
    torch.testing.assert_close(updates.grad, torch.full_like(updates, 0.5))


def test_exact_streaming_factor_path_matches_dense_reference_and_backpropagates():
    chunk_loras = factors(chunks=3, layers=1, d_in=5, d_out=4, seed=71)
    for value in chunk_loras["proj"].values():
        value.requires_grad_()
    config = RepositoryMergerConfig(
        method=RepoMergeMethod.STREAMING_TIES_EXACT,
        output_rank=4,
        ties_keep_fraction=1.0,
        ties_sign_method="sum_of_signs",
        ties_merge_type="mean",
        svd_exact_max_dim=32,
    )
    merger = RepositoryLoRAMerger(config)
    actual = merger(chunk_loras, torch.tensor([3]))

    updates = dense_lora_update(
        chunk_loras["proj"]["A"][:, 0],
        chunk_loras["proj"]["B"][:, 0],
    )
    expected, _ = exact_streaming_ties_merge(updates)
    torch.testing.assert_close(
        dense_from_result(actual)[0, 0], expected, atol=2e-5, rtol=2e-5
    )
    assert merger.last_diagnostics["streaming_algorithm"] == "exact_final_sign"
    assert merger.last_diagnostics["streaming_normalization"] == "total_chunks"

    dense_from_result(actual).square().mean().backward()
    for value in chunk_loras["proj"].values():
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_exact_streaming_zero_svd_has_an_escape_gradient():
    chunk_loras = factors(chunks=2, layers=1, d_in=5, d_out=4, seed=75)
    with torch.no_grad():
        chunk_loras["proj"]["B"].zero_()
    for value in chunk_loras["proj"].values():
        value.requires_grad_()
    merger = RepositoryLoRAMerger(
        RepositoryMergerConfig(
            method=RepoMergeMethod.STREAMING_TIES_EXACT,
            output_rank=4,
            ties_keep_fraction=1.0,
            ties_sign_method="sum_of_signs",
            svd_exact_max_dim=32,
        )
    )
    result = merger(chunk_loras, torch.tensor([2]))
    dense = dense_from_result(result)
    torch.testing.assert_close(dense, torch.zeros_like(dense))

    dense.sum().backward()
    gradient = chunk_loras["proj"]["B"].grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_exact_streaming_bounded_threshold_matches_reference_global_threshold():
    chunk_loras = {
        **factors(chunks=3, layers=2, d_in=5, d_out=4, seed=81),
        "other_proj": factors(
            chunks=3, layers=2, d_in=3, d_out=2, seed=82
        )["proj"],
    }
    merger = RepositoryLoRAMerger(
        RepositoryMergerConfig(
            method=RepoMergeMethod.STREAMING_TIES_EXACT,
            output_rank=4,
            ties_keep_fraction=0.2,
            ties_sign_method="sum_of_signs",
        )
    )
    expected = merger._global_ties_thresholds(chunk_loras)
    actual = merger._bounded_global_ties_thresholds(chunk_loras)
    torch.testing.assert_close(actual, expected)


def test_exact_streaming_ties_rejects_ambiguous_legacy_settings():
    for kwargs, expected_message in (
        ({"ties_sign_method": "sum"}, "sum_of_signs"),
        (
            {"ties_sign_method": "sum_of_signs", "ties_merge_type": "sum"},
            "ties_merge_type=mean",
        ),
    ):
        try:
            RepositoryMergerConfig(
                method=RepoMergeMethod.STREAMING_TIES_EXACT,
                **kwargs,
            )
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError("Expected exact streaming configuration assertion")


def test_direct_ties_promotes_bfloat16_factorization_and_preserves_gradients():
    chunk_loras = factors(chunks=2, layers=1, d_in=9, d_out=7)
    chunk_loras = {
        module: {
            key: value.to(torch.bfloat16).requires_grad_()
            for key, value in values.items()
        }
        for module, values in chunk_loras.items()
    }
    config = RepositoryMergerConfig(
        method=RepoMergeMethod.TIES,
        output_rank=4,
        ties_keep_fraction=0.5,
        svd_exact_max_dim=0,
    )
    actual = RepositoryLoRAMerger(config)(chunk_loras, torch.tensor([2]))
    assert actual["proj"]["A"].dtype == torch.bfloat16
    assert actual["proj"]["B"].dtype == torch.bfloat16
    loss = actual["proj"]["A"].float().square().mean()
    loss = loss + actual["proj"]["B"].float().square().mean()
    loss.backward()
    for values in chunk_loras.values():
        for value in values.values():
            assert value.grad is not None
            assert torch.isfinite(value.grad).all()


def test_direct_ties_drops_null_singular_directions_before_square_root():
    chunk_loras = factors(chunks=2, layers=1, d_in=9, d_out=7)
    with torch.no_grad():
        chunk_loras["proj"]["B"].zero_()
    for values in chunk_loras.values():
        for value in values.values():
            value.requires_grad_()
    config = RepositoryMergerConfig(
        method=RepoMergeMethod.TIES,
        output_rank=4,
        ties_keep_fraction=0.5,
        svd_exact_max_dim=0,
    )
    actual = RepositoryLoRAMerger(config)(chunk_loras, torch.tensor([2]))
    assert torch.count_nonzero(actual["proj"]["A"]) == 0
    assert torch.count_nonzero(actual["proj"]["B"]) == 0
    loss = actual["proj"]["A"].square().sum()
    loss = loss + actual["proj"]["B"].square().sum()
    loss.backward()
    for values in chunk_loras.values():
        for value in values.values():
            assert value.grad is not None
            assert torch.isfinite(value.grad).all()


def test_knots_ties_matches_supplied_column_concat_recipe():
    reference = load_knots_merging_functions()
    chunk_loras = factors(chunks=3, layers=1, d_in=5, d_out=4)
    config = RepositoryMergerConfig(
        method=RepoMergeMethod.KNOTS_TIES,
        output_rank=4,
        ties_keep_fraction=1.0,
        ties_merge_scale=0.5,
        svd_exact_max_dim=32,
    )
    actual = RepositoryLoRAMerger(config)(chunk_loras, torch.tensor([3]))

    updates = dense_lora_update(
        chunk_loras["proj"]["A"][:, 0], chunk_loras["proj"]["B"][:, 0]
    )
    concatenated = torch.cat(list(updates), dim=1)
    U, singular_values, Vh = torch.linalg.svd(
        concatenated.double(), full_matrices=False
    )
    keep = singular_values > config.knots_singular_value_epsilon
    U = U[:, keep].float()
    sV = singular_values[keep].float()[:, None] * Vh[keep].float()
    aligned = list(torch.split(sV, updates.shape[-1], dim=1))
    expected_aligned, _, _ = reference.ties_merging(
        [value.reshape(-1) for value in aligned],
        topK=100,
        merging_type="mean",
    )
    expected = U @ (expected_aligned.reshape_as(aligned[0]) * 0.5)
    torch.testing.assert_close(
        dense_from_result(actual)[0, 0], expected, atol=2e-5, rtol=2e-5
    )


def test_single_chunk_uses_literal_original_doc_to_lora_rank():
    chunk_loras = factors(chunks=1, layers=1, rank=2, d_in=5, d_out=4)
    merger = RepositoryLoRAMerger(
        RepositoryMergerConfig(method=RepoMergeMethod.TIES, output_rank=4)
    )
    actual = merger(chunk_loras, torch.tensor([1]))
    assert actual["proj"]["A"].shape[2] == 4  # original combine_lora allocation
    torch.testing.assert_close(
        dense_from_result(actual), dense_lora_update(
            chunk_loras["proj"]["A"], chunk_loras["proj"]["B"]
        )
    )
    assert merger.last_diagnostics == {
        "method": "ties",
        "svd_reconstruction_error": 0.0,
        "retained_density": 1.0,
        "sign_agreement_density": 1.0,
        "sign_conflict_rate": 0.0,
    }


def test_learned_fusion_is_permutation_invariant_and_backpropagates():
    torch.manual_seed(7)
    config = RepositoryMergerConfig(
        method=RepoMergeMethod.LEARNED_FUSION,
        output_rank=4,
        fusion_num_blocks=2,
        fusion_num_heads=2,
    )
    merger = RepositoryLoRAMerger(config, latent_size=8)
    chunk_loras = factors(chunks=3, layers=1, rank=2, d_in=3, d_out=2)
    latents = torch.randn(3, 1, 1, 2, 8, requires_grad=True)

    def decoder(value):
        value = value[:, :, 0]
        return {"proj": {"A": value[..., :3], "B": value[..., 3:5]}}

    counts = torch.tensor([3])
    actual = merger(
        chunk_loras, counts, chunk_latents=latents, latent_decoder=decoder
    )
    permutation = torch.tensor([2, 0, 1])
    permuted_loras = {
        "proj": {
            key: value[permutation]
            for key, value in chunk_loras["proj"].items()
        }
    }
    reordered = merger(
        permuted_loras,
        counts,
        chunk_latents=latents[permutation],
        latent_decoder=decoder,
    )
    torch.testing.assert_close(
        dense_from_result(actual), dense_from_result(reordered), atol=1e-6, rtol=1e-6
    )
    dense_from_result(actual).square().mean().backward()
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()
    assert latents.grad.abs().sum() > 0


def test_bm25_ties_requires_selection_before_encoding():
    merger = RepositoryLoRAMerger(
        RepositoryMergerConfig(
            method=RepoMergeMethod.BM25_TOPK_TIES,
            output_rank=4,
            retrieval_top_k=2,
        )
    )
    try:
        merger(factors(chunks=3), torch.tensor([3]))
    except ValueError as exc:
        assert "before context encoding" in str(exc)
    else:
        raise AssertionError("Expected pre-encoding BM25 selection assertion")


def test_ties_and_knots_have_finite_chunk_gradients_for_multi_chunk_groups():
    for method in (RepoMergeMethod.TIES, RepoMergeMethod.KNOTS_TIES):
        for num_chunks in (2, 4):
            chunk_loras = factors(
                chunks=num_chunks,
                layers=1,
                rank=2,
                d_in=5,
                d_out=4,
                seed=30 + num_chunks,
            )
            for value in chunk_loras["proj"].values():
                value.requires_grad_()
            merger = RepositoryLoRAMerger(
                RepositoryMergerConfig(
                    method=method,
                    output_rank=4,
                    ties_keep_fraction=1.0,
                    svd_exact_max_dim=32,
                )
            )
            result = merger(chunk_loras, torch.tensor([num_chunks]))
            dense_from_result(result).square().mean().backward()
            for value in chunk_loras["proj"].values():
                assert value.grad is not None
                assert torch.isfinite(value.grad).all()
                per_chunk = value.grad.flatten(1).abs().sum(dim=1)
                assert torch.all(per_chunk > 0)


def test_learned_fusion_state_dict_round_trip_reproduces_output():
    torch.manual_seed(19)
    config = RepositoryMergerConfig(
        method=RepoMergeMethod.LEARNED_FUSION,
        output_rank=4,
        fusion_num_blocks=1,
        fusion_num_heads=2,
    )
    first = RepositoryLoRAMerger(config, latent_size=8).eval()
    second = RepositoryLoRAMerger(config, latent_size=8).eval()
    second.load_state_dict(first.state_dict())
    chunk_loras = factors(chunks=2, layers=1, rank=2, d_in=3, d_out=2)
    latents = torch.randn(2, 1, 1, 2, 8)

    def decoder(value):
        value = value[:, :, 0]
        return {"proj": {"A": value[..., :3], "B": value[..., 3:5]}}

    args = (chunk_loras, torch.tensor([2]))
    kwargs = {"chunk_latents": latents, "latent_decoder": decoder}
    expected = first(*args, **kwargs)
    actual = second(*args, **kwargs)
    for key in ("A", "B"):
        torch.testing.assert_close(actual["proj"][key], expected["proj"][key])
