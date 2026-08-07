import copy

import torch

from ctx_to_lora.modeling.aggregator import Perceiver
from ctx_to_lora.modeling.idefics2 import (
    Idefics2PerceiverAttention,
    Idefics2PerceiverConfig,
    Idefics2PerceiverSdpaAttention,
)


def _config():
    return Idefics2PerceiverConfig(
        input_size=16,
        hidden_size=16,
        intermediate_size_factor=4,
        n_latents=3,
        num_blocks=1,
        num_self_attn_per_block=0,
        shared_weights=False,
        n_heads=4,
        head_dim=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
        attn_implementation="eager",
    )


def test_sdpa_perceiver_matches_eager_outputs_and_gradients():
    torch.manual_seed(19)
    eager = Idefics2PerceiverAttention(_config()).double()
    sdpa = Idefics2PerceiverSdpaAttention(_config()).double()
    sdpa.load_state_dict(copy.deepcopy(eager.state_dict()))

    eager_latents = torch.randn(2, 3, 16, dtype=torch.double, requires_grad=True)
    eager_context = torch.randn(2, 11, 16, dtype=torch.double, requires_grad=True)
    sdpa_latents = eager_latents.detach().clone().requires_grad_(True)
    sdpa_context = eager_context.detach().clone().requires_grad_(True)

    eager_output = eager(eager_latents, eager_context)[0]
    sdpa_output = sdpa(sdpa_latents, sdpa_context)[0]
    torch.testing.assert_close(sdpa_output, eager_output, atol=5e-8, rtol=1e-5)

    eager_output.square().mean().backward()
    sdpa_output.square().mean().backward()
    torch.testing.assert_close(
        sdpa_latents.grad, eager_latents.grad, atol=5e-8, rtol=1e-5
    )
    torch.testing.assert_close(
        sdpa_context.grad, eager_context.grad, atol=5e-8, rtol=1e-5
    )
    for (eager_name, eager_parameter), (sdpa_name, sdpa_parameter) in zip(
        eager.named_parameters(), sdpa.named_parameters()
    ):
        assert eager_name == sdpa_name
        torch.testing.assert_close(
            sdpa_parameter.grad, eager_parameter.grad, atol=5e-8, rtol=1e-5
        )


def test_sdpa_perceiver_builds_through_pretrained_model_dispatch():
    model = Perceiver(
        feature_size=4,
        output_size=8,
        num_layers=3,
        num_modules=1,
        num_extra_modules=0,
        per_rank_gen=True,
        lora_r=2,
        num_latent_factor=1,
        layer_to_layer_ctx_encoder=True,
        n_latent_queries=2,
        num_blocks=1,
        num_self_attn_per_block=0,
        shared_weights=False,
        perceiver_attn_implementation="sdpa",
    )
    model.enable_iterative_mode(True)
    output, _ = model(
        torch.randn(1, 5, 4),
        ctx_position_ids=torch.arange(5).unsqueeze(0),
    )
    assert output.shape == (1, 1, 2, 8)


def test_perceiver_activation_checkpointing_preserves_outputs_and_gradients():
    torch.manual_seed(23)
    common = dict(
        feature_size=4,
        output_size=8,
        num_layers=3,
        num_modules=1,
        num_extra_modules=0,
        per_rank_gen=True,
        lora_r=2,
        num_latent_factor=1,
        layer_to_layer_ctx_encoder=True,
        n_latent_queries=2,
        num_blocks=2,
        num_self_attn_per_block=0,
        shared_weights=False,
        perceiver_attn_implementation="sdpa",
    )
    eager = Perceiver(**common, perceiver_activation_checkpointing=False).double()
    checkpointed = Perceiver(
        **common, perceiver_activation_checkpointing=True
    ).double()
    checkpointed.load_state_dict(copy.deepcopy(eager.state_dict()))
    eager.enable_iterative_mode(True)
    checkpointed.enable_iterative_mode(True)
    eager.train()
    checkpointed.train()

    eager_context = torch.randn(1, 7, 4, dtype=torch.double, requires_grad=True)
    checkpointed_context = eager_context.detach().clone().requires_grad_(True)
    position_ids = torch.arange(7).unsqueeze(0)

    eager_output = eager(eager_context, ctx_position_ids=position_ids)[0]
    checkpointed_output = checkpointed(
        checkpointed_context, ctx_position_ids=position_ids
    )[0]
    torch.testing.assert_close(
        checkpointed_output, eager_output, atol=5e-8, rtol=1e-5
    )

    eager_output.square().mean().backward()
    checkpointed_output.square().mean().backward()
    torch.testing.assert_close(
        checkpointed_context.grad, eager_context.grad, atol=5e-8, rtol=1e-5
    )
    for (eager_name, eager_parameter), (
        checkpointed_name,
        checkpointed_parameter,
    ) in zip(eager.named_parameters(), checkpointed.named_parameters()):
        assert eager_name == checkpointed_name
        torch.testing.assert_close(
            checkpointed_parameter.grad,
            eager_parameter.grad,
            atol=5e-8,
            rtol=1e-5,
        )


def test_chunked_modality_projection_preserves_outputs_and_gradients():
    torch.manual_seed(29)
    common = dict(
        feature_size=4,
        output_size=8,
        num_layers=3,
        num_modules=1,
        num_extra_modules=0,
        per_rank_gen=True,
        lora_r=2,
        num_latent_factor=1,
        layer_to_layer_ctx_encoder=True,
        n_latent_queries=2,
        num_blocks=2,
        num_self_attn_per_block=0,
        shared_weights=False,
        perceiver_attn_implementation="sdpa",
        perceiver_activation_checkpointing=True,
    )
    unchunked = Perceiver(
        **common, perceiver_modality_projection_chunk_size=0
    ).double()
    chunked = Perceiver(
        **common, perceiver_modality_projection_chunk_size=3
    ).double()
    chunked.load_state_dict(copy.deepcopy(unchunked.state_dict()))
    unchunked.enable_iterative_mode(True)
    chunked.enable_iterative_mode(True)
    unchunked.train()
    chunked.train()

    unchunked_context = torch.randn(
        1, 7, 4, dtype=torch.double, requires_grad=True
    )
    chunked_context = unchunked_context.detach().clone().requires_grad_(True)
    position_ids = torch.arange(7).unsqueeze(0)

    unchunked_output = unchunked(
        unchunked_context, ctx_position_ids=position_ids
    )[0]
    chunked_output = chunked(
        chunked_context, ctx_position_ids=position_ids
    )[0]
    torch.testing.assert_close(
        chunked_output, unchunked_output, atol=5e-8, rtol=1e-5
    )

    unchunked_output.square().mean().backward()
    chunked_output.square().mean().backward()
    torch.testing.assert_close(
        chunked_context.grad,
        unchunked_context.grad,
        atol=5e-8,
        rtol=1e-5,
    )
    for (unchunked_name, unchunked_parameter), (
        chunked_name,
        chunked_parameter,
    ) in zip(unchunked.named_parameters(), chunked.named_parameters()):
        assert unchunked_name == chunked_name
        torch.testing.assert_close(
            chunked_parameter.grad,
            unchunked_parameter.grad,
            atol=5e-8,
            rtol=1e-5,
        )
