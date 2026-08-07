import torch

from ctx_to_lora.modeling.aggregator import Perceiver


def make_perceiver():
    return Perceiver(
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
    ).eval()


def test_sequential_layer_perceiver_matches_packed_layer_batch_and_gradients():
    torch.manual_seed(23)
    perceiver = make_perceiver()
    features = torch.randn(1, 3, 5, 4)
    positions = torch.arange(5).unsqueeze(0)

    packed, _ = perceiver(features, ctx_position_ids=positions)
    packed.square().mean().backward()
    packed_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in perceiver.named_parameters()
        if parameter.grad is not None
    }
    perceiver.zero_grad(set_to_none=True)

    perceiver.enable_iterative_mode(True)
    sequential = torch.stack(
        [
            perceiver(features[:, layer], ctx_position_ids=positions)[0]
            for layer in range(features.shape[1])
        ],
        dim=1,
    )
    sequential.square().mean().backward()
    torch.testing.assert_close(sequential, packed, atol=2e-5, rtol=2e-5)
    for name, expected in packed_grads.items():
        torch.testing.assert_close(
            dict(perceiver.named_parameters())[name].grad,
            expected,
            atol=2e-5,
            rtol=2e-5,
        )
