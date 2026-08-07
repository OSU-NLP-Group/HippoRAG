from types import MethodType, SimpleNamespace

import torch
from torch import nn

from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel


def test_packed_canonical_chunks_are_encoded_one_at_a_time_and_reassembled():
    model = object.__new__(ModulatedPretrainedModel)
    nn.Module.__init__(model)
    model.ctx_encoder_args = SimpleNamespace(ctx_chunk_microbatch_size=1)
    model.user_defined_scaling = 1
    calls = []

    def fake_one(self, ctx_ids, ctx_attn_mask, ctx_position_ids, return_latents, **_):
        calls.append(ctx_ids.clone())
        value = ctx_ids.float().sum().reshape(1, 1, 1, 1)
        loras = {"proj": {"A": value, "B": value + 1}}
        latent = value.reshape(1, 1, 1, 1, 1)
        return (loras, None, latent) if return_latents else (loras, None)

    model._generate_weights_one_context_batch = MethodType(fake_one, model)
    ids = torch.tensor([[10, 11, 12, 20, 21]])
    positions = torch.tensor([[0, 1, 2, 0, 1]])
    loras, _, latents = model.generate_weights(
        ids, ctx_position_ids=positions, return_latents=True
    )

    assert [call.tolist() for call in calls] == [[[10, 11, 12]], [[20, 21]]]
    assert loras["proj"]["A"].flatten().tolist() == [33.0, 41.0]
    assert latents.flatten().tolist() == [33.0, 41.0]


def test_chunk_checkpoint_recomputes_and_preserves_hypernetwork_gradients():
    model = object.__new__(ModulatedPretrainedModel)
    nn.Module.__init__(model)
    model.hypernet = SimpleNamespace(target_modules=("proj",))
    model.weight = nn.Parameter(torch.tensor(2.0))
    calls = []

    def fake_one(self, ctx_ids, ctx_attn_mask, ctx_position_ids, return_latents, **_):
        calls.append(ctx_ids.clone())
        value = ctx_ids.float().sum().reshape(1, 1, 1, 1) * self.weight
        return ({"proj": {"A": value, "B": value + 1}}, None)

    model._generate_weights_one_context_batch = MethodType(fake_one, model)
    result, _ = model._checkpointed_chunk_weights(
        torch.tensor([[2, 3]]),
        None,
        torch.tensor([[0, 1]]),
        return_latents=False,
    )
    result["proj"]["A"].sum().backward()
    assert model.weight.grad.item() == 5.0
    assert len(calls) == 2  # forward plus backward recomputation
