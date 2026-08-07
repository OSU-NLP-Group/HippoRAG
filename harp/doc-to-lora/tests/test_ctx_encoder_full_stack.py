from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask

from ctx_to_lora.modeling.ctx_encoder import (
    PerLayerActivations,
    _slice_flex_block_mask_queries,
    context_query_chunked_flex_attention_forward,
    enable_token_chunked_context_mlps,
)


class _AddLayer(nn.Module):
    def __init__(self, value: float, calls: list[float]):
        super().__init__()
        self.value = value
        self.calls = calls

    def forward(self, hidden):
        self.calls.append(self.value)
        return hidden + self.value


class _FakeContextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[float] = []
        self.layers = nn.ModuleList(
            [_AddLayer(float(index), self.calls) for index in range(1, 5)]
        )
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.output_hidden_states = None

    def forward(self, input_ids, output_hidden_states=True, **_kwargs):
        self.output_hidden_states = output_hidden_states
        hidden_states = [input_ids.float().unsqueeze(-1)]
        for layer in self.layers:
            hidden_states.append(layer(hidden_states[-1]))
        return SimpleNamespace(
            hidden_states=tuple(hidden_states) if output_hidden_states else None
        )


def test_per_layer_activations_executes_full_stack_and_returns_block_inputs():
    base = _FakeContextModel()
    encoder = PerLayerActivations(
        base,
        SimpleNamespace(keep_lm_head=False, ctx_encoder_last_layer=None),
    )
    actual = encoder(input_ids=torch.tensor([[3, 5]]))

    assert base.calls == [1.0, 2.0, 3.0, 4.0]
    assert actual.shape == (1, 4, 2, 1)
    # Inputs to blocks 0..3 are embedding, then outputs of blocks 0..2.
    expected = torch.tensor([[[[3.0], [5.0]], [[4.0], [6.0]], [[6.0], [8.0]], [[9.0], [11.0]]]])
    torch.testing.assert_close(actual, expected)


def test_per_layer_activations_retains_only_selected_block_inputs():
    base = _FakeContextModel()
    encoder = PerLayerActivations(
        base,
        SimpleNamespace(keep_lm_head=False, ctx_encoder_last_layer=None),
    )
    encoder.select_layer_inputs([0, 2])
    actual = encoder(input_ids=torch.tensor([[3, 5]]))

    assert base.calls == [1.0, 2.0, 3.0, 4.0]
    assert base.output_hidden_states is False
    assert actual.shape == (1, 2, 2, 1)
    expected = torch.tensor([[[[3.0], [5.0]], [[6.0], [8.0]]]])
    torch.testing.assert_close(actual, expected)


def test_per_layer_activations_can_offload_selected_inputs_without_stacking():
    base = _FakeContextModel()
    encoder = PerLayerActivations(
        base,
        SimpleNamespace(
            keep_lm_head=False,
            ctx_encoder_last_layer=None,
            offload_ctx_layer_inputs_to_cpu=True,
        ),
    )
    encoder.select_layer_inputs([0, 2])
    actual = encoder(input_ids=torch.tensor([[3, 5]]))

    assert isinstance(actual, tuple)
    assert len(actual) == 2
    assert all(value.device.type == "cpu" for value in actual)
    torch.testing.assert_close(actual[0], torch.tensor([[[3.0], [5.0]]]))
    torch.testing.assert_close(actual[1], torch.tensor([[[6.0], [8.0]]]))


def test_per_layer_activations_validates_selected_layers():
    base = _FakeContextModel()
    encoder = PerLayerActivations(
        base,
        SimpleNamespace(keep_lm_head=False, ctx_encoder_last_layer=None),
    )

    for indices in ([], [1, 1], [-1], [4]):
        try:
            encoder.select_layer_inputs(indices)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid layer indices: {indices}")


class _TokenWiseMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(5, 11, bias=False)
        self.up = nn.Linear(5, 11, bias=False)
        self.down = nn.Linear(11, 5, bias=False)

    def forward(self, hidden):
        return self.down(torch.nn.functional.silu(self.gate(hidden)) * self.up(hidden))


class _MlpLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _TokenWiseMlp()


class _MlpStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_MlpLayer(), _MlpLayer()])


def test_token_chunked_context_mlp_preserves_outputs_gradients_and_state_keys():
    torch.manual_seed(29)
    baseline = _MlpStack().double()
    chunked = _MlpStack().double()
    chunked.load_state_dict(baseline.state_dict())
    baseline_keys = tuple(baseline.state_dict())
    enable_token_chunked_context_mlps(chunked, chunk_size=3)
    assert tuple(chunked.state_dict()) == baseline_keys

    baseline_input = torch.randn(2, 10, 5, dtype=torch.double, requires_grad=True)
    chunked_input = baseline_input.detach().clone().requires_grad_(True)
    baseline_output = baseline.layers[0].mlp(baseline_input)
    chunked_output = chunked.layers[0].mlp(chunked_input)
    torch.testing.assert_close(chunked_output, baseline_output)

    baseline_output.square().mean().backward()
    chunked_output.square().mean().backward()
    torch.testing.assert_close(chunked_input.grad, baseline_input.grad)
    for baseline_parameter, chunked_parameter in zip(
        baseline.parameters(), chunked.parameters()
    ):
        if baseline_parameter.grad is None:
            assert chunked_parameter.grad is None
        else:
            torch.testing.assert_close(
                chunked_parameter.grad, baseline_parameter.grad
            )


def test_slice_flex_block_mask_preserves_global_query_semantics():
    def mask_mod(batch, head, query, key_value):
        del batch, head
        return (key_value <= query) & (((query + key_value) % 3) != 1)

    mask = create_block_mask(
        mask_mod,
        B=1,
        H=1,
        Q_LEN=16,
        KV_LEN=16,
        BLOCK_SIZE=4,
        device="cpu",
    )
    chunk = _slice_flex_block_mask_queries(mask, start=4, end=14)

    assert chunk.seq_lengths == (10, 16)
    torch.testing.assert_close(
        chunk.to_dense(),
        mask.to_dense()[..., 1:4, :],
    )
    for local_query in range(10):
        for key_value in range(16):
            assert bool(
                chunk.mask_mod(0, 0, local_query, key_value)
            ) == bool(mask.mask_mod(0, 0, local_query + 4, key_value))


def test_chunked_flex_attention_uses_bounded_forward_kernel_tiles():
    mask = create_block_mask(
        lambda _batch, _head, query, key_value: key_value <= query,
        B=1,
        H=1,
        Q_LEN=16,
        KV_LEN=16,
        BLOCK_SIZE=4,
        device="cpu",
    )
    query = torch.randn(1, 2, 16, 4)
    key = torch.randn(1, 1, 16, 4)
    value = torch.randn(1, 1, 16, 4)
    calls = []

    def fake_flex(_module, chunk_query, _key, _value, chunk_mask, **kwargs):
        calls.append((chunk_query.shape[-2], chunk_mask, kwargs))
        output = chunk_query.transpose(1, 2).contiguous()
        lse = chunk_query.new_zeros(
            chunk_query.shape[0],
            chunk_query.shape[1],
            chunk_query.shape[2],
        )
        return output, lse

    module = SimpleNamespace(
        config=SimpleNamespace(ctx_encoder_flex_query_chunk_size=8),
        training=False,
    )
    with (
        torch.no_grad(),
        patch(
            "ctx_to_lora.modeling.ctx_encoder."
            "transformers_flex_attention_forward",
            side_effect=fake_flex,
        ),
    ):
        output, lse = context_query_chunked_flex_attention_forward(
            module,
            query,
            key,
            value,
            mask,
        )

    torch.testing.assert_close(output, query.transpose(1, 2))
    assert lse.shape == (1, 2, 16)
    assert [length for length, _mask, _kwargs in calls] == [8, 8]
    for _length, chunk_mask, kwargs in calls:
        assert chunk_mask.seq_lengths == (8, 16)
        assert kwargs["kernel_options"] == {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "num_warps": 4,
            "num_stages": 1,
        }


def test_short_flex_attention_fallback_uses_bounded_forward_kernel_tiles():
    mask = create_block_mask(
        lambda _batch, _head, query, key_value: key_value <= query,
        B=1,
        H=1,
        Q_LEN=8,
        KV_LEN=8,
        BLOCK_SIZE=4,
        device="cpu",
    )
    query = torch.randn(1, 2, 8, 4)
    key = torch.randn(1, 1, 8, 4)
    value = torch.randn(1, 1, 8, 4)
    module = SimpleNamespace(
        config=SimpleNamespace(ctx_encoder_flex_query_chunk_size=8),
        training=False,
    )
    expected = query.transpose(1, 2).contiguous()

    with patch(
        "ctx_to_lora.modeling.ctx_encoder."
        "transformers_flex_attention_forward",
        return_value=(expected, None),
    ) as mocked_flex:
        actual, lse = context_query_chunked_flex_attention_forward(
            module,
            query,
            key,
            value,
            mask,
        )

    torch.testing.assert_close(actual, expected)
    assert lse is None
    assert mocked_flex.call_count == 1
    assert mocked_flex.call_args.kwargs["kernel_options"] == {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "num_warps": 4,
        "num_stages": 1,
    }
