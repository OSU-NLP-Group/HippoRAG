import numpy as np
import torch

from ctx_to_lora.data.packing import (
    block_diagonal_causal_mask,
    validate_packed_qa_isolation,
)
from ctx_to_lora.trainer import (
    causal_lm_ce_loss,
    compact_causal_lm_ce_loss,
    logical_qa_weighted_l1,
    per_qa_mean_loss,
    supervised_causal_lm_targets,
)


def test_answer_token_ce_means_each_logical_qa_before_global_mean():
    # QA 1 supervises targets 2,3; QA 2 target 7; QA 3 targets 10,11,12.
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 2, 3, -100, -100, 7, -100, -100, 10, 11, 12]])
    token_loss = torch.arange(1, 13, dtype=torch.float32, requires_grad=True)
    qa = per_qa_mean_loss(token_loss, labels, position_ids)
    expected = torch.tensor([(2 + 3) / 2, 6.0, (9 + 10 + 11) / 3])
    torch.testing.assert_close(qa, expected)
    torch.testing.assert_close(qa.mean(), expected.mean())


def test_answer_ce_masks_before_fp32_and_matches_dense_forward_backward(monkeypatch):
    torch.manual_seed(23)
    vocab_size = 17
    labels = torch.tensor([[-100, -100, 4, 6, -100, 3, 8]])
    shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]

    reference_logits = torch.randn(
        1, labels.shape[1], vocab_size, dtype=torch.bfloat16, requires_grad=True
    )
    reference_loss = torch.nn.functional.cross_entropy(
        reference_logits.float().reshape(-1, vocab_size),
        shifted.reshape(-1),
        reduction="none",
    )
    reference_loss.sum().backward()

    actual_logits = reference_logits.detach().clone().requires_grad_(True)
    original_cross_entropy = torch.nn.functional.cross_entropy
    observed: dict[str, tuple[int, ...] | torch.dtype] = {}

    def inspected_cross_entropy(inputs, targets, *args, **kwargs):
        observed["shape"] = tuple(inputs.shape)
        observed["dtype"] = inputs.dtype
        return original_cross_entropy(inputs, targets, *args, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional, "cross_entropy", inspected_cross_entropy
    )
    actual_loss = causal_lm_ce_loss(actual_logits, labels, vocab_size)
    actual_loss.sum().backward()

    assert observed["shape"] == (int((shifted != -100).sum()), vocab_size)
    assert observed["dtype"] == torch.float32
    torch.testing.assert_close(actual_loss, reference_loss)
    torch.testing.assert_close(actual_logits.grad, reference_logits.grad)


def test_compact_answer_logits_match_dense_causal_ce_forward_backward():
    torch.manual_seed(29)
    vocab_size = 19
    labels = torch.tensor([[-100, -100, 4, 6, -100, 3, 8]])
    logits = torch.randn(
        1, labels.shape[1], vocab_size, dtype=torch.bfloat16, requires_grad=True
    )

    dense_token_loss = causal_lm_ce_loss(logits, labels, vocab_size)
    dense_loss = per_qa_mean_loss(
        dense_token_loss,
        labels,
        torch.arange(labels.shape[1]).unsqueeze(0),
    )[0]
    dense_loss.backward()
    dense_grad = logits.grad.detach().clone()

    logit_positions, supervised_labels = supervised_causal_lm_targets(labels)
    compact_logits = (
        logits.detach()[:, logit_positions, :].clone().requires_grad_(True)
    )
    compact_loss = compact_causal_lm_ce_loss(
        compact_logits,
        supervised_labels,
        vocab_size,
    )
    compact_loss.backward()

    torch.testing.assert_close(compact_loss, dense_loss)
    torch.testing.assert_close(
        compact_logits.grad,
        dense_grad[:, logit_positions, :],
    )
    assert logit_positions.tolist() == [1, 2, 4, 5]
    assert supervised_labels.tolist() == [4, 6, 3, 8]


def test_l1_is_weighted_by_group_logical_qa_multiplicity():
    a = torch.tensor([1.0, 3.0], requires_grad=True).reshape(2, 1, 1, 1)
    b = torch.tensor([2.0, 4.0], requires_grad=True).reshape(2, 1, 1, 1)
    loras = {"proj": {"A": a, "B": b}}
    numerator = logical_qa_weighted_l1(loras, torch.tensor([1, 2]))
    expected = (1 * (1 + 2)) + (2 * (3 + 4))
    torch.testing.assert_close(numerator, torch.tensor(float(expected)))


def test_grouped_and_separate_ce_l1_gradients_match():
    scale_grouped = torch.tensor(2.0, requires_grad=True)
    qa_base = torch.tensor([1.0, 4.0, 7.0])
    grouped_loras = {
        "proj": {
            "A": (scale_grouped * torch.tensor([1.0, 3.0])).reshape(2, 1, 1, 1),
            "B": (scale_grouped * torch.tensor([2.0, 4.0])).reshape(2, 1, 1, 1),
        }
    }
    grouped = (qa_base * scale_grouped).sum() / 3
    grouped = grouped + 0.1 * logical_qa_weighted_l1(
        grouped_loras, torch.tensor([1, 2])
    ) / 3
    grouped.backward()

    scale_separate = torch.tensor(2.0, requires_grad=True)
    separate = (qa_base * scale_separate).sum() / 3
    regularizers = torch.stack(
        [
            scale_separate * (1 + 2),
            scale_separate * (3 + 4),
            scale_separate * (3 + 4),
        ]
    )
    separate = separate + 0.1 * regularizers.sum() / 3
    separate.backward()
    torch.testing.assert_close(grouped, separate)
    torch.testing.assert_close(scale_grouped.grad, scale_separate.grad)


def test_accumulation_and_two_rank_reductions_use_one_global_denominator():
    values = torch.tensor([1.0, 4.0, 7.0, 9.0])
    direct = values.mean()
    accumulation = (values[:1].sum() + values[1:].sum()) / 4
    two_rank = (values[:3].sum() + values[3:].sum()) / (3 + 1)
    torch.testing.assert_close(accumulation, direct)
    torch.testing.assert_close(two_rank, direct)


def test_block_diagonal_mask_matches_separate_causal_attention():
    sequence_ids = np.asarray([0, 0, 0, 1, 1])
    position_ids = np.asarray([0, 1, 2, 0, 1])
    labels = np.asarray([-100, 1, 2, -100, 4])
    validate_packed_qa_isolation(position_ids, sequence_ids, labels)
    mask = torch.tensor(block_diagonal_causal_mask(sequence_ids))
    values = torch.arange(5, dtype=torch.float32)
    packed = torch.stack([values[row].mean() for row in mask])
    separate = torch.cat(
        [
            torch.stack([values[:3][: index + 1].mean() for index in range(3)]),
            torch.stack([values[3:][: index + 1].mean() for index in range(2)]),
        ]
    )
    torch.testing.assert_close(packed, separate)
