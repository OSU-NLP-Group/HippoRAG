from collections import defaultdict
from collections.abc import Callable

import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import EvalPrediction

LENGTH_BINS = [
    # finegrain bins
    (0, 2**7 - 1),
    (2**7, 2**8 - 1),
    (2**8, 2**9 - 1),
    # coarse bins
    (0, 2**9 - 1),
    (2**9, 2**10 - 1),
    (2**10, 2**11 - 1),
    (2**11, 2**12 - 1),
    (2**12, 2**13 - 1),
    (0, 2**13 - 1),
    (2**13, 2**14 - 1),
    (2**14, 2**15 - 1),
    (2**15, float("inf")),
]


def get_length_bin(length: int):
    """Get the length bin for a given length."""
    for i, (start, end) in enumerate(LENGTH_BINS):
        if start <= length < end:
            return (start, end)


def compute_rouge(pred_texts, label_texts):
    out = defaultdict(list)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for pred_text, label_text in zip(pred_texts, label_texts):
        scores = scorer.score(pred_text, label_text)
        for k, v in scores.items():
            out[f"{k}.f1"].append(v.fmeasure)
    out_mean = dict()
    for k in out:
        out_mean[k] = np.mean(out[k])
    return out_mean, out


@torch.inference_mode()
def compute_per_token_acc(shift_logits, shift_labels, valid_masks):
    indices = torch.where(valid_masks)
    acc = (shift_logits.argmax(-1) == shift_labels)[indices].float()
    return {
        "per_token_accs": acc.flatten().tolist(),
        "n_per_token_accs": valid_masks.sum().item(),
    }


@torch.inference_mode()
def compute_prefix_matching(shift_logits, shift_labels, valid_masks):
    lengths = valid_masks.sum(dim=1)

    is_wrong = (shift_logits.argmax(-1) != shift_labels) * valid_masks
    is_correct = (shift_logits.argmax(-1) == shift_labels) * valid_masks
    # NOTE: not reliable for multi-turn conversations
    # ie, all tokens in the following user's turn will be correct
    # still monotonically correlate with perf though
    wrong_pos = torch.argmax(is_wrong, dim=1) - torch.argmax(valid_masks, dim=1)
    perf = wrong_pos / lengths

    # if all tokens are correct, set to 1
    perf = torch.where(is_correct.sum(dim=1) == lengths, 1, perf)
    return {
        "prefix_matchings": perf.tolist(),
        "n_prefix_matchings": valid_masks.shape[0],
    }


@torch.inference_mode()
def compute_perplexity(shift_logits, shift_labels, valid_masks):
    return {"perplexities_ph": [1], "n_perplexities_ph": 1}


class Evaluator:
    def __init__(self, metric_fns: list[Callable]):
        self.metric_fns = metric_fns
        self.reset()

    def reset(self):
        self.accum_metrics = defaultdict(lambda: list((0,)))
        self.count = defaultdict(lambda: list((0,)))

    def update(self, shift_logits, shift_labels, valid_masks, lengths=None):
        for metric_fn in self.metric_fns:
            # overall metric
            metric = metric_fn(shift_logits, shift_labels, valid_masks)
            for k, v in metric.items():
                key = k if not k.startswith("n_") else k[2:]
                if k.startswith("n_"):
                    # prefix "n_" indicates the count of the metric
                    self.count[key].append(v)
                else:
                    self.accum_metrics[key] += v
                for start, end in LENGTH_BINS:
                    key_w_len = f"{key}_len_{start}-{end}"
                    if key_w_len not in self.accum_metrics:
                        # add key here so that it shows up in the output
                        self.accum_metrics[key_w_len] = [0]
                        self.count[key_w_len] = [0]
            # split samples into length groups, calculate metric for each group
            if lengths is not None:
                for start, end in LENGTH_BINS:
                    logits, labels, masks = [], [], []

                    for logit, label, m, len in zip(
                        shift_logits, shift_labels, valid_masks, lengths
                    ):
                        if isinstance(len, torch.Tensor):
                            len = len.item()
                        if start <= len < end:
                            logits.append(logit)
                            labels.append(label)
                            masks.append(m)

                    if not logits:
                        continue

                    metric = metric_fn(
                        torch.stack(logits), torch.stack(labels), torch.stack(masks)
                    )
                    for k, v in metric.items():
                        if k.startswith("n_"):
                            key = f"{k[2:]}_len_{start}-{end}"
                            self.count[key].append(v)
                        else:
                            key = f"{k}_len_{start}-{end}"
                            self.accum_metrics[key] += v

    def compute(self):
        # Get result across entire eval set
        result = {
            k: np.sum(v) / np.sum(self.count[k]) if len(v) > 1 else "None"
            for k, v in self.accum_metrics.items()
        }
        # Reset batch statistics
        self.reset()
        return result


@torch.no_grad()
def compute_metrics(
    eval_pred: EvalPrediction,
    compute_result: bool,
    evaluator: Evaluator,
) -> dict | None:
    inputs = eval_pred.inputs
    len_key = "ctx_ids_len" if "ctx_ids_len" in inputs else "input_ids_len"
    lengths = inputs[len_key]
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    valid_masks = torch.where(shift_labels != -100, 1, 0)
    evaluator.update(shift_logits, shift_labels, valid_masks, lengths)
    if compute_result:
        return evaluator.compute()
