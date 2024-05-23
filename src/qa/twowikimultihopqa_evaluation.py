"""
2Wiki-Multihop QA evaluation script
Adapted from HotpotQA evaluation at https://github.com/hotpotqa/hotpot
"""
import sys
import ujson as json
import re
import string
import itertools
from collections import Counter
import pickle
import os


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def eval_answer(prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    return em, f1, prec, recall


def update_answer(metrics, prediction, golds):
    max_em, max_f1, max_prec, max_recall = 0, 0, 0, 0

    for gold in golds:
        em, f1, prec, recall = eval_answer(prediction, gold)

        max_em = max(max_em, em)
        max_f1 = max(max_f1, f1)
        max_prec = max(max_prec, prec)
        max_recall = max(max_recall, recall)

    metrics['em'] += float(max_em)
    metrics['f1'] += max_f1
    metrics['prec'] += max_prec
    metrics['recall'] += max_recall

    return max_em, max_prec, max_recall


def normalize_sp(sps):
    new_sps = []
    for sp in sps:
        sp = list(sp)
        sp[0] = sp[0].lower()
        new_sps.append(sp)
    return new_sps


def update_sp(metrics, prediction, gold):
    cur_sp_pred = normalize_sp(set(map(tuple, prediction)))
    gold_sp_pred = normalize_sp(set(map(tuple, gold)))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall


def normalize_evi(evidences):
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def recurse(arr):
        for i in range(len(arr)):
            if isinstance(arr[i], str):
                arr[i] = white_space_fix(remove_punc(lower(arr[i])))
            else:
                recurse(arr[i])

    recurse(evidences)

    return evidences


def update_evi(metrics, prediction, gold):
    prediction_normalize = normalize_evi(prediction)
    gold_normalize = normalize_evi(gold)
    #
    cur_evi_pred = set(map(tuple, prediction_normalize))
    gold_evi_pred = list(map(lambda e: set(map(tuple, e)), gold_normalize))
    #
    num_matches = 0
    num_preds = len(cur_evi_pred)
    num_golds = len(gold_evi_pred)

    for pred_evidence in cur_evi_pred:
        for gold_evidences in gold_evi_pred:
            if pred_evidence in gold_evidences:
                num_matches += 1
                break

    prec = num_preds and num_matches / num_preds
    recall = num_golds and num_matches / num_golds
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if num_matches == num_preds == num_golds else 0.0

    metrics['evi_em'] += em
    metrics['evi_f1'] += f1
    metrics['evi_prec'] += prec
    metrics['evi_recall'] += recall

    return em, prec, recall


def eval(prediction_file, gold_file, alias_file):
    aliases = {}

    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)
    with open(alias_file) as f:
        for json_line in map(json.loads, f):
            aliases[json_line["Q_id"]] = {
                "aliases": set(json_line["aliases"] + json_line["demonyms"])
            }

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
               'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
               'evi_em': 0, 'evi_f1': 0, 'evi_prec': 0, 'evi_recall': 0,
               'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        # answer prediction task
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            gold_answers = {dp['answer']}  # Gold span

            if dp['answer_id'] in aliases and aliases[dp['answer_id']]["aliases"]:
                gold_answers.update(aliases[dp['answer_id']]["aliases"])

            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], gold_answers)
        # sentence-level supporting facts prediction task
        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])
        # evidence generation task
        if cur_id not in prediction['evidence']:
            print('missing evidence {}'.format(cur_id))
            can_eval_joint = False
        else:
            gold_evidences = []

            for evidence_idx, (sub_str, rel_str, obj_str) in enumerate(dp['evidences']):
                sub_strs = {sub_str}
                obj_strs = {obj_str}

                if dp['evidences_id'] != []:
                    #
                    assert len(dp['evidences_id']) == len(dp['evidences'])
                    sub_id, rel_id, obj_id = dp['evidences_id'][evidence_idx]

                    assert rel_id == rel_str

                    if sub_id in aliases:
                        sub_strs.update(aliases[sub_id]["aliases"])
                    if obj_id in aliases:
                        obj_strs.update(aliases[obj_id]["aliases"])

                gold_evidence = []

                for sub_str, obj_str in itertools.product(sub_strs, obj_strs):
                    gold_evidence.append([sub_str, rel_str, obj_str])

                gold_evidences.append(gold_evidence)

            evi_em, evi_prec, evi_recall = update_evi(
                metrics, prediction['evidence'][cur_id], gold_evidences)

        if can_eval_joint:
            joint_prec = prec * sp_prec * evi_prec
            joint_recall = recall * sp_recall * evi_recall
            #
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em * evi_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)

    for k in metrics.keys():
        metrics[k] = round(metrics[k] / N * 100, 2)

    print(json.dumps(metrics, indent=4))


if __name__ == '__main__':
    """
    """
    eval(sys.argv[1], sys.argv[2], sys.argv[3])
    # eval("pred.json", "gold.json", "id_aliases.json")
