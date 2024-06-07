# Note that BEIR uses https://github.com/cvangysel/pytrec_eval to evaluate the retrieval results.
import sys

sys.path.append('.')
from src.hipporag import HippoRAG
import os
import pytrec_eval
import argparse
import json

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--extraction_model', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--retrieval_model', type=str, choices=['facebook/contriever', 'colbertv2'])
    parser.add_argument('--doc_ensemble', action='store_true')
    args = parser.parse_args()

    corpus = json.load(open(f'data/{args.dataset}_corpus.json'))
    qrel = json.load(open(f'data/{args.dataset}_qrel.json'))  # note that this is json file processed from tsv file, used for pytrec_eval
    hipporag = HippoRAG(args.dataset, 'openai', args.extraction_model, args.retrieval_model, doc_ensemble=args.doc_ensemble)

    with open(f'data/{args.dataset}_queries.json') as f:
        queries = json.load(f)

    run_output_path = f'output/{args.dataset}_run.json'
    if os.path.isfile(run_output_path):
        run_dict = json.load(open(run_output_path))
    else:
        run_dict = {}  # for pytrec_eval

    to_update_run = False
    for query in tqdm(queries):
        query_text = query['text']
        query_id = query['id']
        if query_id in run_dict:
            continue
        ranks, scores, logs = hipporag.rank_docs(query_text, top_k=10)

        retrieved_docs = [corpus[r] for r in ranks]
        run_dict[query_id] = {doc['idx']: score for doc, score in zip(retrieved_docs, scores)}
        to_update_run = True

    if to_update_run:
        with open(run_output_path, 'w') as f:
            json.dump(run_dict, f)
            print(f'Run saved to {run_output_path}, len: {len(run_dict)}')

    metrics = {'map', 'ndcg'}
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)
    eval_res = evaluator.evaluate(run_dict)
    # get average scores
    avg_scores = {}
    for metric in metrics:
        avg_scores[metric] = round(sum([v[metric] for v in eval_res.values()]) / len(eval_res), 3)
    print(f'Evaluation results: {avg_scores}')
