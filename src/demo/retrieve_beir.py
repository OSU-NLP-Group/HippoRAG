# Note that BEIR uses https://github.com/cvangysel/pytrec_eval to evaluate the retrieval results.
import sys

sys.path.append('.')

from src.data_process.util import merge_chunk_scores, merge_chunks
from src.hipporag import HippoRAG
import os
import pytrec_eval
import argparse
import json
from tqdm import tqdm


def detailed_log(queries, run_dict, eval_res, chunk=False, threshold=None, dpr_only=False):
    logs = []
    for idx, query_id in tqdm(enumerate(run_dict['retrieved']), desc='Error analysis', total=len(run_dict['retrieved'])):
        query_item = queries[idx]
        if threshold is not None and eval_res[query_id]['ndcg'] >= threshold:
            continue
        gold_passages = queries[idx]['paragraphs']
        gold_passage_ids = [p['idx'] for p in query_item['paragraphs']]

        distances = []
        num_dis = 0
        gold_passage_extracted_entities = []
        gold_passage_extracted_triples = []
        if not dpr_only:
            gold_passage_extractions = [hipporag.get_extraction_by_passage_idx(p_idx, chunk) for p_idx in gold_passage_ids]
            gold_passage_extracted_entities = [e for extr in gold_passage_extractions for e in extr['extracted_entities']]
            gold_passage_extracted_triples = [t for extr in gold_passage_extractions for t in extr['extracted_triples']]

            if 'linked_node_scores' in run_dict['log'][query_id]:
                for node_linking in run_dict['log'][query_id]['linked_node_scores']:
                    linked_node_phrase = node_linking[1]
                    distance = []
                    for e in gold_passage_extracted_entities:
                        if e == linked_node_phrase:
                            distance.append(0)
                            num_dis += 1
                        d = hipporag.get_shortest_distance_between_nodes(linked_node_phrase, e)
                        if d > 0:
                            distance.append(d)
                            num_dis += 1
                    distances.append(distance)

        pred_passages = []
        for pred_corpus_id in run_dict['retrieved'][query_id]:
            if not chunk:
                for corpus_item in corpus:
                    if corpus_item['idx'] == pred_corpus_id:
                        pred_passages.append(corpus_item)
            else:
                for corpus_item in corpus:
                    if corpus_item['idx'] == pred_corpus_id or corpus_item['idx'].startswith(f'{pred_corpus_id}_'):
                        pred_passages.append(corpus_item)
        if chunk:
            pred_passages = merge_chunks(pred_passages)

        logs.append({
            'query': queries[idx]['text'],
            'ndcg': eval_res[query_id]['ndcg'],
            'gold_passages': gold_passages,
            'pred_passages': pred_passages,
            'log': run_dict['log'][query_id],
            'distances': distances,
            'avg_distance': sum([sum(d) for d in distances]) / num_dis if num_dis > 0 else None,
            'entities_in_supporting_passage': gold_passage_extracted_entities,
            'triples_in_supporting_passage': gold_passage_extracted_triples,
        })
    return logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name and split, e.g., `sci_fact_test`, `fiqa_dev`.')
    parser.add_argument('--chunk', action='store_true')
    parser.add_argument('--extraction_model', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--retrieval_model', type=str, help="Graph creating retriever name, e.g., 'facebook/contriever', 'colbertv2'")
    parser.add_argument('--linking_model', type=str, help="Node linking model name, e.g., 'facebook/contriever', 'colbertv2'")
    parser.add_argument('--doc_ensemble', action='store_true')
    parser.add_argument('--dpr_only', action='store_true')
    args = parser.parse_args()

    if args.chunk is False and 'chunk' in args.dataset:
        args.chunk = True
    # assert at most only one of them is True
    assert not (args.doc_ensemble and args.dpr_only)
    corpus = json.load(open(f'data/{args.dataset}_corpus.json'))
    qrel = json.load(open(f'data/{args.dataset}_qrel.json'))  # note that this is json file processed from tsv file, used for pytrec_eval
    hipporag = HippoRAG(args.dataset, 'openai', args.extraction_model, args.retrieval_model, doc_ensemble=args.doc_ensemble, dpr_only=args.dpr_only,
                        linking_retriever_name=args.linking_model)

    with open(f'data/{args.dataset}_queries.json') as f:
        queries = json.load(f)

    doc_ensemble_str = 'doc_ensemble' if args.doc_ensemble else 'no_ensemble'
    extraction_str = args.extraction_model.replace('/', '_').replace('.', '_')
    graph_creating_str = args.retrieval_model.replace('/', '_').replace('.', '_')
    if args.linking_model is None:
        args.linking_model = args.retrieval_model
    linking_str = args.linking_model.replace('/', '_').replace('.', '_')
    dpr_only_str = '_dpr_only' if args.dpr_only else ''
    run_output_path = f'exp/{args.dataset}_run_{doc_ensemble_str}_{extraction_str}_{graph_creating_str}_{linking_str}{dpr_only_str}.json'

    metrics = {'map', 'ndcg'}
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)

    if os.path.isfile(run_output_path):
        run_dict = json.load(open(run_output_path))
        print(f'Log file found at {run_output_path}, len: {len(run_dict["retrieved"])}')
    else:
        run_dict = {'retrieved': {}, 'log': {}}  # for pytrec_eval

    to_update_run = False
    for query in tqdm(queries):
        query_text = query['text']
        query_id = query['id']
        if query_id in run_dict['retrieved']:
            continue
        ranks, scores, log = hipporag.rank_docs(query_text, top_k=10)

        retrieved_docs = [corpus[r] for r in ranks]
        run_dict['retrieved'][query_id] = {doc['idx']: score for doc, score in zip(retrieved_docs, scores)}
        run_dict['log'][query_id] = log
        to_update_run = True

    if to_update_run:
        with open(run_output_path, 'w') as f:
            json.dump(run_dict, f)
            print(f'Run saved to {run_output_path}, len: {len(run_dict["retrieved"])}')

    # postprocess run_dict['retrieved'] if the corpus is chunked
    if args.chunk:
        for idx in run_dict['retrieved']:
            run_dict['retrieved'][idx] = merge_chunk_scores(run_dict['retrieved'][idx])

    eval_res = evaluator.evaluate(run_dict['retrieved'])

    # get average scores
    avg_scores = {}
    for metric in metrics:
        avg_scores[metric] = round(sum([v[metric] for v in eval_res.values()]) / len(eval_res), 3)
    print(f'Evaluation results: {avg_scores}')

    logs = detailed_log(queries, run_dict, eval_res, args.chunk, dpr_only=args.dpr_only)
    detailed_log_output_path = f'exp/{args.dataset}_log_{doc_ensemble_str}_{extraction_str}_{graph_creating_str}{linking_str}{dpr_only_str}.json'
    with open(detailed_log_output_path, 'w') as f:
        json.dump(logs, f)
    print(f'Detailed log saved to {detailed_log_output_path}')
