import sys

sys.path.append('.')
import argparse

from src.hipporag import HippoRAG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, default='sample')
    parser.add_argument('--extraction_model', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--retrieval_model', type=str, choices=['facebook/contriever', 'colbertv2'], required=True)
    parser.add_argument('--doc_ensemble', action='store_true')
    parser.add_argument('--dpr_only', action='store_true')
    args = parser.parse_args()

    assert not (args.doc_ensemble and args.dpr_only)
    hipporag = HippoRAG(args.dataset, 'openai', args.extraction_model, args.retrieval_model, doc_ensemble=args.doc_ensemble, dpr_only=args.dpr_only)

    queries = ["Which Stanford University professor works on Alzheimer's"]
    for query in queries:
        ranks, scores, logs = hipporag.rank_docs(query, top_k=10)

        print(ranks)
        print(scores)
