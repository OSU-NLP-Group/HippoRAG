import argparse
from hipporag import HippoRAG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--extraction_model', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--retrieval_model', type=str, choices=['facebook/contriever', 'colbertv2'])
    parser.add_argument('--doc_ensemble', action='store_true')
    args = parser.parse_args()

    hipporag = HippoRAG(args.dataset, 'openai', args.extraction_model, args.retrieval_model, doc_ensemble=args.doc_ensemble)

    queries = ["Which Stanford University professor works on Alzheimer's"]
    for query in queries:
        ranks, scores, logs = hipporag.rank_docs(query, top_k=10)

        print(ranks)
        print(scores)
