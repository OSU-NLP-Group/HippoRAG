import sys

sys.path.append('.')

from src.langchain_util import LangChainModel
from src.qa.qa_reader import qa_read

import argparse

from src.hipporag import HippoRAG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, default='sample')
    parser.add_argument('--extraction_model', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--retrieval_model', type=str, required=True, help='e.g., "facebook/contriever", "colbertv2"')
    parser.add_argument('--doc_ensemble', action='store_true')
    parser.add_argument('--dpr_only', action='store_true')
    args = parser.parse_args()

    assert not (args.doc_ensemble and args.dpr_only)
    hipporag = HippoRAG(args.dataset, 'openai', args.extraction_model, args.retrieval_model, doc_ensemble=args.doc_ensemble, dpr_only=args.dpr_only,
                        qa_model=LangChainModel('openai', 'gpt-3.5-turbo'))

    queries = ["Which Stanford University professor works on Alzheimer's"]
    # qa_few_shot_samples = [{'document': '', 'question': '', 'thought': '', 'answer': ''}]
    # Prepare a list for few-shot QA, where each element is a dict with keys 'document', 'question', 'thought', 'answer' ('document' and 'thought' are optional)
    qa_few_shot_samples = None

    for query in queries:
        ranks, scores, logs = hipporag.rank_docs(query, top_k=10)
        retrieved_passages = [hipporag.get_passage_by_idx(rank) for rank in ranks]

        response = qa_read(query, retrieved_passages, qa_few_shot_samples, hipporag.qa_model)
        print(ranks)
        print(scores)
        print(response)
