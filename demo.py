import os
from typing import List
import json
import argparse
import logging

from hipporag import HippoRAG

def main():

    corpus_path = f"reproduce/dataset/sample_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    # Prepare datasets and evaluation
    samples = json.load(open(f"reproduce/dataset/sample.json", "r"))
    all_queries = [s['question'] for s in samples]

    hipporag = HippoRAG(working_dir="default")

    hipporag.index(docs=docs)

    # Retrieval and QA
    retrieval_results = hipporag.retrieve(queries=all_queries)
    hipporag.qa(retrieval_results)

    hipporag.rag_qa(queries=retrieval_results)
    hipporag.rag_qa(queries=all_queries)

if __name__ == "__main__":
    main()
