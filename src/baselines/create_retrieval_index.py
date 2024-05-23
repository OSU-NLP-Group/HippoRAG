import os
import sys

sys.path.append('.')

import argparse
import json
import time

import faiss
import numpy as np
import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.elastic_search_tool import create_and_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--retriever', type=str)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--unit', type=str, choices=['hippo', 'proposition'], default='hippo')
    args = parser.parse_args()

    norm = True

    if args.corpus == 'hotpotqa_1000':
        corpus = json.load(open('data/hotpotqa_corpus.json', 'r'))
        index_name = f'hotpotqa_{len(corpus)}_bm25'
        corpus_contents = [title + '\n' + ''.join(text) for title, text in corpus.items()]
    elif args.corpus == 'hotpotqa_1000_proposition':
        corpus = json.load(open('data/dense_x_retrieval/hotpotqa_proposition_corpus.json', 'r'))
        index_name = f'hotpotqa_{len(corpus)}_proposition_bm25'
        corpus_contents = [item['title'] + '\n' + item['propositions'] for item in corpus]
    elif args.corpus == 'musique_1000':
        corpus = json.load(open('data/musique_corpus.json', 'r'))
        index_name = f'musique_{len(corpus)}_bm25'
        corpus_contents = [item['title'] + '\n' + item['text'] for item in corpus]
    elif args.corpus == 'musique_1000_proposition':
        corpus = json.load(open('data/dense_x_retrieval/musique_proposition_corpus.json', 'r'))
        index_name = f'musique_{len(corpus)}_proposition_bm25'
        corpus_contents = [item['title'] + '\n' + item['propositions'] for item in corpus]
    elif args.corpus == '2wikimultihopqa_1000':
        corpus = json.load(open('data/2wikimultihopqa_corpus.json', 'r'))
        index_name = f'2wikimultihopqa_{len(corpus)}_bm25'
        corpus_contents = [item['title'] + '\n' + item['text'] for item in corpus]
    elif args.corpus == '2wikimultihopqa_1000_proposition':
        corpus = json.load(open('data/dense_x_retrieval/2wikimultihopqa_proposition_corpus.json', 'r'))
        index_name = f'2wikimultihopqa_{len(corpus)}_proposition_bm25'
        corpus_contents = [item['title'] + '\n' + item['propositions'] for item in corpus]
    else:
        raise NotImplementedError('Invalid corpus name')

    if args.retriever == 'bm25':
        start_time = time.time()
        print('Creating index', index_name)
        es = Elasticsearch([{"host": "localhost", "port": 9200, "scheme": "http"}])
        create_and_index(es, index_name, corpus_contents, 'BM25')
        print('BM25 index created, consumed time:', round(time.time() - start_time, 2))

    elif args.retriever.startswith('sentence-transformers/'):
        retriever_label = args.retriever.replace('/', '_').replace('.', '_')
        if norm:
            vector_path = f'data/{args.dataset}/{args.dataset}_{retriever_label}_{args.unit}_vectors_norm.npy'
            index_path = f'data/{args.dataset}/{args.dataset}_{retriever_label}_{args.unit}_ip_norm.index'
        else:
            vector_path = f'data/{args.dataset}/{args.dataset}_{retriever_label}_{args.unit}_vectors.npy'
            index_path = f'data/{args.dataset}/{args.dataset}_{retriever_label}_{args.unit}_ip.index'

        model = SentenceTransformer(args.retriever).to('cuda')

        # Encode passages in batches for efficiency
        batch_size = 16 * torch.cuda.device_count()
        vectors = np.zeros((len(corpus_contents), args.dim))
        for start_idx in tqdm(range(0, len(corpus_contents), batch_size), desc='encoding corpus'):
            end_idx = min(start_idx + batch_size, len(corpus_contents))
            batch_passages = corpus_contents[start_idx:end_idx]

            try:
                batch_embeddings = model.encode(batch_passages)
                if norm:
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    batch_embeddings = batch_embeddings / norms
            except Exception as e:
                batch_embeddings = torch.zeros((len(batch_passages), args.dim))
                print(f'Error at {start_idx}:', e)

            vectors[start_idx:end_idx] = batch_embeddings

        # save vectors to file
        np.save(vector_path, vectors)
        print('vectors saved to {}'.format(vector_path))

        # build inner-product index for corpus
        if os.path.isfile(index_path):
            print('index file already exists:', index_path)
            print('index size: {}'.format(faiss.read_index(index_path).ntotal))
        else:
            print('Building index...')
            index = faiss.IndexFlatIP(args.dim)
            vectors = vectors.astype('float32')
            index.add(vectors)

            # save faiss index to file
            faiss.write_index(index, index_path)
            print('index saved to {}'.format(index_path))
            print('index size: {}'.format(index.ntotal))

    else:
        raise NotImplementedError('Invalid retriever name')
