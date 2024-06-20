import os
import sys

from src.processing import mean_pooling, mean_pooling_embedding_with_normalization

sys.path.append('.')

import argparse
import json
import numpy as np

import faiss
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='facebook/contriever')
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--unit', type=str, default='hippo', choices=['hippo', 'proposition'])
    args = parser.parse_args()

    dim = args.dim
    norm = True
    model_label = args.model.replace('/', '_').replace('.', '_')
    if norm:
        vector_path = f'data/{args.dataset}/{args.dataset}_{model_label}_{args.unit}_vectors_norm.npy'
        index_path = f'data/{args.dataset}/{args.dataset}_{model_label}_{args.unit}_ip_norm.index'
    else:
        vector_path = f'data/{args.dataset}/{args.dataset}_{model_label}_{args.unit}_vectors.npy'
        index_path = f'data/{args.dataset}/{args.dataset}_{model_label}_{args.unit}_ip.index'

    corpus_contents = []
    if args.dataset == 'hotpotqa':
        if args.unit == 'hippo':
            corpus = json.load(open('data/hotpotqa_corpus.json', 'r'))
            for title, text in corpus.items():
                corpus_contents.append(title + '\n' + '\n'.join(text))
        elif args.unit == 'proposition':
            corpus = json.load(open('data/dense_x_retrieval/hotpotqa_proposition_corpus.json', 'r'))
    elif args.dataset == 'musique':
        if args.unit == 'hippo':
            corpus = json.load(open('data/musique_corpus.json', 'r'))
            corpus_contents = [item['title'] + '\n' + item['text'] for item in corpus]
        elif args.unit == 'proposition':
            corpus = json.load(open('data/dense_x_retrieval/musique_proposition_corpus.json', 'r'))
    elif args.dataset == '2wikimultihopqa':
        if args.unit == 'hippo':
            corpus = json.load(open('data/2wikimultihopqa_corpus.json', 'r'))
            corpus_contents = [item['title'] + '\n' + item['text'] for item in corpus]
        elif args.unit == 'proposition':
            corpus = json.load(open('data/dense_x_retrieval/2wikimultihopqa_proposition_corpus.json', 'r'))

    if args.unit == 'proposition':
        for item in corpus:
            corpus_contents.append(item['title'] + '\n' + item['propositions'])

    print('corpus size: {}'.format(len(corpus_contents)))

    if os.path.isfile(vector_path):
        print('Loading existing vectors:', vector_path)
        vectors = np.load(vector_path)
        print('Vectors loaded:', len(vectors))
    else:
        # load model
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)

        # Check if multiple GPUs are available and if so, use them all
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        model.to('cuda')

        # Encode passages in batches for efficiency
        batch_size = 16 * torch.cuda.device_count()
        vectors = np.zeros((len(corpus_contents), dim))
        for start_idx in tqdm(range(0, len(corpus_contents), batch_size), desc='encoding corpus'):
            end_idx = min(start_idx + batch_size, len(corpus_contents))
            batch_passages = corpus_contents[start_idx:end_idx]

            if norm:
                batch_embeddings = mean_pooling_embedding_with_normalization(batch_passages, tokenizer, model)
            else:
                try:
                    inputs = tokenizer(batch_passages, padding=True, truncation=True, return_tensors='pt', max_length=384).to('cuda')
                    outputs = model(**inputs)
                    batch_embeddings = mean_pooling(outputs[0], inputs['attention_mask'])

                except Exception as e:
                    batch_embeddings = torch.zeros((len(batch_passages), dim))
                    print(f'Error at {start_idx}:', e)

            vectors[start_idx:end_idx] = batch_embeddings.to('cpu').detach().numpy()

        # save vectors to file
        np.save(vector_path, vectors)
        print('vectors saved to {}'.format(vector_path))

    # build inner-product index for corpus
    if os.path.isfile(index_path):
        print('index file already exists:', index_path)
        print('index size: {}'.format(faiss.read_index(index_path).ntotal))
    else:
        print('Building index...')
        index = faiss.IndexFlatIP(dim)
        vectors = vectors.astype('float32')
        index.add(vectors)

        # save faiss index to file
        faiss.write_index(index, index_path)
        print('index saved to {}'.format(index_path))
        print('index size: {}'.format(index.ntotal))
