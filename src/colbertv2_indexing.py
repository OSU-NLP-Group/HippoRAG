import argparse
import json
import pickle

import ipdb
import numpy as np
from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--phrase', type=str)
    args = parser.parse_args()

    checkpoint_path = 'exp/colbertv2.0'

    corpus = json.load(open(args.corpus, 'r'))
    # get corpus tsv
    if 'hotpotqa' in args.dataset:
        corpus_contents = [x[0] + ' ' + ''.join(x[1]) for x in corpus.items()]
    else:
        corpus_contents = [x['title'] + ' ' + x['text'].replace('\n', ' ') for x in corpus]

    corpus_tsv_file_path = f'data/lm_vectors/colbert/{args.dataset}_corpus_{len(corpus_contents)}.tsv'
    with open(corpus_tsv_file_path, 'w') as f:  # save to tsv
        for pid, p in enumerate(corpus_contents):
            f.write(f"{pid}\t\"{p}\"" + '\n')

    root_path = f'data/lm_vectors/colbert/{args.dataset}'
    # indexing corpus
    with Run().context(RunConfig(nranks=1, experiment='corpus', root=root_path)):
        config = ColBERTConfig(
            nbits=2,
            root=root_path,
        )
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=f"nbits_2", collection=corpus_tsv_file_path, overwrite=True)

    kb_phrase_dict = pickle.load(open(args.phrase, 'rb'))
    phrases = np.array(list(kb_phrase_dict.keys()))[np.argsort(list(kb_phrase_dict.values()))]
    phrases = phrases.tolist()
    # get phrases tsv
    phrases_tsv_file_path = f'data/lm_vectors/colbert/{args.dataset}_phrases_{len(phrases)}.tsv'
    with open(phrases_tsv_file_path, 'w') as f:  # save to tsv
        for pid, p in enumerate(phrases):
            f.write(f"{pid}\t\"{p}\"" + '\n')
    # indexing phrases
    with Run().context(RunConfig(nranks=1, experiment='phrase', root=root_path)):
        config = ColBERTConfig(
            nbits=2,
            root=root_path,
        )
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=f"nbits_2", collection=phrases, overwrite=True)
