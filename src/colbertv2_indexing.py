import argparse
import json
import pickle

import numpy as np
from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig


def colbertv2_index(corpus: list, dataset_name: str, exp_name: str, index_name='nbits_2', checkpoint_path='exp/colbertv2.0', overwrite='reuse'):
    """
    Indexing corpus and phrases using colbertv2
    @param corpus:
    @return:
    """
    corpus_processed = [x.replace('\n', '\t') for x in corpus]

    corpus_tsv_file_path = f'data/lm_vectors/colbert/{dataset_name}_{exp_name}_{len(corpus_processed)}.tsv'
    with open(corpus_tsv_file_path, 'w') as f:  # save to tsv
        for pid, p in enumerate(corpus_processed):
            f.write(f"{pid}\t\"{p}\"" + '\n')
    root_path = f'data/lm_vectors/colbert/{dataset_name}'

    # indexing corpus
    with Run().context(RunConfig(nranks=1, experiment=exp_name, root=root_path)):
        config = ColBERTConfig(
            nbits=2,
            root=root_path,
        )
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=corpus_tsv_file_path, overwrite=overwrite)


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

    colbertv2_index(corpus_contents, args.dataset, 'corpus', checkpoint_path, overwrite=True)

    kb_phrase_dict = pickle.load(open(args.phrase, 'rb'))
    phrases = np.array(list(kb_phrase_dict.keys()))[np.argsort(list(kb_phrase_dict.values()))]
    phrases = phrases.tolist()
    colbertv2_index(phrases, args.dataset, 'phrase', checkpoint_path, overwrite=True)
