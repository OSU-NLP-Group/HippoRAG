import argparse
import json
import os.path

import ipdb
import pandas as pd
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries
from processing import *
import pickle


def retrieve_knn(kb, queries, duplicate=True, nns=100):
    checkpoint_path = 'exp/colbertv2.0'

    if duplicate:
        kb = list(set(list(kb) + list(queries)))  # Duplicating queries to obtain score of query to query and normalize

    with open('data/lm_vectors/colbert/corpus.tsv', 'w') as f:  # save to tsv
        for pid, p in enumerate(kb):
            f.write(f"{pid}\t\"{p}\"" + '\n')

    with open('data/lm_vectors/colbert/queries.tsv', 'w') as f:  # save to tsv
        for qid, q in enumerate(queries):
            f.write(f"{qid}\t{q}" + '\n')

    ranking_output_path = 'nbits_2_ranking.tsv'
    # index
    with Run().context(RunConfig(nranks=1, experiment="colbert", root="")):
        config = ColBERTConfig(
            nbits=2,
            root="data/lm_vectors/colbert"
        )
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name="nbits_2", collection="data/lm_vectors/colbert/corpus.tsv", overwrite=True)

    # retrieval
    with Run().context(RunConfig(nranks=1, experiment="colbert", root="")):
        config = ColBERTConfig(
            root="data/lm_vectors/colbert",
        )
        searcher = Searcher(index="nbits_2", config=config)
        queries = Queries("data/lm_vectors/colbert/queries.tsv")
        ranking = searcher.search_all(queries, k=nns)

    ranking_dict = {}

    for i in range(len(queries)):
        query = queries[i]
        rank = ranking.data[i]
        max_score = rank[0][2]
        if duplicate:
            rank = rank[1:]
        ranking_dict[query] = ([kb[r[0]] for r in rank], [r[2] / max_score for r in rank])

    return ranking_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()

    string_filename = args.filename

    # prepare tsv data
    string_df = pd.read_csv(string_filename, sep='\t')
    string_df.strings = [processing_phrases(str(s)) for s in string_df.strings]

    queries = string_df[string_df.type == 'query']
    kb = string_df[string_df.type == 'kb']

    nearest_neighbors = retrieve_knn(kb.strings.values, queries.strings.values)
    output_path = 'data/lm_vectors/colbert/nearest_neighbor_{}.p'.format(string_filename.split('/')[1].split('.')[0])
    pickle.dump(nearest_neighbors, open(output_path, 'wb'))
    print('Saved nearest neighbors to {}'.format(output_path))
