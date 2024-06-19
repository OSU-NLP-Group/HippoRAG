import argparse
import json
import os.path
from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig


def run_colbertv2_index(dataset_name: str, index_name: str, corpus_tsv_path: str, checkpoint_path='exp/colbertv2.0', overwrite=False):
    with Run().context(RunConfig(nranks=1, experiment="colbert", root=f"exp/{dataset_name}/")):
        config = ColBERTConfig(
            nbits=2,
            root=f"exp/{dataset_name}/colbert",
        )
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=corpus_tsv_path, overwrite=overwrite)
        print(f'Indexing done for dataset {dataset_name}, index {index_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    checkpoint_path = 'exp/colbertv2.0'
    assert os.path.isdir(checkpoint_path)
    if args.corpus == 'hotpotqa_1000_proposition':
        corpus = json.load(open('data/dense_x_retrieval/hotpotqa_proposition_corpus.json', 'r'))
        corpus_contents = [item['title'] + '\t' + item['propositions'].replace('\n', ' ') for item in corpus]
    elif args.corpus == 'hotpotqa_1000':
        corpus = json.load(open('data/hotpotqa_corpus.json', 'r'))
        corpus_contents = [key + '\t' + ''.join(value) for key, value in corpus.items()]
    elif args.corpus == 'musique_1000_proposition':
        corpus = json.load(open('data/dense_x_retrieval/musique_proposition_corpus.json', 'r'))
        corpus_contents = [item['title'] + '\t' + item['propositions'].replace('\n', ' ') for item in corpus]
    elif args.corpus == 'musique_1000':
        corpus = json.load(open('data/musique_corpus.json', 'r'))
        corpus_contents = [item['title'] + '\t' + item['text'].replace('\n', ' ') for item in corpus]
    elif args.corpus == '2wikimultihopqa_1000_proposition':
        corpus = json.load(open('data/dense_x_retrieval/2wikimultihopqa_proposition_corpus.json', 'r'))
        corpus_contents = [item['title'] + '\t' + item['propositions'].replace('\n', ' ') for item in corpus]
    elif args.corpus == '2wikimultihopqa_1000':
        corpus = json.load(open('data/2wikimultihopqa_corpus.json', 'r'))
        corpus_contents = [item['title'] + '\t' + item['text'].replace('\n', ' ') for item in corpus]
    else:
        raise NotImplementedError(f'Corpus {args.corpus} not implemented')

    print('corpus len', len(corpus_contents))

    if 'proposition' in args.corpus:
        corpus_tsv_path = f'data/dense_x_retrieval/{args.corpus}_colbertv2_corpus.tsv'
    else:
        corpus_tsv_path = f'data/{args.dataset}/{args.corpus}_colbertv2_corpus.tsv'
    with open(corpus_tsv_path, 'w') as f:
        for pid, p in enumerate(corpus_contents):
            f.write(f"{pid}\t\"{p}\"" + '\n')
    print(f'Corpus tsv saved: {corpus_tsv_path}', len(corpus_contents))

    run_colbertv2_index(args.dataset, args.corpus + '_nbits_2', corpus_tsv_path, 'exp/colbertv2.0', overwrite=True)
