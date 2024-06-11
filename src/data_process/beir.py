import argparse
import json
import os


def subset_relevant_corpus_statistics(subset_path: str, split: str):
    relevant_corpus_ids = set()
    if os.path.isfile(os.path.join(subset_path, f'qrels/{split}.tsv')):
        with open(os.path.join(subset_path, f'qrels/{split}.tsv')) as f:
            for line in f:
                # skip the first line in tsv file
                if line.startswith('query-id'):
                    continue
                query_id, corpus_id, score = line.strip().split('\t')
                if int(score) == 0:
                    continue
                relevant_corpus_ids.add(corpus_id)
        return len(relevant_corpus_ids)
    else:
        return None


def subset_statistics(subset_path: str):
    # reading full corpus from corpus.jsonl
    full_corpus = []
    with open(os.path.join(subset_path, 'corpus.jsonl')) as f:
        for line in f:
            item = json.loads(line)
            full_corpus.append({'title': item['title'], 'text': item['text'], 'idx': item['_id']})

    # checking corpus relevant to each split set
    len_train_corpus = subset_relevant_corpus_statistics(subset_path, 'train')
    len_dev_corpus = subset_relevant_corpus_statistics(subset_path, 'dev')
    len_test_corpus = subset_relevant_corpus_statistics(subset_path, 'test')

    print(f'{subset_path[subset_path.find("beir/"):]}\t{len(full_corpus)}\t{len_train_corpus}\t{len_dev_corpus}\t{len_test_corpus}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='directory path to BEIR root')
    args = parser.parse_args()

    assert os.path.isdir(args.data)
    for subset_name in os.listdir(args.data):
        if not os.path.isdir(os.path.join(args.data, subset_name)):
            continue
        if len(os.listdir(os.path.join(args.data, subset_name))) == 0:
            continue
        if 'corpus.jsonl' in os.listdir(os.path.join(args.data, subset_name)):
            subset_statistics(os.path.join(args.data, subset_name))
        else:
            for second_subset_name in os.listdir(os.path.join(args.data, subset_name)):
                subset_statistics(os.path.join(args.data, subset_name, second_subset_name))
