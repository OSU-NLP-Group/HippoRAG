import argparse
import json
import os.path

from src.data_process.util import chunk_corpus


def generate_dataset_with_relevant_corpus(split: str, qrels_path: str, chunk=False):
    """

    @param split: split name, e.g., 'train', 'test'
    @param qrels_path: the path to BEIR subset's qrels file
    @return: None
    """
    chunk_state = '_chunk' if chunk else ''
    with open(qrels_path) as f:
        qrels = f.readlines()
    qrels = [q.split() for q in qrels[1:]]
    print(f'#{split}', len(qrels))
    split_corpus = []
    split_corpus_ids = set()
    split_queries = []
    split_query_ids = set()
    query_to_corpus = {}  # query_id -> [corpus_id]
    for idx, item in enumerate(qrels):
        query_id = item[0]
        corpus_id = item[1]
        score = item[2]
        if int(score) == 0:
            continue
        try:
            corpus_item = corpus[corpus_id]
        except KeyError:
            print(f'corpus_id {corpus_id} not found')
            continue
        query_item = queries[query_id]

        if corpus_item['_id'] not in split_corpus_ids:
            split_corpus.append({'title': corpus_item['title'], 'text': corpus_item['text'], 'idx': corpus_item['_id']})
            split_corpus_ids.add(corpus_item['_id'])
        if query_item['_id'] not in split_query_ids:
            split_queries.append({**query_item, 'id': query_item['_id'], 'question': query_item['text']})
            split_query_ids.add(query_item['_id'])
        if query_id not in query_to_corpus:
            query_to_corpus[query_id] = {}
        query_to_corpus[query_id][corpus_id] = int(score)

    # add supporting passages to query info
    for query in split_queries:
        query['paragraphs'] = []
        for c in query_to_corpus[query['_id']]:
            corpus_item = corpus[c]
            query['paragraphs'].append({'title': corpus_item['title'], 'text': corpus_item['text'], 'idx': corpus_item['_id']})

    if chunk:
        split_corpus = chunk_corpus(split_corpus)
    # save split_corpus
    corpus_output_path = f'data/beir_{subset_name}_{split}{chunk_state}_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(split_corpus, f)
        print(f'{split} corpus saved to {corpus_output_path}, len: {len(split_corpus)}')

    # save split_queries
    queries_output_path = f'data/beir_{subset_name}_{split}{chunk_state}_queries.json'
    with open(queries_output_path, 'w') as f:
        json.dump(split_queries, f)
        print(f'{split} queries saved to {queries_output_path}, len: {len(split_queries)}')

    # save qrel json file processed from tsv file
    qrels_output_path = f'data/beir_{subset_name}_{split}{chunk_state}_qrel.json'
    with open(qrels_output_path, 'w') as f:
        json.dump(query_to_corpus, f)
        print(f'{split} qrels saved to {qrels_output_path}, len: {len(query_to_corpus)}')


def generate_dataest_with_full_corpus(split, qrels_path: str, chunk=False):
    chunk_state = '_chunk' if chunk else ''
    with open(qrels_path) as f:
        qrels = f.readlines()
    qrels = [q.split() for q in qrels[1:]]
    print(f'#{split}', len(qrels))
    split_queries = []
    split_query_ids = set()
    query_to_corpus = {}  # query_id -> [corpus_id]
    corpus_ids = set()
    full_corpus = []
    for idx, item in enumerate(qrels):
        query_id = item[0]
        corpus_id = item[1]
        score = item[2]
        if int(score) == 0:
            continue
        query_item = queries[query_id]

        if query_item['_id'] not in split_query_ids:
            split_queries.append({**query_item, 'id': query_item['_id'], 'question': query_item['text']})
            split_query_ids.add(query_item['_id'])
        if query_id not in query_to_corpus:
            query_to_corpus[query_id] = {}
        query_to_corpus[query_id][corpus_id] = int(score)

    # add supporting passages to query info
    for query in split_queries:
        query['paragraphs'] = []
        for c in query_to_corpus[query['_id']]:
            if c not in corpus:
                print(f'corpus_id {c} not found')
                continue
            corpus_item = corpus[c]
            query['paragraphs'].append({'title': corpus_item['title'], 'text': corpus_item['text'], 'idx': corpus_item['_id']})

    # read jsonl file to get full corpus
    with open(os.path.join(args.data, 'corpus.jsonl')) as f:
        # read each line as json
        for line in f:
            item = json.loads(line)
            if item['_id'] not in corpus_ids:
                full_corpus.append({'title': item['title'], 'text': item['text'], 'idx': item['_id']})
                corpus_ids.add(item['_id'])

    if chunk:
        full_corpus = chunk_corpus(full_corpus)
    # save split_corpus
    corpus_output_path = f'data/beir_{subset_name}_{split}{chunk_state}_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(full_corpus, f)
        print(f'{split} corpus saved to {corpus_output_path}, len: {len(full_corpus)}')

    # save split_queries
    queries_output_path = f'data/beir_{subset_name}_{split}{chunk_state}_queries.json'
    with open(queries_output_path, 'w') as f:
        json.dump(split_queries, f)
        print(f'{split} queries saved to {queries_output_path}, len: {len(split_queries)}')

    # save qrel json file processed from tsv file
    qrels_output_path = f'data/beir_{subset_name}_{split}{chunk_state}_qrel.json'
    with open(qrels_output_path, 'w') as f:
        json.dump(query_to_corpus, f)
        print(f'{split} qrels saved to {qrels_output_path}, len: {len(query_to_corpus)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='directory path to a BEIR subset')
    parser.add_argument('--corpus', type=str, choices=['full', 'relevant'], help='full or relevant corpus', default='full')
    parser.add_argument('--chunk', action='store_true')
    args = parser.parse_args()

    print(args)
    subset_name = args.data.split('/')[-1]
    with open(os.path.join(args.data, 'queries.jsonl')) as f:
        queries = f.readlines()
    queries = [json.loads(q) for q in queries]
    queries = {q['_id']: q for q in queries}

    with open(os.path.join(args.data, 'corpus.jsonl')) as f:
        corpus = f.readlines()
    corpus = [json.loads(c) for c in corpus]
    corpus = {c['_id']: c for c in corpus}

    for split in ['train', 'dev', 'test']:
        if os.path.isfile(os.path.join(args.data, f'qrels/{split}.tsv')):
            if args.corpus == 'relevant':
                generate_dataset_with_relevant_corpus(split, os.path.join(args.data, f'qrels/{split}.tsv'), args.chunk)
            elif args.corpus == 'full':
                generate_dataest_with_full_corpus(split, os.path.join(args.data, f'qrels/{split}.tsv'), args.chunk)
        else:
            print(f'{split} not found, skipped')
