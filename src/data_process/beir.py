import argparse
import json
import os.path


def generate_dataset(split: str, qrels_path: str):
    """

    @param split: split name, e.g., 'train', 'test'
    @param qrels_path: the path to BEIR subset's qrels file
    @return: None
    """
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
        corpus_item = corpus[corpus_id]
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

    # save split_corpus
    corpus_output_path = f'data/beir_{subset_name}_{split}_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(split_corpus, f)
        print(f'{split} corpus saved to {corpus_output_path}, len: {len(split_corpus)}')

    # save split_queries
    queries_output_path = f'data/beir_{subset_name}_{split}_queries.json'
    with open(queries_output_path, 'w') as f:
        json.dump(split_queries, f)
        print(f'{split} queries saved to {queries_output_path}, len: {len(split_queries)}')

    # save qrel json file processed from tsv file
    qrels_output_path = f'data/beir_{subset_name}_{split}_qrel.json'
    with open(qrels_output_path, 'w') as f:
        json.dump(query_to_corpus, f)
        print(f'{split} qrels saved to {qrels_output_path}, len: {len(query_to_corpus)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='directory to a BEIR subset')
    args = parser.parse_args()

    subset_name = args.data.split('/')[-1]
    with open(os.path.join(args.data, 'queries.jsonl')) as f:
        queries = f.readlines()
    queries = [json.loads(q) for q in queries]
    queries = {q['_id']: q for q in queries}

    with open(os.path.join(args.data, 'corpus.jsonl')) as f:
        corpus = f.readlines()
    corpus = [json.loads(c) for c in corpus]
    corpus = {c['_id']: c for c in corpus}

    if os.path.isfile(os.path.join(args.data, 'qrels/test.tsv')):  # test set
        # note to skip first line in tsv file
        generate_dataset('test', os.path.join(args.data, 'qrels/test.tsv'))
