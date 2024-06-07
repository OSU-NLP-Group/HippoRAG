import argparse
import json
import os.path

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

    if os.path.isfile(os.path.join(args.data, 'qrels/test.tsv')):
        # note to skip first line in tsv file
        with open(os.path.join(args.data, 'qrels/test.tsv')) as f:
            qrels_test = f.readlines()
        qrels_test = [q.split() for q in qrels_test[1:]]
        print('#test', len(qrels_test))

        test_corpus = []
        test_corpus_ids = set()
        test_queries = []
        test_queries_ids = set()

        for idx, item in enumerate(qrels_test):
            query_id = item[0]
            corpus_id = item[1]
            score = item[2]
            corpus_item = corpus[corpus_id]
            query_item = queries[query_id]

            if corpus_item['_id'] not in test_corpus_ids:
                test_corpus.append({'title': corpus_item['title'], 'text': corpus_item['text'], 'idx': corpus_item['_id']})
                test_corpus_ids.add(corpus_item['_id'])
            if query_item['_id'] not in test_queries_ids:
                test_queries.append({**query_item, 'id': query_item['_id'], 'question': query_item['text']})
                test_queries_ids.add(query_item['_id'])

        corpus_output_path = f'data/beir_{subset_name}_test_corpus.json'
        with open(corpus_output_path, 'w') as f:
            json.dump(test_corpus, f)
            print(f'test corpus saved to {corpus_output_path}, len: {len(test_corpus)}')
        queries_output_path = f'data/beir_{subset_name}_test_queries.json'
        with open(queries_output_path, 'w') as f:
            json.dump(test_queries, f)
            print(f'test queries saved to {queries_output_path}, len: {len(test_queries)}')
