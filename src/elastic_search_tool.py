import time

from tqdm import tqdm


def create_and_index(es, index_name, corpus_contents, similarity):
    if not es.indices.exists(index=index_name):
        create_index_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index": {
                    "similarity": {
                        "default": {
                            "type": similarity
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text"
                    }
                }
            }
        }
        es.indices.create(index=index_name, body=create_index_body)

        for idx, doc in tqdm(enumerate(corpus_contents), total=len(corpus_contents), desc='indexing'):
            # try several times if indexing failed because of network issue
            for num_attempt in range(10):
                try:
                    es.index(index=index_name, id=idx, body={"content": doc})
                    break
                except Exception as e:
                    print('Error', e)
                    es.indices.refresh(index=index_name)
                    time.sleep(num_attempt + 1)
        es.indices.refresh(index=index_name)
    else:
        print(f'Index {index_name} already exists, skipping indexing.')


def search(es, index_name, query, top_k):
    search_query = {
        "size": top_k,
        "query": {
            "match": {
                "content": query
            }
        }
    }

    search_results = es.search(index=index_name, body=search_query)
    return [hit['_id'] for hit in search_results['hits']['hits']]


def search_with_score(es, index_name, query, top_k):
    search_query = {
        "size": top_k,
        "query": {
            "match": {
                "content": query
            }
        }
    }

    search_results = es.search(index=index_name, body=search_query)
    hits = search_results['hits']['hits']
    return [(hit['_id'], hit['_score']) for hit in hits]


def search_with_id_and_content(es, index_name, query, top_k):
    res = es.search(index=index_name, body={"query": {"match": {"content": query}}}, size=top_k)
    return [(hit["_id"], hit["_source"]["content"]) for hit in res['hits']['hits']]


def search_with_id_score_and_content(es, index_name, query, top_k):
    res = es.search(index=index_name, body={"query": {"match": {"content": query}}}, size=top_k)
    return [(hit["_id"], hit["_score"], hit["_source"]["content"]) for hit in res['hits']['hits']]


def clear_index(es, index_name):
    es.delete_by_query(
        index=index_name,
        body={
            "query": {
                "match_all": {}
            }
        }
    )


def search_content(es, index_name, query, top_k):
    search_query = {
        "size": top_k,
        "query": {
            "match": {
                "content": query
            }
        }
    }

    search_results = es.search(index=index_name, body=search_query)
    return [hit['_source']['content'] for hit in search_results['hits']['hits']]


def search_content_with_score(es, index_name, query, top_k):
    search_query = {
        "size": top_k,
        "query": {
            "match": {
                "content": query
            }
        }
    }

    search_results = es.search(index=index_name, body=search_query)
    hits = search_results['hits']['hits']
    return [(hit['_source']['content'], hit['_score']) for hit in hits]


def score_all_with_scroll(es, index_name, query, scroll='2m', size=100):
    search_query = {
        "size": size,
        "query": {
            "match": {
                "content": query
            }
        }
    }

    search_results = es.search(index=index_name, body=search_query, scroll=scroll)
    contents_scores = [(hit['_source']['content'], hit['_score']) for hit in search_results['hits']['hits']]

    while True:
        res = es.scroll(scroll_id=search_results['_scroll_id'], scroll=scroll)
        hits = res['hits']['hits']
        if not hits:
            break
        contents_scores.extend([(hit['_source']['content'], hit['_score']) for hit in hits])

    return contents_scores
