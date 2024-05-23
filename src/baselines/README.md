# Baselines

The working dir is project root. `ircot.py` contains both single-step and multi-step baselines. Use the following command to prepare data and run the baselines.

## BM25

Before using BM25, ElasticSearch should be installed and running. Assuming ElasticSearch is installed in home dir, running the service:

```shell
cd ~
./elasticsearch-8.10.4/bin/elasticsearch
```

Note that when launching ElasticSearch, you need to keep it running in the background (e.g., tmux or nohup).

Listing all indexes in ElasticSearch, port is 9200 by default:

```shell
curl -X GET "localhost:9200/_cat/indices?v"
```

Create index and run baseline (step=1 means single-step retrieval):

```shell
# MuSiQue
python src/baselines/create_retrieval_index.py --retriever bm25 --dataset musique --corpus musique_1000
python src/baselines/ircot.py --dataset musique --retriever bm25 --max_steps 1 --num_demo 0

# 2Wiki
python src/baselines/create_retrieval_index.py --retriever bm25 --dataset 2wikimultihopqa --corpus 2wikimultihopqa_1000
python src/baselines/ircot.py --dataset 2wikimultihopqa --retriever bm25 --max_steps 1 --num_demo 0

# HotpotQA
python src/baselines/create_retrieval_index.py --retriever bm25 --dataset hotpotqa --corpus hotpotqa_1000
python src/baselines/ircot.py --dataset hotpotqa --retriever bm25 --max_steps 1 --num_demo 0
```

## Contriever

Installing faiss and building faiss index is needed for Contriever.

```shell
# MuSiQue
export CUDA_VISIBLE_DEVICES=0
python src/baselines/mean_pooling_ip_faiss.py --dataset musique --model facebook/contriever
python src/baselines/ircot.py --dataset musique --retriever facebook/contriever --max_steps 1 --num_demo 0

# 2Wiki
python src/baselines/mean_pooling_ip_faiss.py --dataset 2wikimultihopqa --model facebook/contriever
python src/baselines/ircot.py --dataset 2wikimultihopqa --retriever facebook/contriever --max_steps 1 --num_demo 0

# HotpotQA
python src/baselines/mean_pooling_ip_faiss.py --dataset hotpotqa --model facebook/contriever
python src/baselines/ircot.py --dataset hotpotqa --retriever facebook/contriever --max_steps 1 --num_demo 0
```

## ColBERTv2

As mentioned in the main README, download the pre-trained ColBERTv2 [checkpoint](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz) and put it under `exp/colbertv2.0`.

```shell
# MuSiQue
python src/baselines/create_colbertv2_index.py --dataset musique --corpus musique_1000
python src/baselines/ircot.py --dataset musique --retriever colbertv2 --max_steps 1 --num_demo 0

# 2Wiki
python src/baselines/create_colbertv2_index.py --dataset 2wikimultihopqa --corpus 2wikimultihopqa_1000
python src/baselines/ircot.py --dataset 2wikimultihopqa --retriever colbertv2 --max_steps 1 --num_demo 0

# HotpotQA
python src/baselines/create_colbertv2_index.py --dataset hotpotqa --corpus hotpotqa_1000
python src/baselines/ircot.py --dataset hotpotqa --retriever colbertv2 --max_steps 1 --num_demo 0
```

## GTR

See [GTR paper](https://huggingface.co/papers/2112.07899) here.

```shell
# MuSiQue
python src/baselines/create_retrieval_index.py --retriever sentence-transformers/gtr-t5-base --dataset musique --corpus musique_1000 --dim 768
python src/baselines/ircot.py --dataset musique --retriever sentence-transformers/gtr-t5-base --max_steps 1 --num_demo 0

# 2Wiki
python src/baselines/create_retrieval_index.py --retriever sentence-transformers/gtr-t5-base --dataset 2wikimultihopqa --corpus 2wikimultihopqa_1000 --dim 768
python src/baselines/ircot.py --dataset 2wikimultihopqa --retriever sentence-transformers/gtr-t5-base --max_steps 1 --num_demo 0

# HotpotQA
python src/baselines/create_retrieval_index.py --retriever sentence-transformers/gtr-t5-base --dataset hotpotqa --corpus hotpotqa_1000 --dim 768
python src/baselines/ircot.py --dataset hotpotqa --retriever sentence-transformers/gtr-t5-base --max_steps 1 --num_demo 0
```

## Proposition

See [Dense X Retrieval](https://arxiv.org/abs/2312.06648) paper here.

### Proposition ColBERT v2

```shell
# MuSiQue
python src/baselines/create_colbertv2_index.py --corpus musique_1000_proposition --dataset musique
python src/baselines/ircot.py --dataset musique --retriever colbertv2 --unit proposition --max_steps 1 --num_demo 0

# 2Wiki
python src/baselines/create_colbertv2_index.py --corpus 2wikimultihopqa_1000_proposition --dataset 2wikimultihopqa
python src/baselines/ircot.py --dataset 2wikimultihopqa --retriever colbertv2 --unit proposition --max_steps 1 --num_demo 0

# HotpotQA
python src/baselines/create_colbertv2_index.py --corpus hotpotqa_1000_proposition --dataset hotpotqa
python src/baselines/ircot.py --dataset hotpotqa --retriever colbertv2 --unit proposition --max_steps 1 --num_demo 0
```

### Proposition GTR

```shell
# MuSiQue
python src/baselines/create_retrieval_index.py --dataset musique --corpus musique_1000_proposition --retriever sentence-transformers/gtr-t5-base --unit proposition --dim 768
python src/baselines/ircot.py --dataset musique --retriever sentence-transformers/gtr-t5-base --unit proposition --max_steps 1 --num_demo 0

# 2Wiki
python src/baselines/create_retrieval_index.py --dataset 2wikimultihopqa --corpus 2wikimultihopqa_1000_proposition --retriever sentence-transformers/gtr-t5-base --unit proposition --dim 768
python src/baselines/ircot.py --dataset 2wikimultihopqa --retriever sentence-transformers/gtr-t5-base --unit proposition --max_steps 1 --num_demo 0

# HotpotQA
python src/baselines/create_retrieval_index.py --dataset hotpotqa --corpus hotpotqa_1000_proposition --retriever sentence-transformers/gtr-t5-base --unit proposition --dim 768
python src/baselines/ircot.py --dataset hotpotqa --retriever sentence-transformers/gtr-t5-base --unit proposition --max_steps 1 --num_demo 0
```

## IRCoT

If index has been created during single-step retrieval, you could skip the indexing step and call `ircot.py` directly.
To run BM25, set up ElasticSearch as described above.

### IRCoT BM25

```shell
# MuSiQue
python src/baselines/create_retrieval_index.py --retriever bm25 --dataset musique --corpus musique_1000
python src/baselines/ircot.py --dataset musique --retriever bm25 --max_steps 4 --num_demo 1

# 2Wiki
python src/baselines/create_retrieval_index.py --retriever bm25 --dataset 2wikimultihopqa --corpus 2wikimultihopqa_1000
python src/baselines/ircot.py --dataset 2wikimultihopqa --retriever bm25 --max_steps 2 --num_demo 1

# HotpotQA
python src/baselines/create_retrieval_index.py --retriever bm25 --dataset hotpotqa --corpus hotpotqa_1000
python src/baselines/ircot.py --dataset hotpotqa --retriever bm25 --max_steps 2 --num_demo 1
```

### IRCoT Contriever

```shell
# MuSiQue
python src/baselines/mean_pooling_ip_faiss.py --dataset musique --model facebook/contriever
python src/baselines/ircot.py --dataset musique --retriever facebook/contriever --max_steps 4 --num_demo 1

# 2Wiki
python src/baselines/mean_pooling_ip_faiss.py --dataset 2wikimultihopqa --model facebook/contriever
python src/baselines/ircot.py --dataset 2wikimultihopqa --retriever facebook/contriever --max_steps 2 --num_demo 1

# HotpotQA
python src/baselines/mean_pooling_ip_faiss.py --dataset hotpotqa --model facebook/contriever
python src/baselines/ircot.py --dataset hotpotqa --retriever facebook/contriever --max_steps 2 --num_demo 1
```

### IRCoT ColBERTv2

```shell
# MuSiQue
python src/baselines/create_colbertv2_index.py --dataset musique --corpus musique_1000
python src/baselines/ircot.py --dataset musique --retriever colbertv2 --max_steps 4 --num_demo 1

# 2Wiki
python src/baselines/create_colbertv2_index.py --dataset 2wikimultihopqa --corpus 2wikimultihopqa_1000
python src/baselines/ircot.py --dataset 2wikimultihopqa --retriever colbertv2 --max_steps 2 --num_demo 1

# HotpotQA
python src/baselines/create_colbertv2_index.py --dataset hotpotqa --corpus hotpotqa_1000
python src/baselines/ircot.py --dataset hotpotqa --retriever colbertv2 --max_steps 2 --num_demo 1
```
