from hipporag import HippoRAG
from hipporag.utils.sample_data import ANSWERS, DOCS, GOLD_DOCS, QUERIES


def run_demo(**kwargs):
    rag = HippoRAG(**kwargs)
    rag.index(docs=DOCS)
    print(rag.rag_qa(queries=QUERIES, gold_docs=GOLD_DOCS, gold_answers=ANSWERS))
