from hipporag import HippoRAG
from hipporag.utils.sample_data import ANSWERS, DOCS, EXTRA_DOCS, GOLD_DOCS, QUERIES


def run_lifecycle(**kwargs):
    rag = HippoRAG(**kwargs)
    rag.index(docs=DOCS)
    print(rag.rag_qa(queries=QUERIES, gold_docs=GOLD_DOCS, gold_answers=ANSWERS)[-2:])

    reloaded_rag = HippoRAG(**kwargs)
    print(reloaded_rag.rag_qa(queries=QUERIES, gold_docs=GOLD_DOCS, gold_answers=ANSWERS)[-2:])
    reloaded_rag.index(docs=EXTRA_DOCS)
    print(reloaded_rag.rag_qa(queries=QUERIES, gold_docs=GOLD_DOCS, gold_answers=ANSWERS)[-2:])
    reloaded_rag.delete(EXTRA_DOCS)
    print(reloaded_rag.rag_qa(queries=QUERIES, gold_docs=GOLD_DOCS, gold_answers=ANSWERS)[-2:])
