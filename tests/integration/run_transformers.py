import logging

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.sample_data import ANSWERS, DOCS, GOLD_DOCS, QUERIES


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    config = BaseConfig(openie_mode="Transformers-offline", information_extraction_model_name="Transformers/Qwen/Qwen2.5-7B-Instruct")
    rag = HippoRAG(config, save_dir="outputs/transformers_test", llm_model_name="Transformers/Qwen/Qwen2.5-7B-Instruct", embedding_model_name="Transformers/BAAI/bge-m3")
    rag.index(docs=DOCS)
    print(rag.rag_qa(queries=QUERIES, gold_docs=GOLD_DOCS, gold_answers=ANSWERS)[-2:])
