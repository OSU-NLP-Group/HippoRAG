import os
from typing import List
import json
import argparse
import logging

from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

logging.basicConfig(level=logging.DEBUG)

def main():



    # Prepare datasets and evaluation
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County."
    ]

    save_dir = 'outputs/local_test'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'Transformers/Qwen/Qwen2.5-7B-Instruct'  # Any OpenAI model name
    embedding_model_name = 'Transformers/BAAI/bge-m3'  # Embedding model name (NV-Embed, GritLM or Contriever for now)

    global_config = BaseConfig(
        openie_mode='Transformers-offline',
        information_extraction_model_name='Transformers/Qwen/Qwen2.5-7B-Instruct'
    )

    # Startup a HippoRAG instance
    hipporag = HippoRAG(global_config,
                        save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        )

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    # For Evaluation
    answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"]
    ]

    gold_docs = [
        ["George Rankin is a politician."],
        ["Cinderella attended the royal ball.",
         "The prince used the lost glass slipper to search the kingdom.",
         "When the slipper fit perfectly, Cinderella was reunited with the prince."],
        ["Erik Hort's birthplace is Montebello.",
         "Montebello is a part of Rockland County."]
    ]

    print(hipporag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers)[-2:])

if __name__ == "__main__":
    main()
