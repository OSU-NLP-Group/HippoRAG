import argparse
import json
import logging
import os
from typing import List

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.StandardRAG import StandardRAG
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import string_to_bool


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_titles = {item[0] for item in sample["supporting_facts"]}
            supporting_contexts = [item for item in sample["context"] if item[0] in gold_titles]
            separator = "" if dataset_name.startswith("hotpotqa") else " "
            gold_doc = [item[0] + "\n" + separator.join(item[1]) for item in supporting_contexts]
        elif "contexts" in sample:
            gold_doc = [item["title"] + "\n" + item["text"] for item in sample["contexts"] if item["is_supporting"]]
        else:
            assert "paragraphs" in sample, "`paragraphs` should be in sample, or disable retrieval evaluation"
            paragraphs = [item for item in sample["paragraphs"] if item.get("is_supporting", True)]
            gold_doc = [item["title"] + "\n" + (item["text"] if "text" in item else item["paragraph_text"]) for item in paragraphs]
        gold_docs.append(list(set(gold_doc)))
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample in samples:
        if "answer" in sample or "gold_ans" in sample:
            gold_answer = sample.get("answer", sample.get("gold_ans"))
        elif "reference" in sample:
            gold_answer = sample["reference"]
        elif "obj" in sample:
            gold_answer = [sample["obj"], sample["o_wiki_title"]]
            for field in ("possible_answers", "o_aliases"):
                value = sample[field]
                gold_answer.extend(value if isinstance(value, list) else [value])
        else:
            raise ValueError("Each query sample must contain an answer field")
        answers = [gold_answer] if isinstance(gold_answer, str) else list(gold_answer)
        answers = set(answers)
        answers.update(sample.get("answer_aliases", []))
        gold_answers.append(answers)
    return gold_answers


def parse_args():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA experiments")
    parser.add_argument("--dataset", default="musique", help="Dataset name under reproduce/dataset")
    parser.add_argument("--rag_type", choices=["hipporag", "standard"], default="hipporag", help="Retrieval method; standard reproduces the DPR-style dense baseline")
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1", help="OpenAI-compatible LLM base URL")
    parser.add_argument("--llm_name", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--embedding_name", default="nvidia/NV-Embed-v2", help="Embedding model name")
    parser.add_argument("--azure_endpoint", help="Azure OpenAI chat completions endpoint")
    parser.add_argument("--azure_embedding_endpoint", help="Azure OpenAI embeddings endpoint")
    parser.add_argument("--embedding_batch_size", type=int, default=8, help="Embedding batch size")
    parser.add_argument("--force_index_from_scratch", default="false", help="Rebuild all stored index and graph data")
    parser.add_argument("--force_openie_from_scratch", default="false", help="Regenerate stored OpenIE results")
    parser.add_argument("--openie_mode", choices=["online", "offline"], default="online", help="OpenIE execution mode")
    parser.add_argument("--save_dir", default="outputs", help="Save directory")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = os.path.join(args.save_dir, args.dataset) if args.save_dir == "outputs" else f"{args.save_dir}_{args.dataset}"
    with open(f"reproduce/dataset/{args.dataset}_corpus.json") as file:
        corpus = json.load(file)
    with open(f"reproduce/dataset/{args.dataset}.json") as file:
        samples = json.load(file)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    queries = [sample["question"] for sample in samples]
    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, args.dataset)
    except (AssertionError, KeyError):
        logging.warning("Retrieval evaluation is disabled because supporting documents are unavailable")
        gold_docs = None

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        azure_endpoint=args.azure_endpoint,
        azure_embedding_endpoint=args.azure_embedding_endpoint,
        dataset=args.dataset,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=string_to_bool(args.force_index_from_scratch),
        force_openie_from_scratch=string_to_bool(args.force_openie_from_scratch),
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=5,
        embedding_batch_size=args.embedding_batch_size,
        openie_mode=args.openie_mode,
    )
    logging.basicConfig(level=logging.INFO)
    rag_class = HippoRAG if args.rag_type == "hipporag" else StandardRAG
    rag = rag_class(global_config=config)
    rag.index(docs)
    rag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=gold_answers)


if __name__ == "__main__":
    main()
