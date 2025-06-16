from mcp.server.fastmcp import FastMCP
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import QuerySolution
from src.hipporag.utils.misc_utils import string_to_bool
from main import get_gold_docs, get_gold_answers
import pathlib, logging, os, json
import argparse
import logging

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Global variables
hipporag = None
config = None

mcp = FastMCP(
    "HippoRAG MCP Server",
    dependencies=["hipporag"]
)

@mcp.tool(
    name="retrieve",
    description="Retrieve top-k passages for a query via HippoRAG"
)
def retrieve(query: str, top_k: int = 10) -> dict:
    """
    Parameters
    ----------
    query : natural-language query
    top_k : number of hits to return (≤ retrieval_top_k)

    Returns
    -------
    dict  –  {
      "results": [
        { "node_id": "...", "score": 0.83, "text": "..." },
        ...
      ]
    }
    """
    solutions = hipporag.retrieve([query], num_to_retrieve=top_k)
    first: QuerySolution = solutions[0]                      # single-query call
    return {
        "question": first.question,
        "docs": first.docs[:5],
        "doc_scores": first.doc_scores[:5],
    }

@mcp.tool(
    name="rag",
    description="Answer a question with retrieved documents"
)
def rag(query: str, max_answers: int = 3) -> dict:
    """
    Performs RAG + reasoning and returns top answers with evidence.
    """
    answers: QuerySolution = hipporag.rag_qa([query])[0]
    return {
        "question": answers.question,
        "answers": answers.answer,
        "docs": answers.docs[:5],
    }


@mcp.resource("hipporag://config")
def hipporag_config() -> str:
    """Return the current HippoRAG configuration."""
    return json.dumps(config.__dict__, indent=2, default=str)


def parse_args():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA")
    parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini', help='LLM name')
    parser.add_argument('--embedding_name', type=str, default='text-embedding-3-small', help='embedding model name')
    parser.add_argument('--azure_endpoint', type=str, default=None, help='Azure Endpoint URL')
    parser.add_argument('--azure_embedding_endpoint', type=str, default=None, help='Azure Embedding Endpoint')
    parser.add_argument('--force_index_from_scratch', type=str, default='false',
                        help='If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false', help='If set to False, will try to first reuse openie results for the corpus if they exist.')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online',
                        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes")
    parser.add_argument('--save_dir', type=str, default='outputs', help='Save directory')
    args = parser.parse_args()


def instantiate_hipporag(args):
    global hipporag, config
    save_dir = args.save_dir
    dataset_name = args.dataset
    save_dir = save_dir + '_' + dataset_name

    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    azure_endpoint = args.azure_endpoint
    azure_embedding_endpoint = args.azure_embedding_endpoint

    corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    # Prepare datasets and evaluation
    samples = json.load(open(f"reproduce/dataset/{dataset_name}.json", "r"))
    all_queries = [s['question'] for s in samples]

    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert len(all_queries) == len(gold_docs) == len(gold_answers), "Length of queries, gold_docs, and gold_answers should be the same."
    except:
        gold_docs = None

        

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        azure_endpoint=azure_endpoint,
        azure_embedding_endpoint=azure_embedding_endpoint,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=10,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode=args.openie_mode
    )

    logging.basicConfig(level=logging.INFO)

    hipporag = HippoRAG(global_config=config)

    hipporag.index(docs)
    return hipporag

if __name__ == "__main__":
    args = parse_args()
    hipporag = instantiate_hipporag(args)
    mcp.run()
