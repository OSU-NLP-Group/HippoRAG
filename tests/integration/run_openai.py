import argparse
import os

from _shared import run_lifecycle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the OpenAI integration test")
    parser.add_argument(
        "--llm_model_name",
        default=os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        help="LLM model name (env: OPENAI_LLM_MODEL; default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--embedding_model_name",
        default=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        help="Embedding model name (env: OPENAI_EMBEDDING_MODEL; default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--save_dir",
        default="outputs/openai_test",
        help="HippoRAG working directory (default: outputs/openai_test)",
    )
    args = parser.parse_args()
    run_lifecycle(
        save_dir=args.save_dir,
        llm_model_name=args.llm_model_name,
        embedding_model_name=args.embedding_model_name,
    )
