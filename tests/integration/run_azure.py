import argparse

from _shared import run_lifecycle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Azure OpenAI integration test")
    parser.add_argument("--azure_endpoint", required=True)
    parser.add_argument("--azure_embedding_endpoint", required=True)
    args = parser.parse_args()
    run_lifecycle(save_dir="outputs/azure_test", llm_model_name="gpt-4o-mini", embedding_model_name="text-embedding-3-small", azure_endpoint=args.azure_endpoint, azure_embedding_endpoint=args.azure_embedding_endpoint)
