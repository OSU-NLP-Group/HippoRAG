import argparse
import re

from _shared import run_lifecycle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the OpenAI-compatible gateway integration test")
    parser.add_argument("--llm_model_name", required=True, help="<vendor/model> as the gateway names it (or a registered <prefix><vendor/model>).")
    parser.add_argument("--llm_base_url", default=None, help="Gateway base URL, e.g. https://example.com/v1.")
    parser.add_argument("--llm_api_key_env", default=None, help="Environment variable holding the gateway API key.")
    parser.add_argument("--embedding_model_name", default="text-embedding-3-small")
    parser.add_argument("--embedding_base_url", default=None, help="OpenAI-compatible embedding base URL.")
    parser.add_argument("--embedding_api_key_env", default=None, help="Environment variable holding the embedding API key.")
    args = parser.parse_args()
    run_lifecycle(
        save_dir=f"outputs/gateway_test/{re.sub(r'[^A-Za-z0-9._-]+', '_', args.llm_model_name)}",
        llm_model_name=args.llm_model_name,
        llm_base_url=args.llm_base_url,
        llm_api_key_env=args.llm_api_key_env,
        embedding_model_name=args.embedding_model_name,
        embedding_base_url=args.embedding_base_url,
        embedding_api_key_env=args.embedding_api_key_env,
    )
