import argparse

from hipporag.llm import _get_llm_class
from hipporag.utils.config_utils import BaseConfig


def main():
    parser = argparse.ArgumentParser(description="Send one request through an OpenAI-compatible gateway.")
    parser.add_argument("--llm_name", required=True, help="<vendor/model> as the gateway names it (or a registered <prefix><vendor/model>).")
    parser.add_argument("--llm_base_url", default=None, help="Gateway base URL, e.g. https://example.com/v1.")
    parser.add_argument("--llm_api_key_env", default=None, help="Environment variable holding the gateway API key.")
    args = parser.parse_args()

    config = BaseConfig(
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        llm_api_key_env=args.llm_api_key_env,
        save_dir="outputs/gateway",
    )
    llm = _get_llm_class(config)
    message, metadata, cached = llm.infer([{"role": "user", "content": "Reply with exactly: HippoRAG gateway test passed"}])
    print(message)
    print({"metadata": metadata, "cached": cached})


if __name__ == "__main__":
    main()
