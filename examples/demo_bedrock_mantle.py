from hipporag.llm import _get_llm_class
from hipporag.utils.config_utils import BaseConfig


def main():
    config = BaseConfig(
        llm_name="bedrock-mantle/openai.gpt-5.5",
        llm_base_url="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
        save_dir="outputs/bedrock-mantle",
    )
    llm = _get_llm_class(config)
    message, metadata, cached = llm.infer([{"role": "user", "content": "Reply with exactly: HippoRAG Bedrock test passed"}])
    print(message)
    print({"metadata": metadata, "cached": cached})


if __name__ == "__main__":
    main()
