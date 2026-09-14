from hipporag.llm import _get_llm_class
from hipporag.utils.config_utils import BaseConfig


def main():
    config = BaseConfig(
        llm_name="atlascloud/deepseek-ai/deepseek-v3.2",
        save_dir="outputs/atlascloud",
    )
    llm = _get_llm_class(config)
    message, metadata, cached = llm.infer([{"role": "user", "content": "Reply with exactly: HippoRAG Atlas Cloud test passed"}])
    print(message)
    print({"metadata": metadata, "cached": cached})


if __name__ == "__main__":
    main()
