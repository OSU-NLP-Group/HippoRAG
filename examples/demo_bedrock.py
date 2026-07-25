from _shared import run_demo


if __name__ == "__main__":
    run_demo(save_dir="outputs/bedrock", llm_model_name="bedrock/anthropic.claude-3-5-haiku-20241022-v1:0", embedding_model_name="cohere.embed-multilingual-v3")
