from _shared import run_lifecycle


if __name__ == "__main__":
    run_lifecycle(save_dir="outputs/openai_test", llm_model_name="gpt-4o-mini", embedding_model_name="text-embedding-3-small")
