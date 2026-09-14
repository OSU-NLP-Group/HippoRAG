from _shared import run_lifecycle


if __name__ == "__main__":
    run_lifecycle(
        save_dir="outputs/atlascloud_test",
        llm_model_name="atlascloud/deepseek-ai/deepseek-v3.2",
        embedding_model_name="text-embedding-3-small",
    )
