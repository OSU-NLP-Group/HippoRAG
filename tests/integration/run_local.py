from _shared import run_lifecycle


if __name__ == "__main__":
    run_lifecycle(save_dir="outputs/local_test", llm_model_name="meta-llama/Llama-3.1-8B-Instruct", embedding_model_name="nvidia/NV-Embed-v2", llm_base_url="http://localhost:6578/v1")
