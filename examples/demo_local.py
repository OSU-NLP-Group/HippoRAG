from _shared import run_demo


if __name__ == "__main__":
    run_demo(save_dir="outputs/demo_llama", llm_model_name="meta-llama/Llama-3.1-8B-Instruct", embedding_model_name="GritLM/GritLM-7B", llm_base_url="http://localhost:6578/v1")
