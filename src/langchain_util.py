import os

import tiktoken

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


def num_tokens_by_tiktoken(text: str):
    return len(enc.encode(text))


def init_langchain_model(llm: str, model_name: str, temperature: float = 0.0, max_retries=5, timeout=60, **kwargs):
    """
    Initialize a language model from the langchain library.
    :param llm: The LLM to use, e.g., 'openai', 'together'
    :param model_name: The model name to use, e.g., 'gpt-3.5-turbo'
    """
    if llm == 'openai':
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model=model_name, temperature=temperature, max_retries=max_retries, timeout=timeout, **kwargs)
    elif llm == 'together':
        from langchain_together import ChatTogether
        return ChatTogether(api_key=os.environ.get("TOGETHER_API_KEY"), model=model_name, temperature=temperature, **kwargs)
    elif llm == 'llama_cpp':
        from langchain_community.llms import LlamaCpp
        return LlamaCpp(model_path=model_name, temperature=temperature, **kwargs)
    else:
        # add any LLMs you want to use here using LangChain
        raise NotImplementedError(f"LLM '{llm}' not implemented yet.")
