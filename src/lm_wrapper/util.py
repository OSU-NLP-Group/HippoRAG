def init_embedding_model(model_name):
    if 'GritLM/' in model_name:
        from src.lm_wrapper.gritlm import GritWrapper
        return GritWrapper(model_name)
    elif model_name not in ['colbertv2', 'bm25']:
        from src.lm_wrapper.huggingface_util import HuggingFaceWrapper
        return HuggingFaceWrapper(model_name)  # HuggingFace model for retrieval
