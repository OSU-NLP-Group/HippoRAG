import numpy as np

from src.llm.gritlm import GritWrapper
from src.llm.huggingface_emb import HuggingFaceWrapper


def init_embedding_model(model_name):
    if model_name not in ['colbertv2', 'bm25']:
        return HuggingFaceWrapper(model_name)  # HuggingFace model for retrieval
    elif 'GritLM/' in model_name:
        return GritWrapper(model_name)


class EmbeddingModelWrapper:
    def encode_text(self, text, instruction: str, norm: bool, return_cpu: bool, return_numpy: bool) -> np.ndarray:
        raise NotImplementedError
