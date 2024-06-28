import numpy as np

from src.lm_wrapper import EmbeddingModelWrapper


class ColBERTv2Wrapper(EmbeddingModelWrapper):
    def __init__(self):
        pass

    def encode_text(self, text, instruction: str, norm: bool, return_cpu: bool, return_numpy: bool) -> np.ndarray:
        raise NotImplementedError
