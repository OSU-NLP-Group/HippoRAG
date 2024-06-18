import numpy as np


class EmbeddingModelWrapper:
    def encode_text(self, text, instruction: str, norm: bool, return_cpu: bool, return_numpy: bool) -> np.ndarray:
        raise NotImplementedError
