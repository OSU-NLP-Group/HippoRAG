from typing import Union, List

import torch
from transformers import AutoModel, AutoTokenizer

from src.llm import EmbeddingModelWrapper
from src.processing import mean_pooling_embedding_with_normalization, mean_pooling_embedding


class HuggingFaceWrapper(EmbeddingModelWrapper):
    def __init__(self, model_name: str, device='cuda'):
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    def encode_text(self, text: Union[str, List], instruction=None, norm=True, return_cpu=False, return_numpy=False):
        with torch.no_grad():
            if norm:
                res = mean_pooling_embedding_with_normalization(text, self.tokenizer, self.model, self.device)
            else:
                res = mean_pooling_embedding(text, self.tokenizer, self.model, self.device)

        if return_cpu:
            res = res.to('cpu')
        if return_numpy:
            res = res.numpy()
        return res
