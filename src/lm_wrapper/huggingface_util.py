from typing import Union, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.lm_wrapper import EmbeddingModelWrapper
from src.processing import mean_pooling_embedding_with_normalization, mean_pooling_embedding


class HuggingFaceWrapper(EmbeddingModelWrapper):
    def __init__(self, model_name: str, device='cuda'):
        self.model_name = model_name
        self.model_name_processed = model_name.replace('/', '_').replace('.', '_')
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    def encode_text(self, text: Union[str, List], instruction=None, norm=True, return_cpu=False, return_numpy=False):
        encoding_func = mean_pooling_embedding_with_normalization if norm else mean_pooling_embedding
        with torch.no_grad():
            if isinstance(text, str):
                text = [text]
            res = []
            if len(text) > 1:
                for t in tqdm(text, total=len(text), desc=f"HF model {self.model_name} encoding"):
                    res.append(encoding_func(t, self.tokenizer, self.model, self.device))
            else:
                res = [encoding_func(text[0], self.tokenizer, self.model, self.device)]
            res = torch.stack(res)
            res = torch.squeeze(res, dim=1)

        if return_cpu:
            res = res.cpu()
        if return_numpy:
            res = res.numpy()
        return res

    def get_query_doc_scores(self, query_vec: np.ndarray, doc_vecs: np.ndarray):
        """

        @param query_vec: query vector
        @param doc_vecs: doc matrix
        @return: a matrix of query-doc scores
        """
        return np.dot(doc_vecs, query_vec.T)
