from typing import List
import numpy as np
from tqdm import tqdm

from .base import BaseEmbeddingModel
from ..utils.config_utils import BaseConfig
from ..prompts.linking import get_query_instruction
import requests

class VLLMEmbeddingModel(BaseEmbeddingModel):
    """
    To select this implementation you can initialise HippoRAG with:
        embedding_model_name starts with "VLLM/"
    The embedding base url should contain the v1/embeddings.
    """
    def __init__(self, global_config:BaseConfig, embedding_model_name:str) -> None:
        super().__init__(global_config=global_config)

        self.model_id = embedding_model_name[len("VLLM/"):]
        self.embedding_type = 'float'
        self.batch_size = 32

        self.url = global_config.embedding_base_url

        self.search_query_instr = set([
            get_query_instruction('query_to_fact'),
            get_query_instruction('query_to_passage')
        ])

    def call_model(self, input_text) -> List[np.ndarray]:
        if isinstance(input_text, str):
            input_text = [input_text]
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "input": input_text,
        }

        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return np.array([result["data"][i]["embedding"] for i in range(len(result["data"]))])

    def encode(self, texts: List[str]) -> np.array:
        response = self.call_model(texts)
        return response

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if len(texts) < self.batch_size:
            return self.encode(texts)
        
        results = []
        batch_indexes = list(range(0, len(texts), self.batch_size))
        for i in tqdm(batch_indexes, desc="Batch Encoding"):
            results.append(self.encode(texts[i:i + self.batch_size]))
        return np.concatenate(results)
