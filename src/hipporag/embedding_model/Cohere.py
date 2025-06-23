from typing import List
import json

import boto3
from botocore.exceptions import ClientError
import numpy as np
from tqdm import tqdm

from .base import BaseEmbeddingModel
from ..utils.config_utils import BaseConfig
from ..prompts.linking import get_query_instruction


class CohereEmbeddingModel(BaseEmbeddingModel):
    """
    To select this implementation you can initialise HippoRAG with:
        embedding_model_name="cohere.embed-english-v3"
    """
    def __init__(self, global_config:BaseConfig, embedding_model_name:str) -> None:
        super().__init__(global_config=global_config)

        self.model_id = embedding_model_name
        self.embedding_type = 'float'
        self.batch_size = 64

        self.bedrock_runtime = boto3.client(service_name='bedrock-runtime')

        self.search_query_instr = set([
            get_query_instruction('query_to_fact'),
            get_query_instruction('query_to_passage')
        ])

    def encode(self, texts: List[str], input_type) -> None:
        request = {
             'texts': texts,
             'input_type': input_type,
             'embedding_types': [self.embedding_type]
        }
        try:
            response = self.bedrock_runtime.invoke_model(
                body=json.dumps(request),
                modelId=self.model_id,
                accept='*/*',
                contentType='application/json'
            )
        except ClientError as err:
            raise Exception(f"A client error occurred: {err.response['Error']['Message']}")
        
        response = json.loads(response.get('body').read())
        return np.array(response['embeddings'][self.embedding_type])

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        input_type = 'search_query' if (kwargs.get("instruction") in self.search_query_instr) else 'search_document'

        if len(texts) < self.batch_size:
            return self.encode(texts, input_type)
        
        results = []
        batch_indexes = list(range(0, len(texts), self.batch_size))
        for i in tqdm(batch_indexes, desc="Batch Encoding"):
            results.append(self.encode(texts[i:i + self.batch_size], input_type))
        return np.concatenate(results)
