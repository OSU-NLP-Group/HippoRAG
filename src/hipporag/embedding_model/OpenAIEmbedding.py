from copy import deepcopy
from typing import List, Optional

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)

class OpenAI_Compatible_EmbeddingModel(BaseEmbeddingModel):
    """OpenAI compatible embedding model implementation."""

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        self.client = OpenAI(
            api_key=self.global_config.embedding_api_key,
            base_url=self.global_config.embedding_base_url
        )
        
        try:
            test_embedding = self.client.embeddings.create(
                model=self.embedding_model_name,
                input=["Test text"]
            )
            self.embedding_dim = len(test_embedding.data[0].embedding)
        except Exception:
            # Default embedding dimension for text-embedding-3-small is 1536
            self.embedding_dim = 1536

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "encode_params": {
                "batch_size": self.global_config.embedding_batch_size,
                "model": self.embedding_model_name,  # Use the model name for the API call
                "dimensions": getattr(self.global_config, "embedding_dimensions", None)  # Optional dimension parameter
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str], instruction: str = ""):
        """
        Encode a batch of texts using OpenAI's embedding API.
        
        Args:
            texts: List of strings to encode
            instruction: Optional instruction to prepend to each text (if supported)
            
        Returns:
            Numpy array of embeddings
        """
        # Prepare the input texts with instructions if provided
        input_texts = texts
        if instruction:
            input_texts = [f"{instruction}\n{text}" for text in texts]
        
        # Prepare parameters for the API call
        params = {
            "model": self.embedding_config.encode_params["model"],
            "input": input_texts,
        }
        
        # Add dimensions parameter if specified
        if self.embedding_config.encode_params.get("dimensions"):
            params["dimensions"] = self.embedding_config.encode_params["dimensions"]
            
        # Call the API and get embeddings
        response = self.client.embeddings.create(**params)
        
        # Extract embeddings from the response
        embeddings = np.array([item.embedding for item in response.data])
        
        return embeddings

    def batch_encode(self, texts: List[str], instruction: str = "", **kwargs) -> np.ndarray:
        """
        Encode texts in batches.
        
        Args:
            texts: List of strings to encode
            instruction: Optional instruction to prepend to each text
            **kwargs: Additional parameters to override default encoding parameters
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str): 
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: 
            params.update(kwargs)

        batch_size = params.pop("batch_size", 16)

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
        
        if len(texts) <= batch_size:
            results = self.encode(texts, instruction=instruction)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                batch_results = self.encode(texts[i:i + batch_size], instruction=instruction)
                results.append(batch_results)
                pbar.update(min(batch_size, len(texts) - i))
            pbar.close()
            results = np.vstack(results)

        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results 