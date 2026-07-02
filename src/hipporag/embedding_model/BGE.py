from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


def mean_pooling(token_embeddings, mask):
    """Mean pooling for BGE models (same as Contriever)."""
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


class BGEEmbeddingModel(BaseEmbeddingModel):
    """BGE (BAAI General Embedding) model implementation.

    Added for GraphRAG-Benchmark compatibility so that local BGE checkpoints
    (e.g. ``BAAI/bge-large-en-v1.5``) can be selected via ``embedding_model_name``.
    Mirrors the local-HF pattern of :class:`ContrieverModel`: ``AutoModel`` +
    mean pooling, exposed through the ``batch_encode`` interface that the rest
    of HippoRAG relies on. Asymmetric query/corpus encoding is supported by
    prepending an ``instruction`` to the input texts (BGE queries typically get
    an instruction, while corpus passages do not).
    """

    def __init__(self, global_config: Optional[BaseConfig] = None,
                 embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        # Initializing the embedding model
        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(**self.embedding_config.model_init_params)
        self.embedding_model.eval()
        self.embedding_dim = self.embedding_model.config.hidden_size

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                "torch_dtype": self.global_config.embedding_model_dtype,
                'device_map': "auto",  # added this line to use multiple GPUs
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str], instruction: str = "",
               max_length: Optional[int] = None) -> torch.Tensor:
        """
        Encode texts using the BGE model with mean pooling.

        Args:
            texts: List of texts to encode.
            instruction: Optional instruction prepended to every text. BGE uses
                asymmetric encoding, so queries usually receive an instruction
                while corpus passages do not.
            max_length: Max token length for tokenization. Falls back to the
                value configured in ``embedding_config.encode_params``.

        Returns:
            torch.Tensor: Encoded sentence embeddings (unnormalized). Normalization
            is applied later in :meth:`batch_encode` to keep behavior consistent
            with the other local embedding models.
        """
        if instruction:
            texts = [f"{instruction}{text}" for text in texts]

        if max_length is None:
            max_length = self.embedding_config.encode_params["max_length"]

        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.embedding_model.device)

        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output.last_hidden_state,
                                               encoded_input['attention_mask'])

        return sentence_embeddings

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Batched encoding entry point used by the rest of HippoRAG.

        Honors the ``instruction`` and ``norm`` keyword arguments that callers
        such as :meth:`HippoRAG.get_query_embeddings` pass in
        (``batch_encode(texts, instruction=..., norm=True)``).
        """
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        instruction = params.pop("instruction", "")
        # An explicit `norm` kwarg (e.g. norm=True from the retrieval path)
        # overrides the config default so callers can force normalization.
        norm = params.pop("norm", self.embedding_config.norm)
        batch_size = params.pop("batch_size", 16)
        max_length = params.get("max_length", None)

        logger.debug(f"Calling {self.__class__.__name__} with instruction='{instruction}', norm={norm}")
        if len(texts) <= batch_size:
            results = self.encode(texts, instruction=instruction, max_length=max_length)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                results.append(self.encode(texts[i:i + batch_size],
                                           instruction=instruction, max_length=max_length))
                pbar.update(batch_size)
            pbar.close()
            results = torch.cat(results, dim=0)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()

        if norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
