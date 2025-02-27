import json
from dataclasses import dataclass, field, asdict
from typing import (
    Optional,
    Tuple,
    Any, 
    Dict,
    List
)
import numpy as np
import threading
import multiprocessing


from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig


logger = get_logger(__name__)


@dataclass
class EmbeddingConfig:
    _data: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __getattr__(self, key: str) -> Any:
        # Define patterns to ignore for Jupyter/IPython-related attributes
        ignored_prefixes = ("_ipython_", "_repr_")
        if any(key.startswith(prefix) for prefix in ignored_prefixes):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
        if key in self._data:
            return self._data[key]
        
        logger.error(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")


    def __setattr__(self, key: str, value: Any) -> None:
        if key == '_data':
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __delattr__(self, key: str) -> None:
        if key in self._data:
            del self._data[key]
        else:
            logger.error(f"'{self.__class__.__name__}' object has no attribute '{key}'")
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style key lookup."""
        if key in self._data:
            return self._data[key]
        logger.error(f"'{key}' not found in configuration.")
        raise KeyError(f"'{key}' not found in configuration.")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-style key assignment."""
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Allow dict-style key deletion."""
        if key in self._data:
            del self._data[key]
        else:
            logger.error(f"'{key}' not found in configuration.")
            raise KeyError(f"'{key}' not found in configuration.")

    def __contains__(self, key: str) -> bool:
        """Allow usage of 'in' to check for keys."""
        return key in self._data
    
    
    def batch_upsert(self, updates: Dict[str, Any]) -> None:
        """Update existing attributes or add new ones from the given dictionary."""
        self._data.update(updates)

    def to_dict(self) -> Dict[str, Any]:
        """Export the configuration as a JSON-serializable dictionary."""
        return self._data

    def to_json(self) -> str:
        """Export the configuration as a JSON string."""
        return json.dumps(self._data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        """Create an LLMConfig instance from a dictionary."""
        instance = cls()
        instance.batch_upsert(config_dict)
        return instance

    @classmethod
    def from_json(cls, json_str: str) -> "LLMConfig":
        """Create an LLMConfig instance from a JSON string."""
        instance = cls()
        instance.batch_upsert(json.loads(json_str))
        return instance

    def __str__(self) -> str:
        """Provide a user-friendly string representation of the configuration."""
        return json.dumps(self._data, indent=4)
    

from filelock import FileLock
import sqlite3
import hashlib
import os
import torch
def make_cache_embed(encode_func, cache_file_name, device):
    def wrapper(**kwargs):
        # FOCUS_KEYS = ["instruction", "prompts", "max_length"]
        instruction = kwargs.get("instruction", "")
        max_length = kwargs.get("max_length", "")

        hash_strs = []
        for prompt in kwargs['prompts']:
            key_str = json.dumps({
                "instruction": instruction,
                "promps": prompt,
                "max_length": max_length
            }, sort_keys=True, default=str)
            hash_strs.append(hashlib.sha256(key_str.encode("utf-8")).hexdigest())

        missed_prompts = []
        lock_file = cache_file_name + ".lock"

        with FileLock(lock_file):
            with sqlite3.connect(cache_file_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS embeddings (
                        hash TEXT PRIMARY KEY,
                        embedding BLOB
                    )
                ''')
                conn.commit()

                embeddings = []
                for i, hash_str in enumerate(hash_strs):
                    cursor.execute('SELECT embedding FROM embeddings WHERE hash = ?', (hash_str,))
                    result = cursor.fetchone()
                    if result:
                        # Convert the BLOB back to a NumPy array
                        emb = np.frombuffer(result[0], dtype=np.float32)
                        embeddings.append(emb)
                    else:
                        missed_prompts.append(i)
                        embeddings.append(None)

        if missed_prompts:
            # Update kwargs to include only the missed prompts.
            kwargs['prompts'] = [kwargs['prompts'][i] for i in missed_prompts]
            # Call the encoder function (which returns a 2D torch tensor)
            new_embeddings = encode_func(**kwargs)
            # Insert the new embeddings back into the correct positions.
            for idx, embedding in enumerate(new_embeddings):
                embeddings[missed_prompts[idx]] = embedding

            # Save the new embeddings to the cache.
            with FileLock(lock_file):
                with sqlite3.connect(cache_file_name) as conn:
                    cursor = conn.cursor()
                    for i in missed_prompts:
                        hash_str = hash_strs[i]
                        emb = embeddings[i]
                        # Convert torch tensor to numpy bytes if necessary.
                        if isinstance(emb, torch.Tensor):
                            emb_bytes = emb.cpu().numpy().tobytes()
                        else:
                            emb_bytes = emb.tobytes()
                        cursor.execute('INSERT INTO embeddings (hash, embedding) VALUES (?, ?)', (hash_str, emb_bytes))
                    conn.commit()

        # Convert all embeddings to torch tensors if they're not already
        final_embeddings = [
            emb if isinstance(emb, torch.Tensor) else torch.Tensor(emb.copy())
            for emb in embeddings
        ]

        final_embeddings = [emb.to(device) for emb in final_embeddings]
        # Return a 2D tensor where each row is an embedding.
        return torch.stack(final_embeddings)

    return wrapper
    
class BaseEmbeddingModel:
    global_config: BaseConfig
    embedding_model_name: str # Class name indicating which embedding model to use.
    embedding_config: EmbeddingConfig
    
    embedding_dim: int # Need subclass to init
    
    def __init__(self, global_config: Optional[BaseConfig] = None) -> None:
        if global_config is None: 
            logger.debug("global config is not given. Using the default ExperimentConfig instance.")
            self.global_config = BaseConfig()
        else: self.global_config = global_config
        logger.debug(f"Loading {self.__class__.__name__} with global_config: {asdict(self.global_config)}")
        
        
        self.embedding_model_name = self.global_config.embedding_model_name

        logger.debug(f"Init {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        raise NotImplementedError
    
    
    def get_query_doc_scores(self, query_vec: np.ndarray, doc_vecs: np.ndarray):
        # """
        # @param query_vec: query vector
        # @param doc_vecs: doc matrix
        # @return: a matrix of query-doc scores
        # """
        return np.dot(query_vec, doc_vecs.T)
    


class EmbeddingCache:
    """A multiprocessing-safe global cache for storing embeddings."""
    
    _manager = multiprocessing.Manager()
    _cache = _manager.dict()  # Shared dictionary for multiprocessing
    _lock = threading.Lock()  # Thread-safe lock for concurrent access

    @classmethod
    def get(cls, content):
        """Retrieve the embedding if cached."""
        return cls._cache.get(content)

    @classmethod
    def set(cls, content, embedding):
        """Store an embedding in the cache."""
        with cls._lock:  # Ensures thread safety
            cls._cache[content] = embedding

    @classmethod
    def contains(cls, content):
        """Check if the embedding exists in cache."""
        return content in cls._cache

    @classmethod
    def clear(cls):
        """Clear the entire cache."""
        with cls._lock:
            cls._cache.clear()