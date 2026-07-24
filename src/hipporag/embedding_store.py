import numpy as np
import os
from abc import ABC, abstractmethod
from hashlib import md5
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import pandas as pd


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Return prefix + MD5 hex digest of content. Mirrors utils.misc_utils."""
    return prefix + md5(content.encode()).hexdigest()

logger = logging.getLogger(__name__)


class BaseEmbeddingStore(ABC):
    """
    Abstract base class for all embedding store backends.

    All backends must expose a ``text_to_hash_id`` dict-like attribute that
    maps raw text → its hash ID, because HippoRAG.delete() accesses it directly.
    """

    namespace: str
    embedding_model: Any
    batch_size: int

    # Must be kept in-sync by every subclass (populated on init, insert, delete)
    text_to_hash_id: Dict[str, str]

    # ------------------------------------------------------------------
    # Concrete helper — same logic regardless of backend
    # ------------------------------------------------------------------

    def get_missing_string_hash_ids(self, texts: List[str]) -> Dict[str, Dict]:
        """Return {hash_id: row} for texts not yet stored."""
        existing = set(self.get_all_ids())
        result = {}
        for text in texts:
            h = compute_mdhash_id(text, prefix=self.namespace + "-")
            if h not in existing:
                result[h] = {"hash_id": h, "content": text}
        return result

    def get_hash_id(self, text: str) -> str:
        return self.text_to_hash_id[text]

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def insert_strings(self, texts: List[str]) -> None: ...

    @abstractmethod
    def delete(self, hash_ids) -> None: ...

    @abstractmethod
    def get_row(self, hash_id: str) -> Dict: ...

    @abstractmethod
    def get_rows(self, hash_ids: List[str]) -> Dict[str, Dict]: ...

    @abstractmethod
    def get_all_ids(self) -> List[str]: ...

    @abstractmethod
    def get_all_id_to_rows(self) -> Dict[str, Dict]: ...

    @abstractmethod
    def get_all_texts(self) -> Set[str]: ...

    @abstractmethod
    def get_embedding(self, hash_id: str, dtype=np.float32) -> np.ndarray: ...

    @abstractmethod
    def get_embeddings(self, hash_ids: List[str], dtype=np.float32) -> List[np.ndarray]: ...

    def close(self) -> None:
        """Release any held resources (connections, file locks). No-op by default."""


class EmbeddingStore(BaseEmbeddingStore):
    """Default backend: stores embeddings in a local Parquet file."""

    def __init__(self, embedding_model, db_filename: str, batch_size: int, namespace: str):
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(db_filename, f"vdb_{self.namespace}.parquet")
        self._load_data()

    def get_missing_string_hash_ids(self, texts: List[str]):
        nodes_dict = {}
        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return {}

        existing = self.hash_id_to_row.keys()
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, texts: List[str]):
        nodes_dict = {}
        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return

        existing = self.hash_id_to_row.keys()
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(f"Inserting {len(missing_ids)} new records, "
                    f"{len(all_hash_ids) - len(missing_ids)} records already exist.")

        if not missing_ids:
            return

        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)
        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["content"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids = []
            self.texts = []
            self.embeddings = []
            self.hash_id_to_idx = {}
            self.hash_id_to_row = {}
            self.hash_id_to_text = {}
            self.text_to_hash_id = {}

    def _save_data(self):
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t}
            for h, t, e in zip(self.hash_ids, self.texts, self.embeddings)
        }
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        logger.info(f"Saving new records.")
        self._save_data()

    def delete(self, hash_ids):
        indices = [self.hash_id_to_idx[h] for h in hash_ids]
        sorted_indices = np.sort(indices)[::-1]
        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)
        logger.info(f"Saving record after deletion.")
        self._save_data()

    def get_row(self, hash_id: str) -> Dict:
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text: str) -> str:
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids: List[str], dtype=np.float32) -> Dict[str, Dict]:
        if not hash_ids:
            return {}
        return {id_: self.hash_id_to_row[id_] for id_ in hash_ids}

    def get_all_ids(self) -> List[str]:
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self) -> Dict[str, Dict]:
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self) -> Set[str]:
        return set(row['content'] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id: str, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)

    def get_embeddings(self, hash_ids: List[str], dtype=np.float32) -> List[np.ndarray]:
        if not hash_ids:
            return []
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]
        return embeddings


def get_embedding_store(
    embedding_model,
    db_path: str,
    batch_size: int,
    namespace: str,
    global_config=None,
) -> BaseEmbeddingStore:
    """
    Factory that returns the appropriate EmbeddingStore backend based on
    ``global_config.vector_store_type``.

    Defaults to the local Parquet backend when no config is supplied.
    """
    store_type = getattr(global_config, "vector_store_type", "parquet") if global_config else "parquet"

    if store_type == "parquet":
        return EmbeddingStore(embedding_model, db_path, batch_size, namespace)
    elif store_type == "qdrant":
        from .vector_stores.qdrant_store import QdrantEmbeddingStore
        return QdrantEmbeddingStore(embedding_model, db_path, batch_size, namespace, global_config)
    elif store_type == "chroma":
        from .vector_stores.chroma_store import ChromaEmbeddingStore
        return ChromaEmbeddingStore(embedding_model, db_path, batch_size, namespace, global_config)
    elif store_type == "milvus":
        from .vector_stores.milvus_store import MilvusEmbeddingStore
        return MilvusEmbeddingStore(embedding_model, db_path, batch_size, namespace, global_config)
    else:
        raise ValueError(
            f"Unknown vector_store_type: '{store_type}'. "
            f"Choose from 'parquet', 'qdrant', 'chroma', or 'milvus'."
        )
