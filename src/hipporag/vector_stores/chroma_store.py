"""
ChromaDB-backed embedding store for HippoRAG.

Requires:
    pip install chromadb

Supports both local (persistent file) and remote (HTTP) ChromaDB:

    # Local — no server needed
    HippoRAG(..., vector_store_type="chroma")

    # Remote
    HippoRAG(..., vector_store_type="chroma",
              chroma_host="localhost", chroma_port=8000)
"""
from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ..embedding_store import BaseEmbeddingStore, compute_mdhash_id

logger = logging.getLogger(__name__)


def _get_chroma_client(db_path: str, global_config):
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "chromadb is required for the Chroma backend. "
            "Install it with:  pip install chromadb"
        ) from exc

    host = getattr(global_config, "chroma_host", None)
    port = getattr(global_config, "chroma_port", 8000)

    if host:
        logger.info(f"Connecting to remote ChromaDB at {host}:{port}")
        return chromadb.HttpClient(host=host, port=port)
    else:
        chroma_dir = os.path.join(db_path, "chroma_storage")
        os.makedirs(chroma_dir, exist_ok=True)
        logger.info(f"Using local ChromaDB at {chroma_dir}")
        return chromadb.PersistentClient(path=chroma_dir)


class ChromaEmbeddingStore(BaseEmbeddingStore):
    """
    Embedding store backed by ChromaDB.

    ChromaDB natively supports arbitrary string IDs, which maps perfectly to
    HippoRAG's MD5-based hash IDs — no ID conversion needed.

    Each (db_path, namespace) pair gets its own ChromaDB collection.
    """

    def __init__(
        self,
        embedding_model,
        db_path: str,
        batch_size: int,
        namespace: str,
        global_config=None,
    ):
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace
        self.global_config = global_config

        self.client = _get_chroma_client(db_path, global_config)

        # ChromaDB collection name must be 3-63 chars, only [a-zA-Z0-9._-]
        # We use a sanitised version of the namespace.
        safe_ns = namespace.replace("-", "_")
        self.collection_name = f"hipporag_{safe_ns}"

        # We disable ChromaDB's built-in embedding — HippoRAG provides vectors.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

        # In-memory caches kept in sync with every insert/delete
        self.text_to_hash_id: Dict[str, str] = {}
        self._hash_id_to_row: Dict[str, Dict] = {}

        self._load_caches()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_caches(self):
        """Populate in-memory dicts from the existing ChromaDB collection."""
        try:
            # get() with no filter returns all records (may be slow on huge collections)
            result = self.collection.get(include=["documents"])
        except Exception as exc:
            logger.warning(f"Could not load existing Chroma records: {exc}")
            return

        for hash_id, content in zip(result["ids"], result["documents"]):
            self._hash_id_to_row[hash_id] = {"hash_id": hash_id, "content": content}
            self.text_to_hash_id[content] = hash_id

        logger.info(
            f"Loaded {len(self._hash_id_to_row)} records from ChromaDB "
            f"collection '{self.collection_name}'."
        )

    # ------------------------------------------------------------------
    # BaseEmbeddingStore interface
    # ------------------------------------------------------------------

    def insert_strings(self, texts: List[str]) -> None:
        nodes_dict = {
            compute_mdhash_id(t, prefix=self.namespace + "-"): t for t in texts
        }
        missing_ids = [h for h in nodes_dict if h not in self._hash_id_to_row]

        logger.info(
            f"Inserting {len(missing_ids)} new records, "
            f"{len(nodes_dict) - len(missing_ids)} already exist."
        )
        if not missing_ids:
            return

        texts_to_encode = [nodes_dict[h] for h in missing_ids]
        embeddings = self.embedding_model.batch_encode(texts_to_encode)

        # Upsert in batches
        for i in range(0, len(missing_ids), self.batch_size):
            batch_ids = missing_ids[i : i + self.batch_size]
            batch_texts = texts_to_encode[i : i + self.batch_size]
            batch_embs = embeddings[i : i + self.batch_size]

            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=[np.array(e).astype(np.float32).tolist() for e in batch_embs],
            )

        # Update caches
        for h, text in zip(missing_ids, texts_to_encode):
            self._hash_id_to_row[h] = {"hash_id": h, "content": text}
            self.text_to_hash_id[text] = h

        logger.info(f"Inserted {len(missing_ids)} records into '{self.collection_name}'.")

    def delete(self, hash_ids) -> None:
        hash_ids = list(hash_ids)
        self.collection.delete(ids=hash_ids)

        for h in hash_ids:
            row = self._hash_id_to_row.pop(h, None)
            if row is not None:
                self.text_to_hash_id.pop(row["content"], None)

        logger.info(f"Deleted {len(hash_ids)} records from '{self.collection_name}'.")

    def get_row(self, hash_id: str) -> Dict:
        return self._hash_id_to_row[hash_id]

    def get_rows(self, hash_ids: List[str]) -> Dict[str, Dict]:
        return {h: self._hash_id_to_row[h] for h in hash_ids if h in self._hash_id_to_row}

    def get_all_ids(self) -> List[str]:
        return list(self._hash_id_to_row.keys())

    def get_all_id_to_rows(self) -> Dict[str, Dict]:
        return deepcopy(self._hash_id_to_row)

    def get_all_texts(self) -> Set[str]:
        return {row["content"] for row in self._hash_id_to_row.values()}

    def get_embedding(self, hash_id: str, dtype=np.float32) -> np.ndarray:
        result = self.collection.get(ids=[hash_id], include=["embeddings"])
        if not result["ids"]:
            raise KeyError(f"hash_id '{hash_id}' not found in ChromaDB.")
        return np.array(result["embeddings"][0], dtype=dtype)

    def get_embeddings(self, hash_ids: List[str], dtype=np.float32) -> np.ndarray:
        if not hash_ids:
            return np.array([])

        # Retrieve in batches, then reassemble in the original order
        id_to_emb: Dict[str, np.ndarray] = {}
        for i in range(0, len(hash_ids), self.batch_size):
            batch = hash_ids[i : i + self.batch_size]
            result = self.collection.get(ids=batch, include=["embeddings"])
            for h, emb in zip(result["ids"], result["embeddings"]):
                id_to_emb[h] = np.array(emb, dtype=dtype)

        return np.array([id_to_emb[h] for h in hash_ids], dtype=dtype)
