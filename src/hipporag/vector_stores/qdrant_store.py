"""
Qdrant-backed embedding store for HippoRAG.

Requires:
    pip install qdrant-client

Supports both local (file-based) and remote Qdrant deployments:

    # Local — no server needed
    HippoRAG(..., vector_store_type="qdrant")

    # Remote
    HippoRAG(..., vector_store_type="qdrant",
              qdrant_url="http://localhost:6333",
              qdrant_api_key="<key>")
"""
from __future__ import annotations

import logging
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ..embedding_store import BaseEmbeddingStore, compute_mdhash_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: convert a HippoRAG hash-ID to a Qdrant-compatible UUID string.
# Qdrant point IDs must be unsigned integers or UUID v1/v4/v5 strings.
# We derive a deterministic UUID v5 from the hash_id so we never need to
# store a separate mapping from UUID → hash_id inside Qdrant.
# ---------------------------------------------------------------------------
_UUID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # uuid.NAMESPACE_URL


def _to_qdrant_id(hash_id: str) -> str:
    return str(uuid.uuid5(_UUID_NS, hash_id))


def _get_qdrant_client(global_config):
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError(
            "qdrant-client is required for the Qdrant backend. "
            "Install it with:  pip install qdrant-client"
        ) from exc

    url = getattr(global_config, "qdrant_url", None)
    api_key = getattr(global_config, "qdrant_api_key", None)

    if url:
        logger.info(f"Connecting to remote Qdrant at {url}")
        return QdrantClient(url=url, api_key=api_key)
    else:
        # Default: local file-based mode (no server required)
        return QdrantClient(path=getattr(global_config, "qdrant_path", None))


class QdrantEmbeddingStore(BaseEmbeddingStore):
    """
    Embedding store backed by Qdrant.

    Each (save_dir, namespace) pair maps to its own Qdrant collection so
    chunk / entity / fact embeddings stay isolated.
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

        # Derive a short unique collection name from path + namespace.
        # We hash the path so the name stays valid on all OSes (no colons, slashes, etc.)
        import hashlib
        path_hash = hashlib.md5(db_path.encode()).hexdigest()[:16]
        self.collection_name = f"hipporag_{path_hash}_{namespace}"

        # Build the client; for local mode use db_path as the storage dir
        if global_config and not getattr(global_config, "qdrant_url", None):
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise ImportError(
                    "qdrant-client is required for the Qdrant backend. "
                    "Install it with:  pip install qdrant-client"
                ) from exc
            import os
            qdrant_dir = os.path.join(db_path, "qdrant_storage")
            os.makedirs(qdrant_dir, exist_ok=True)
            self.client = QdrantClient(path=qdrant_dir)
        else:
            self.client = _get_qdrant_client(global_config)

        # In-memory caches (rebuilt from Qdrant on init)
        self.text_to_hash_id: Dict[str, str] = {}
        self._hash_id_to_text: Dict[str, str] = {}
        self._hash_id_to_row: Dict[str, Dict] = {}

        self._init_collection()
        self._load_caches()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_collection(self):
        """Create the Qdrant collection if it doesn't already exist."""
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            # Dimension is unknown until the first insert; we create the
            # collection lazily in _ensure_collection_for_dim().
            logger.info(
                f"Qdrant collection '{self.collection_name}' will be created on first insert."
            )

    def _ensure_collection(self, dim: int):
        """Create collection with the correct vector dimension if needed."""
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info(
                f"Created Qdrant collection '{self.collection_name}' with dim={dim}."
            )

    def _load_caches(self):
        """Scroll all points from Qdrant and rebuild in-memory dicts."""
        from qdrant_client.models import Filter

        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            return  # Nothing to load yet

        offset = None
        while True:
            result, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_vectors=False,
                with_payload=True,
            )
            for point in result:
                hash_id = point.payload["hash_id"]
                content = point.payload["content"]
                self._hash_id_to_text[hash_id] = content
                self._hash_id_to_row[hash_id] = {"hash_id": hash_id, "content": content}
                self.text_to_hash_id[content] = hash_id

            if next_offset is None:
                break
            offset = next_offset

        logger.info(
            f"Loaded {len(self._hash_id_to_row)} records from Qdrant "
            f"collection '{self.collection_name}'."
        )

    # ------------------------------------------------------------------
    # BaseEmbeddingStore interface
    # ------------------------------------------------------------------

    def insert_strings(self, texts: List[str]) -> None:
        from qdrant_client.models import PointStruct

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

        if len(missing_ids) > 0:
            dim = len(embeddings[0]) if hasattr(embeddings[0], "__len__") else embeddings.shape[1]
            self._ensure_collection(dim)

        points = [
            PointStruct(
                id=_to_qdrant_id(h),
                vector=np.array(emb).astype(np.float32).tolist(),
                payload={"hash_id": h, "content": text},
            )
            for h, text, emb in zip(missing_ids, texts_to_encode, embeddings)
        ]

        # Upsert in batches to avoid overloading the client
        for i in range(0, len(points), self.batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + self.batch_size],
            )

        # Update in-memory caches
        for h, text in zip(missing_ids, texts_to_encode):
            self._hash_id_to_text[h] = text
            self._hash_id_to_row[h] = {"hash_id": h, "content": text}
            self.text_to_hash_id[text] = h

        logger.info(f"Upserted {len(points)} points to '{self.collection_name}'.")

    def delete(self, hash_ids) -> None:
        from qdrant_client.models import PointIdsList

        hash_ids = list(hash_ids)
        qdrant_ids = [_to_qdrant_id(h) for h in hash_ids]

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=qdrant_ids),
        )

        for h in hash_ids:
            text = self._hash_id_to_text.pop(h, None)
            self._hash_id_to_row.pop(h, None)
            if text is not None:
                self.text_to_hash_id.pop(text, None)

        logger.info(f"Deleted {len(hash_ids)} points from '{self.collection_name}'.")

    def get_row(self, hash_id: str) -> Dict:
        return self._hash_id_to_row[hash_id]

    def get_rows(self, hash_ids: List[str]) -> Dict[str, Dict]:
        return {h: self._hash_id_to_row[h] for h in hash_ids if h in self._hash_id_to_row}

    def get_all_ids(self) -> List[str]:
        return list(self._hash_id_to_row.keys())

    def get_all_id_to_rows(self) -> Dict[str, Dict]:
        return deepcopy(self._hash_id_to_row)

    def get_all_texts(self) -> Set[str]:
        return set(self._hash_id_to_text.values())

    def get_embedding(self, hash_id: str, dtype=np.float32) -> np.ndarray:
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[_to_qdrant_id(hash_id)],
            with_vectors=True,
            with_payload=False,
        )
        if not result:
            raise KeyError(f"hash_id '{hash_id}' not found in Qdrant.")
        return np.array(result[0].vector, dtype=dtype)

    def close(self) -> None:
        """Release the Qdrant client's file lock so another instance can open the same path."""
        self.client.close()

    def get_embeddings(self, hash_ids: List[str], dtype=np.float32) -> np.ndarray:
        if not hash_ids:
            return np.array([])

        qdrant_ids = [_to_qdrant_id(h) for h in hash_ids]
        # Retrieve in batches for large collections
        all_results: Dict[str, Any] = {}
        for i in range(0, len(qdrant_ids), self.batch_size):
            batch = qdrant_ids[i : i + self.batch_size]
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=batch,
                with_vectors=True,
                with_payload=False,
            )
            for rec in records:
                all_results[rec.id] = rec.vector

        # Return in the same order as hash_ids
        ordered = [all_results[_to_qdrant_id(h)] for h in hash_ids]
        return np.array(ordered, dtype=dtype)
