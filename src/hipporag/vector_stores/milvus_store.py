"""
Milvus-backed embedding store for HippoRAG.

Requires:
    pip install "pymilvus[milvus_lite]"

Supports Milvus Lite, self-hosted Milvus, and Zilliz Cloud:

    # Local Milvus Lite, stored inside save_dir
    HippoRAG(global_config=BaseConfig(vector_store_type="milvus"))

    # Remote Milvus or Zilliz Cloud
    HippoRAG(global_config=BaseConfig(vector_store_type="milvus",
                                     milvus_uri="http://localhost:19530",
                                     milvus_token="<token>"))
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np

from ..embedding_store import BaseEmbeddingStore, compute_mdhash_id

logger = logging.getLogger(__name__)

_ID_FIELD = "id"
_CONTENT_FIELD = "content"
_VECTOR_FIELD = "vector"
_HASH_ID_MAX_LENGTH = 512
_CONTENT_MAX_LENGTH = 65535
_DEFAULT_BATCH_SIZE = 1000
_VALID_CONSISTENCY_LEVELS = {"Strong", "Session", "Bounded", "Eventually"}


def _safe_collection_name(db_path: str, namespace: str) -> str:
    path_hash = hashlib.md5(os.path.abspath(db_path).encode()).hexdigest()[:16]
    safe_namespace = re.sub(r"[^0-9A-Za-z_]", "_", namespace).strip("_") or "default"
    return f"hipporag_{path_hash}_{safe_namespace[:48]}"


def _is_local_uri(uri: str) -> bool:
    return "://" not in uri


def _get_config_value(global_config, attr_name: str, env_name: str, default=None):
    value = getattr(global_config, attr_name, None) if global_config is not None else None
    if value is not None:
        return value
    return os.getenv(env_name, default)


def _build_local_milvus_uri(db_path: str) -> str:
    os.makedirs(db_path, exist_ok=True)
    return os.path.join(db_path, "milvus_lite.db")


def _get_milvus_client(uri: str, token: Optional[str] = None, db_name: Optional[str] = None):
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise ImportError(
            "pymilvus is required for the Milvus backend. "
            'Install it with: pip install "pymilvus[milvus_lite]"'
        ) from exc

    if _is_local_uri(uri):
        local_dir = os.path.dirname(os.path.abspath(uri))
        os.makedirs(local_dir, exist_ok=True)

    client_kwargs = {"uri": uri}
    if token:
        client_kwargs["token"] = token
    if db_name:
        client_kwargs["db_name"] = db_name

    try:
        return MilvusClient(**client_kwargs)
    except Exception as exc:
        if _is_local_uri(uri) and "milvus-lite is required" in str(exc).lower():
            raise ImportError(
                "Milvus Lite requires the milvus-lite extra. "
                'Install it with: pip install "pymilvus[milvus_lite]"'
            ) from exc
        raise


def _get_field(fields: Iterable[Dict[str, Any]], field_name: str) -> Optional[Dict[str, Any]]:
    for field in fields:
        if field.get("name") == field_name:
            return field
    return None


def _get_vector_dim(field: Optional[Dict[str, Any]]) -> Optional[int]:
    if not field:
        return None
    params = field.get("params") or {}
    dim = params.get("dim", field.get("dim"))
    return int(dim) if dim is not None else None


class MilvusEmbeddingStore(BaseEmbeddingStore):
    """
    Embedding store backed by Milvus.

    Each (db_path, namespace) pair maps to a separate Milvus collection so
    chunk, entity, and fact embeddings remain isolated.
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

        configured_uri = _get_config_value(
            global_config,
            "milvus_uri",
            "MILVUS_URI",
        )
        self.milvus_uri = configured_uri or _build_local_milvus_uri(db_path)
        self.milvus_token = _get_config_value(global_config, "milvus_token", "MILVUS_TOKEN")
        self.milvus_db_name = _get_config_value(global_config, "milvus_db_name", "MILVUS_DB_NAME")
        self.milvus_consistency_level = _get_config_value(
            global_config,
            "milvus_consistency_level",
            "MILVUS_CONSISTENCY_LEVEL",
        )
        if (
            self.milvus_consistency_level is not None
            and self.milvus_consistency_level not in _VALID_CONSISTENCY_LEVELS
        ):
            raise ValueError(
                "milvus_consistency_level must be one of "
                f"{sorted(_VALID_CONSISTENCY_LEVELS)}."
            )

        self.collection_name = _safe_collection_name(db_path, namespace)
        self.client = _get_milvus_client(
            uri=self.milvus_uri,
            token=self.milvus_token,
            db_name=self.milvus_db_name,
        )

        self.text_to_hash_id: Dict[str, str] = {}
        self._hash_id_to_text: Dict[str, str] = {}
        self._hash_id_to_row: Dict[str, Dict] = {}

        if self.client.has_collection(collection_name=self.collection_name):
            self._validate_collection()
            self._load_caches()
        else:
            logger.info(
                "Milvus collection '%s' will be created on first insert.",
                self.collection_name,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_collection(self, expected_dim: Optional[int] = None) -> None:
        description = self.client.describe_collection(collection_name=self.collection_name)
        fields = description.get("fields") or []
        id_field = _get_field(fields, _ID_FIELD)
        content_field = _get_field(fields, _CONTENT_FIELD)
        vector_field = _get_field(fields, _VECTOR_FIELD)

        if id_field is None or content_field is None or vector_field is None:
            raise ValueError(
                f"Milvus collection '{self.collection_name}' does not match the "
                "HippoRAG schema. Expected id, content, and vector fields."
            )

        existing_dim = _get_vector_dim(vector_field)
        if expected_dim is not None and existing_dim is not None and existing_dim != expected_dim:
            raise ValueError(
                f"Milvus collection '{self.collection_name}' has vector dim "
                f"{existing_dim}, but this embedding model produced dim {expected_dim}."
            )

    def _ensure_collection(self, dim: int) -> None:
        if self.client.has_collection(collection_name=self.collection_name):
            self._validate_collection(expected_dim=dim)
            return

        from pymilvus import DataType

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name=_ID_FIELD,
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=_HASH_ID_MAX_LENGTH,
        )
        schema.add_field(
            field_name=_CONTENT_FIELD,
            datatype=DataType.VARCHAR,
            max_length=_CONTENT_MAX_LENGTH,
        )
        schema.add_field(field_name=_VECTOR_FIELD, datatype=DataType.FLOAT_VECTOR, dim=dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=_VECTOR_FIELD,
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        collection_kwargs = {
            "collection_name": self.collection_name,
            "schema": schema,
            "index_params": index_params,
        }
        if self.milvus_consistency_level:
            collection_kwargs["consistency_level"] = self.milvus_consistency_level

        self.client.create_collection(**collection_kwargs)
        logger.info(
            "Created Milvus collection '%s' with dim=%s.",
            self.collection_name,
            dim,
        )

    def _iter_records(self, output_fields: List[str]) -> Iterable[Dict[str, Any]]:
        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            batch_size=max(self.batch_size, _DEFAULT_BATCH_SIZE),
            output_fields=output_fields,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                yield from batch
        finally:
            iterator.close()

    def _load_caches(self) -> None:
        for row in self._iter_records([_ID_FIELD, _CONTENT_FIELD]):
            hash_id = row[_ID_FIELD]
            content = row[_CONTENT_FIELD]
            self._hash_id_to_text[hash_id] = content
            self._hash_id_to_row[hash_id] = {"hash_id": hash_id, "content": content}
            self.text_to_hash_id[content] = hash_id

        logger.info(
            "Loaded %s records from Milvus collection '%s'.",
            len(self._hash_id_to_row),
            self.collection_name,
        )

    # ------------------------------------------------------------------
    # BaseEmbeddingStore interface
    # ------------------------------------------------------------------

    def insert_strings(self, texts: List[str]) -> None:
        nodes_dict = {
            compute_mdhash_id(text, prefix=self.namespace + "-"): text for text in texts
        }
        missing_ids = [hash_id for hash_id in nodes_dict if hash_id not in self._hash_id_to_row]

        logger.info(
            "Inserting %s new records, %s already exist.",
            len(missing_ids),
            len(nodes_dict) - len(missing_ids),
        )
        if not missing_ids:
            return

        texts_to_encode = [nodes_dict[hash_id] for hash_id in missing_ids]
        embeddings = self.embedding_model.batch_encode(texts_to_encode)
        dim = len(embeddings[0]) if hasattr(embeddings[0], "__len__") else embeddings.shape[1]
        self._ensure_collection(dim=dim)

        records = [
            {
                _ID_FIELD: hash_id,
                _CONTENT_FIELD: text,
                _VECTOR_FIELD: np.array(embedding).astype(np.float32).tolist(),
            }
            for hash_id, text, embedding in zip(missing_ids, texts_to_encode, embeddings)
        ]

        for i in range(0, len(records), self.batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                data=records[i : i + self.batch_size],
            )

        for hash_id, text in zip(missing_ids, texts_to_encode):
            self._hash_id_to_text[hash_id] = text
            self._hash_id_to_row[hash_id] = {"hash_id": hash_id, "content": text}
            self.text_to_hash_id[text] = hash_id

        logger.info("Upserted %s records to '%s'.", len(records), self.collection_name)

    def delete(self, hash_ids) -> None:
        hash_ids = list(hash_ids)
        if not hash_ids or not self.client.has_collection(collection_name=self.collection_name):
            return

        self.client.delete(collection_name=self.collection_name, ids=hash_ids)

        for hash_id in hash_ids:
            text = self._hash_id_to_text.pop(hash_id, None)
            self._hash_id_to_row.pop(hash_id, None)
            if text is not None:
                self.text_to_hash_id.pop(text, None)

        logger.info("Deleted %s records from '%s'.", len(hash_ids), self.collection_name)

    def get_row(self, hash_id: str) -> Dict:
        return self._hash_id_to_row[hash_id]

    def get_rows(self, hash_ids: List[str]) -> Dict[str, Dict]:
        return {
            hash_id: self._hash_id_to_row[hash_id]
            for hash_id in hash_ids
            if hash_id in self._hash_id_to_row
        }

    def get_all_ids(self) -> List[str]:
        return list(self._hash_id_to_row.keys())

    def get_all_id_to_rows(self) -> Dict[str, Dict]:
        return deepcopy(self._hash_id_to_row)

    def get_all_texts(self) -> Set[str]:
        return set(self._hash_id_to_text.values())

    def get_embedding(self, hash_id: str, dtype=np.float32) -> np.ndarray:
        if not self.client.has_collection(collection_name=self.collection_name):
            raise KeyError(f"hash_id '{hash_id}' not found in Milvus.")

        result = self.client.get(
            collection_name=self.collection_name,
            ids=[hash_id],
            output_fields=[_VECTOR_FIELD],
        )
        if not result:
            raise KeyError(f"hash_id '{hash_id}' not found in Milvus.")
        return np.array(result[0][_VECTOR_FIELD], dtype=dtype)

    def get_embeddings(self, hash_ids: List[str], dtype=np.float32) -> np.ndarray:
        if not hash_ids:
            return np.array([])
        if not self.client.has_collection(collection_name=self.collection_name):
            raise KeyError("Milvus collection has not been created yet.")

        id_to_embedding: Dict[str, Any] = {}
        for i in range(0, len(hash_ids), self.batch_size):
            batch_ids = hash_ids[i : i + self.batch_size]
            records = self.client.get(
                collection_name=self.collection_name,
                ids=batch_ids,
                output_fields=[_ID_FIELD, _VECTOR_FIELD],
            )
            for record in records:
                id_to_embedding[record[_ID_FIELD]] = record[_VECTOR_FIELD]

        return np.array([id_to_embedding[hash_id] for hash_id in hash_ids], dtype=dtype)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            close()
