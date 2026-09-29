import ast
import json
import os
import logging
from dataclasses import asdict
from typing import Union, Optional, List, Set, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
import igraph as ig
import re
import time
from filelock import FileLock

from .llm import _get_llm_class, BaseLLM, find_gateway
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore, get_embedding_store
from .information_extraction import OpenIE
# VLLMOfflineOpenIE and TransformersOfflineOpenIE are imported lazily inside
# __init__ so that vllm (Linux-only) and heavy Transformers deps are never
# loaded unless the user explicitly requests offline OpenIE mode.
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import (
    Chunk,
    NerRawOutput,
    QuerySolution,
    RetrievalResult,
    TripleRawOutput,
    compute_mdhash_id,
    ensure_list_input,
    extract_entity_nodes,
    flatten_facts,
    min_max_normalize,
    reformat_openie_results,
    text_processing,
    validate_parallel_input_lengths,
)
from .preprocessing import BaseTextPreprocessor, TextPreprocessor
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig
from .utils.state_utils import StateConsistencyError, component_class_identity, embedding_index_identity, remove_sources_from_mapping, validate_or_create_index_manifest
from .utils.qa_utils import reason_step
from .utils.logging_utils import redact_config

logger = logging.getLogger(__name__)

class HippoRAG:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        resources = [
            getattr(self, "chunk_embedding_store", None),
            getattr(self, "entity_embedding_store", None),
            getattr(self, "fact_embedding_store", None),
        ]
        if getattr(self, "_owns_embedding_model", False):
            resources.append(getattr(self, "embedding_model", None))
        if getattr(self, "_owns_openie", False):
            resources.append(getattr(self, "openie", None))
        if getattr(self, "_owns_llm_model", False):
            resources.append(getattr(self, "llm_model", None))
        closed_ids = getattr(self, "_closed_resource_ids", set())
        for resource in resources:
            if resource is None or id(resource) in closed_ids:
                continue
            close = getattr(resource, "close", None)
            if callable(close):
                close()
            closed_ids.add(id(resource))
        self._closed_resource_ids = closed_ids

    def _construct_or_cleanup(self, factory):
        try:
            return factory()
        except Exception:
            self.close()
            raise

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None,
                 extraction_llm: BaseLLM = None,
                 qa_llm: BaseLLM = None,
                 embedding_model: BaseEmbeddingModel = None,
                 text_preprocessor: BaseTextPreprocessor = None,
                 azure_api_version=None,
                 azure_chat_deployment=None,
                 azure_embedding_api_version=None,
                 azure_embedding_deployment=None,
                 index_identity=None,
                 embedding_provider=None,
                 llm_api_key_env=None,
                 embedding_api_key_env=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific HippoRAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
            llm_api_key_env: Environment variable holding the LLM API key for an OpenAI-compatible endpoint.
            embedding_api_key_env: Environment variable holding the API key for an OpenAI-compatible embedding endpoint.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if llm_api_key_env is not None:
            self.global_config.llm_api_key_env = llm_api_key_env

        if embedding_api_key_env is not None:
            self.global_config.embedding_api_key_env = embedding_api_key_env

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        if azure_api_version is not None:
            self.global_config.azure_api_version = azure_api_version

        if azure_chat_deployment is not None:
            self.global_config.azure_chat_deployment = azure_chat_deployment

        if azure_embedding_api_version is not None:
            self.global_config.azure_embedding_api_version = azure_embedding_api_version

        if azure_embedding_deployment is not None:
            self.global_config.azure_embedding_deployment = azure_embedding_deployment

        if embedding_provider is not None:
            self.global_config.embedding_provider = embedding_provider

        self.global_config.validate()
        if index_identity is not None and (not isinstance(index_identity, str) or not index_identity.strip()):
            raise ValueError("index_identity must be a non-empty string when provided.")
        self.index_identity = index_identity
        injected_index_producer = embedding_model is not None or extraction_llm is not None or text_preprocessor is not None or (qa_llm is not None and extraction_llm is None)
        if index_identity is None and injected_index_producer:
            raise ValueError("index_identity is required when injecting an embedding model, extraction LLM, or text preprocessor.")
        self.text_preprocessor = text_preprocessor or TextPreprocessor()

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in redact_config(asdict(self.global_config)).items()])
        logger.debug(f"HippoRAG init with config:\n  {_print_config}\n")

        #LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self._owns_openie = False
        if self.global_config.openie_mode == 'offline':
            self._owns_llm_model = False
            self.llm_model = extraction_llm or qa_llm
            self.extraction_llm = extraction_llm or self.llm_model
            self.qa_llm = qa_llm or self.llm_model
            from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
            self.openie = self._construct_or_cleanup(lambda: VLLMOfflineOpenIE(self.global_config))
            self._owns_openie = True
        else:
            self._owns_llm_model = extraction_llm is None and qa_llm is None
            self.llm_model: BaseLLM = extraction_llm or qa_llm or self._construct_or_cleanup(lambda: _get_llm_class(self.global_config))
            self.extraction_llm: BaseLLM = extraction_llm or self.llm_model
            self.qa_llm: BaseLLM = qa_llm or self.llm_model

        if self.global_config.openie_mode == 'online':
            self.openie = self._construct_or_cleanup(
                lambda: OpenIE(
                    llm_model=self.extraction_llm,
                    max_workers=self.global_config.openie_max_workers,
                    ner_max_tokens=self.global_config.openie_ner_max_tokens,
                    triple_max_tokens=self.global_config.openie_triple_max_tokens,
                )
            )
            self._owns_openie = True
        elif self.global_config.openie_mode == 'Transformers-offline':
            from .information_extraction.openie_transformers_offline import TransformersOfflineOpenIE
            self.openie = self._construct_or_cleanup(lambda: TransformersOfflineOpenIE(self.global_config, shared_llm=self.extraction_llm))
            self._owns_openie = True
        elif self.global_config.openie_mode != 'offline':
            raise ValueError(f"Unsupported openie_mode: {self.global_config.openie_mode}")

        self.graph = self._construct_or_cleanup(self.initialize_graph)

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
            self._owns_embedding_model = False
        else:
            self._owns_embedding_model = embedding_model is None
            self.embedding_model: BaseEmbeddingModel = embedding_model or self._construct_or_cleanup(
                lambda: _get_embedding_model_class(
                    embedding_model_name=self.global_config.embedding_model_name,
                    provider=self.global_config.embedding_provider,
                )(
                    global_config=self.global_config,
                    embedding_model_name=self.global_config.embedding_model_name,
                )
            )
        self.chunk_embedding_store = self._construct_or_cleanup(
            lambda: get_embedding_store(
                self.embedding_model,
                os.path.join(self.working_dir, "chunk_embeddings"),
                self.global_config.embedding_batch_size,
                'chunk',
                self.global_config,
            )
        )
        self.entity_embedding_store = self._construct_or_cleanup(
            lambda: get_embedding_store(
                self.embedding_model,
                os.path.join(self.working_dir, "entity_embeddings"),
                self.global_config.embedding_batch_size,
                'entity',
                self.global_config,
            )
        )
        self.fact_embedding_store = self._construct_or_cleanup(
            lambda: get_embedding_store(
                self.embedding_model,
                os.path.join(self.working_dir, "fact_embeddings"),
                self.global_config.embedding_batch_size,
                'fact',
                self.global_config,
            )
        )
        self.index_manifest_path = os.path.join(self.working_dir, "index_manifest.json")
        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')
        self.openie_state_path = os.path.join(self.working_dir, "openie_state.json")
        self.chunk_metadata_path = os.path.join(self.working_dir, "chunk_metadata.json")
        if self.global_config.openie_mode != 'offline':
            try:
                self._validate_or_create_index_manifest()
            except Exception:
                self.close()
                raise

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.rerank_filter = None if self.global_config.openie_mode == 'offline' else self._construct_or_cleanup(lambda: DSPyFilter(self))

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None
        self._openie_info = None
        self._openie_provenance = None
        self.chunk_metadata = self._construct_or_cleanup(self._load_chunk_metadata)

    def _validate_or_create_index_manifest(self) -> None:
        manifest = {
            "schema_version": 4,
            "rag_type": "hipporag",
            "text_normalization": "unicode_alnum_casefold_v1",
            "embedding": embedding_index_identity(self.global_config),
            "components": {
                "embedding_model": component_class_identity(self.embedding_model) if self.embedding_model is not None else None,
                "extraction_llm": component_class_identity(self.extraction_llm),
                "text_preprocessor": component_class_identity(self.text_preprocessor),
                "explicit_identity": self.index_identity,
            },
            "openie": self._openie_provenance_for_manifest(),
            "graph_construction": {
                "synonymy_edge_topk": self.global_config.synonymy_edge_topk,
                "synonymy_edge_sim_threshold": self.global_config.synonymy_edge_sim_threshold,
                "synonym_algorithm_schema": "cosine_knn_v1",
            },
            "is_directed_graph": self.global_config.is_directed_graph,
        }
        validate_or_create_index_manifest(
            self.index_manifest_path,
            manifest,
            (self.chunk_embedding_store, self.entity_embedding_store, self.fact_embedding_store),
            legacy_state_paths=tuple(
                path for path in (getattr(self, "_graph_pickle_filename", None), getattr(self, "chunk_metadata_path", None)) if path
            ),
            allow_empty_rewrite=self.global_config.force_openie_from_scratch,
        )

    def _openie_state_identity(self) -> Dict[str, Any]:
        return {
            "model_name": self.global_config.llm_name,
            "azure_endpoint": self.global_config.azure_endpoint,
            "azure_api_version": self.global_config.azure_api_version,
            "azure_deployment": (self.global_config.azure_chat_deployment or self.global_config.llm_name) if self.global_config.azure_endpoint else None,
            "temperature": self.global_config.temperature,
            "seed": self.global_config.seed,
            "response_format": self.global_config.response_format,
            "ner_max_tokens": self.global_config.openie_ner_max_tokens,
            "triple_max_tokens": self.global_config.openie_triple_max_tokens,
            "prompt_schema": "hipporag_openie_v1",
            "text_preprocessor": component_class_identity(self.text_preprocessor) if hasattr(self, "text_preprocessor") else None,
            "explicit_identity": getattr(self, "index_identity", None),
        }

    def _current_openie_provenance(self) -> Dict[str, Any]:
        endpoint = None
        region = None
        gateway = find_gateway(self.global_config.llm_name)
        if self.global_config.openie_mode == "online" and self.global_config.llm_name.startswith("bedrock/"):
            region = self.global_config.bedrock_region or os.getenv("AWS_REGION_NAME") or os.getenv("AWS_DEFAULT_REGION")
        elif self.global_config.openie_mode == "online" and self.global_config.llm_name.startswith("bedrock-mantle/"):
            endpoint = self.global_config.llm_base_url
            region = self.global_config.bedrock_region or os.getenv("AWS_REGION_NAME") or os.getenv("AWS_DEFAULT_REGION")
        elif self.global_config.openie_mode == "online" and gateway is not None:
            endpoint = self.global_config.llm_base_url or gateway.default_base_url
        elif self.global_config.openie_mode == "online" and not self.global_config.llm_name.startswith("Transformers/"):
            endpoint = self.global_config.azure_endpoint or self.global_config.llm_base_url or "https://api.openai.com/v1"
        if self.global_config.openie_mode == "online":
            producer_component = getattr(self, "extraction_llm", None)
        else:
            producer_component = getattr(self, "openie", None)
        return {
            "identity": self._openie_state_identity(),
            "producer": {
                "mode": self.global_config.openie_mode,
                "class": component_class_identity(producer_component),
                "endpoint": endpoint,
                "region": region,
            },
        }

    def _validate_openie_provenance(self, provenance: Any, source_path: str) -> Dict[str, Any]:
        if not isinstance(provenance, dict) or provenance.get("identity") != self._openie_state_identity():
            raise StateConsistencyError(f"OpenIE provenance is missing or incompatible in {source_path}. Use a fresh save_dir or force a full re-extraction before building derived state.")
        stored_producer = provenance.get("producer")
        current_producer = self._current_openie_provenance()["producer"]
        phase_transition = isinstance(stored_producer, dict) and stored_producer.get("mode") == "offline" and current_producer["mode"] == "online"
        if stored_producer != current_producer and not phase_transition:
            raise StateConsistencyError(f"OpenIE producer changed for {source_path}. Use a fresh save_dir or force a full re-extraction.")
        return provenance

    def _openie_provenance_for_manifest(self) -> Dict[str, Any]:
        if self.global_config.force_openie_from_scratch:
            return self._current_openie_provenance()
        candidate_paths = (getattr(self, "openie_state_path", None), getattr(self, "openie_results_path", None))
        existing_path = next((path for path in candidate_paths if path and os.path.isfile(path)), None)
        if existing_path is None:
            return self._current_openie_provenance()
        try:
            with open(existing_path, encoding="utf-8") as openie_file:
                payload = json.load(openie_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise StateConsistencyError(f"Cannot read OpenIE state from {existing_path}: {exc}") from exc
        docs = payload.get("docs", [])
        if not isinstance(docs, list):
            raise StateConsistencyError(f"Invalid OpenIE state in {existing_path}: 'docs' must be a list.")
        if not docs:
            return self._current_openie_provenance()
        return self._validate_openie_provenance(payload.get("provenance"), existing_path)

    def _load_chunk_metadata(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.chunk_metadata_path):
            return {}
        with open(self.chunk_metadata_path, "r", encoding="utf-8") as metadata_file:
            return json.load(metadata_file)

    def _save_chunk_metadata(self) -> None:
        with open(self.chunk_metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(self.chunk_metadata, metadata_file, ensure_ascii=False, indent=2)

    def _preprocess_docs(self, docs: List[Union[str, Chunk]]) -> List[Chunk]:
        chunks = self.text_preprocessor.preprocess(docs)
        if not all(isinstance(chunk, Chunk) for chunk in chunks):
            raise TypeError("Text preprocessors must return a list of Chunk instances.")
        return chunks


    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            self._graph_state_available = False
            self._graph_edge_schema_available = True
            graph = ig.Graph(directed=self.global_config.is_directed_graph)
            graph["hipporag_edge_schema"] = 1
            return graph
        else:
            self._graph_state_available = True
            self._graph_edge_schema_available = preloaded_graph["hipporag_edge_schema"] == 1 if "hipporag_edge_schema" in preloaded_graph.attributes() else False
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def pre_openie(self, docs: List[Union[str, Chunk]]):
        ensure_list_input(docs, "docs")
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        processed_chunks = self._preprocess_docs(docs)
        chunks = {
            compute_mdhash_id(chunk.content, prefix="chunk-"): {"hash_id": compute_mdhash_id(chunk.content, prefix="chunk-"), "content": chunk.content}
            for chunk in processed_chunks
        }

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(
            chunks.keys(), force_reextract=self.global_config.force_openie_from_scratch)
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        self._openie_info = all_openie_info
        self._save_openie_state(all_openie_info)

        raise RuntimeError("Offline OpenIE completed. Run indexing again with openie_mode='online' to build the graph.")

    def index(self, docs: List[Union[str, Chunk]]):
        """
        Indexes the given documents based on the HippoRAG 2 framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        ensure_list_input(docs, "docs")
        if not self._graph_edge_schema_available:
            raise StateConsistencyError("This graph predates source-aware edges. Rebuild with force_index_from_scratch=True before indexing more documents.")
        if self.global_config.force_openie_from_scratch and (
            self.entity_embedding_store.get_all_ids() or self.fact_embedding_store.get_all_ids() or self.graph.vcount() > 0
        ):
            raise StateConsistencyError("force_openie_from_scratch cannot replace OpenIE after entity/fact or graph state exists. Disable the flag to reuse the current OpenIE state, or use a fresh save_dir for a full re-extraction.")
        if not self._graph_state_available and self.chunk_embedding_store.get_all_ids() and not (
            self.global_config.force_index_from_scratch or self.global_config.force_openie_from_scratch
        ):
            raise StateConsistencyError("Graph state is missing while embedding stores are non-empty. Set force_index_from_scratch=True to authorize an explicit graph rebuild.")
        self.ready_to_retrieve = False
        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        processed_chunks = self._preprocess_docs(docs)
        chunk_texts = [chunk.content for chunk in processed_chunks]

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(processed_chunks)

        self.chunk_embedding_store.insert_strings(chunk_texts)
        for chunk in processed_chunks:
            chunk_id = self.chunk_embedding_store.get_hash_id(chunk.content)
            metadata = dict(chunk.metadata)
            if chunk.source_id is not None:
                metadata["source_id"] = chunk.source_id
            previous_metadata = self.chunk_metadata.get(chunk_id)
            if previous_metadata is not None and previous_metadata != metadata:
                logger.warning(f"Replacing metadata for duplicate chunk {chunk_id}.")
            self.chunk_metadata[chunk_id] = metadata
        self._save_chunk_metadata()
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(
            chunk_to_rows.keys(), force_reextract=self.global_config.force_openie_from_scratch)
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        self._openie_info = all_openie_info
        self._save_openie_state(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        if not len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict):
            raise StateConsistencyError(f"OpenIE/chunk count mismatch: chunks={len(chunk_to_rows)}, NER={len(ner_results_dict)}, triples={len(triple_results_dict)}.")

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        new_entity_ids = list(self.entity_embedding_store.get_missing_string_hash_ids(entity_nodes))
        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}
        self._fact_edge_source_counts = {}
        self._passage_edge_sources = {}
        self._synonym_edge_scores = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self._pending_synonymy_entity_ids = new_entity_ids if self._graph_state_available else None
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

        self.ready_to_retrieve = False

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the HippoRAG class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        ensure_list_input(docs_to_delete, "docs_to_delete")
        #Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        #Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])
        if not chunk_ids_to_delete:
            return
        if self.graph.ecount() > 0 and not self._graph_edge_schema_available:
            raise StateConsistencyError("This graph predates source-aware edges, so deletion could leave stale facts. Rebuild explicitly with force_index_from_scratch=True before deleting documents.")

        #Find triples in chunks to delete
        all_openie_info = self._openie_info
        if all_openie_info is None:
            all_openie_info, _ = self.load_existing_openie([])
        openie_chunk_ids = {item['idx'] for item in all_openie_info}
        missing_provenance = chunk_ids_to_delete.difference(openie_chunk_ids)
        if missing_provenance:
            raise StateConsistencyError(f"Cannot delete chunks without persisted OpenIE provenance: {sorted(missing_provenance)}")
        triples_to_delete = []

        remaining_openie_info = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                remaining_openie_info.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        affected_processed_triples = {
            tuple(text_processing(list(triple))) for triple in triples_to_delete
        }
        unreferenced_processed_triples = []
        for processed_triple in affected_processed_triples:
            if remove_sources_from_mapping(self.proc_triples_to_docs, str(processed_triple), chunk_ids_to_delete):
                unreferenced_processed_triples.append(processed_triple)

        # Shared triples remain indexed, but every affected entity must lose the deleted chunk sources.
        affected_entities, _ = extract_entity_nodes([list(affected_processed_triples)])
        triple_ids_to_delete = {
            self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in unreferenced_processed_triples
        }

        affected_entity_ids = [self.entity_embedding_store.text_to_hash_id[ent] for ent in affected_entities]

        unreferenced_entity_ids = []

        for ent_node in affected_entity_ids:
            if remove_sources_from_mapping(self.ent_node_to_chunk_ids, ent_node, chunk_ids_to_delete):
                unreferenced_entity_ids.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(unreferenced_entity_ids)} Entities")

        self._save_openie_state(remaining_openie_info)
        self._openie_info = remaining_openie_info

        self.entity_embedding_store.delete(unreferenced_entity_ids)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)
        for chunk_id in chunk_ids_to_delete:
            self.chunk_metadata.pop(chunk_id, None)
        self._save_chunk_metadata()

        #Delete Nodes from Graph
        self._remove_fact_sources_from_graph(chunk_ids_to_delete)
        self.graph.delete_vertices(list(unreferenced_entity_ids) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    def _remove_fact_sources_from_graph(self, chunk_ids_to_delete: Set[str]) -> None:
        """Remove deleted chunk contributions from source-aware fact edges."""
        if not chunk_ids_to_delete or self.graph.ecount() == 0:
            return
        edges_to_delete = []
        for edge in self.graph.es:
            source_counts = edge["fact_source_counts"] or {}
            if not isinstance(source_counts, dict) or not chunk_ids_to_delete.intersection(source_counts):
                continue
            remaining_counts = {source: int(count) for source, count in source_counts.items() if source not in chunk_ids_to_delete}
            synonym_score = float(edge["synonym_score"] or 0.0)
            if not remaining_counts and synonym_score <= 0:
                edges_to_delete.append(edge.index)
                continue
            edge["fact_source_counts"] = remaining_counts
            edge["weight"] = max(float(sum(remaining_counts.values())), synonym_score)
            edge["edge_kind"] = "fact+synonym" if remaining_counts and synonym_score > 0 else ("fact" if remaining_counts else "synonym")
        if edges_to_delete:
            self.graph.delete_edges(edges_to_delete)

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the HippoRAG 2 framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        ensure_list_input(queries, "queries")
        validate_parallel_input_lengths(queries, gold_docs=gold_docs)
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k
        if not isinstance(num_to_retrieve, int) or num_to_retrieve < 1:
            raise ValueError("num_to_retrieve must be a positive integer.")

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            rerank_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            rerank_end = time.time()

            self.rerank_time += rerank_end - rerank_start

            if len(top_k_facts) == 0:
                logger.info('No facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                         link_top_k=self.global_config.linking_top_k,
                                                                                         query_fact_scores=query_fact_scores,
                                                                                         top_k_facts=top_k_facts,
                                                                                         top_k_fact_indices=top_k_fact_indices,
                                                                                         passage_node_weight=self.global_config.passage_node_weight)

            result = self._build_retrieval_result(query, sorted_doc_ids, sorted_doc_scores, num_to_retrieve, top_k_facts)
            retrieval_results.append(QuerySolution(question=result.query, docs=result.docs, doc_scores=result.scores,
                                                   doc_metadata=result.doc_metadata, graph_seeds=result.graph_seeds))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def _build_retrieval_result(self, query: str, sorted_doc_ids: np.ndarray, sorted_doc_scores: np.ndarray,
                                num_to_retrieve: int, graph_seeds: Optional[List[Tuple]] = None) -> RetrievalResult:
        passage_keys = [self.passage_node_keys[idx] for idx in sorted_doc_ids[:num_to_retrieve]]
        docs = [self.chunk_embedding_store.get_row(key)["content"] for key in passage_keys]
        metadata = [dict(self.chunk_metadata.get(key, {})) for key in passage_keys]
        return RetrievalResult(query=query, docs=docs, scores=np.asarray(sorted_doc_scores[:num_to_retrieve]),
                               doc_metadata=metadata, graph_seeds=graph_seeds or [])

    def retrieve_ircot(self,
                       queries: List[str],
                       max_qa_steps: int,
                       num_to_retrieve: int = None,
                       gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """Retrieve documents iteratively by alternating HippoRAG 2 retrieval and one-step reasoning."""
        ensure_list_input(queries, "queries")
        validate_parallel_input_lengths(queries, gold_docs=gold_docs)
        if max_qa_steps < 1:
            raise ValueError("max_qa_steps must be at least 1.")
        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k
        if not isinstance(num_to_retrieve, int) or num_to_retrieve < 1:
            raise ValueError("num_to_retrieve must be a positive integer.")

        prompt_name = f'ircot_{self.global_config.dataset}'
        if max_qa_steps > 1 and not self.prompt_template_manager.is_template_name_valid(prompt_name):
            raise ValueError(f"IRCoT prompt template '{prompt_name}' is not available.")

        retrieval_results = []
        for query in tqdm(queries, desc="IRCoT retrieval"):
            step_result = self.retrieve([query], num_to_retrieve=num_to_retrieve)[0]
            merged_doc_scores = dict(zip(step_result.docs, step_result.doc_scores.tolist()))
            merged_doc_metadata = dict(zip(step_result.docs, step_result.doc_metadata or []))
            thoughts = []

            for _ in range(1, max_qa_steps):
                ranked_docs = sorted(merged_doc_scores, key=merged_doc_scores.get, reverse=True)
                thought = reason_step(self.global_config.dataset, self.prompt_template_manager, query,
                                      ranked_docs[:num_to_retrieve], thoughts, self.qa_llm)
                thoughts.append(thought)
                if 'So the answer is:' in thought:
                    break

                step_result = self.retrieve([thought], num_to_retrieve=num_to_retrieve)[0]
                for doc, score in zip(step_result.docs, step_result.doc_scores.tolist()):
                    merged_doc_scores[doc] = max(merged_doc_scores.get(doc, float('-inf')), score)
                merged_doc_metadata.update(dict(zip(step_result.docs, step_result.doc_metadata or [])))

            ranked_items = sorted(merged_doc_scores.items(), key=lambda item: item[1], reverse=True)
            retrieval_results.append(QuerySolution(question=query,
                                                   docs=[doc for doc, _ in ranked_items],
                                                   doc_scores=np.asarray([score for _, score in ranked_items]),
                                                   thoughts=thoughts,
                                                   doc_metadata=[merged_doc_metadata.get(doc, {}) for doc, _ in ranked_items]))

        if gold_docs is None:
            return retrieval_results

        retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)
        k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
        overall_retrieval_result, _ = retrieval_recall_evaluator.calculate_metric_scores(
            gold_docs=gold_docs, retrieved_docs=[result.docs for result in retrieval_results], k_list=k_list)
        return retrieval_results, overall_retrieval_result

    def answer_with_ircot(self,
                          queries: List[str],
                          max_qa_steps: int,
                          gold_docs: List[List[str]] = None,
                          gold_answers: List[List[str]] = None):
        """Run QA with optional IRCoT retrieval while leaving the default rag_qa path unchanged."""
        ensure_list_input(queries, "queries")
        validate_parallel_input_lengths(queries, gold_docs=gold_docs, gold_answers=gold_answers)
        if gold_docs is None:
            query_solutions = self.retrieve_ircot(queries, max_qa_steps=max_qa_steps)
            overall_retrieval_result = None
        else:
            query_solutions, overall_retrieval_result = self.retrieve_ircot(
                queries, max_qa_steps=max_qa_steps, gold_docs=gold_docs)

        query_solutions, all_response_message, all_metadata = self.qa(query_solutions)
        if gold_answers is None:
            return query_solutions, all_response_message, all_metadata

        qa_em_evaluator = QAExactMatch(global_config=self.global_config)
        qa_f1_evaluator = QAF1Score(global_config=self.global_config)
        overall_qa_results, _ = qa_em_evaluator.calculate_metric_scores(
            gold_answers=gold_answers, predicted_answers=[result.answer for result in query_solutions], aggregation_fn=np.max)
        overall_qa_f1_result, _ = qa_f1_evaluator.calculate_metric_scores(
            gold_answers=gold_answers, predicted_answers=[result.answer for result in query_solutions], aggregation_fn=np.max)
        overall_qa_results.update(overall_qa_f1_result)
        overall_qa_results = {key: round(float(value), 4) for key, value in overall_qa_results.items()}
        for idx, result in enumerate(query_solutions):
            result.gold_answers = list(gold_answers[idx])
            if gold_docs is not None:
                result.gold_docs = gold_docs[idx]
        return query_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using the HippoRAG 2 framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        ensure_list_input(queries, "queries")
        validate_parallel_input_lengths(queries, gold_docs=gold_docs, gold_answers=gold_answers)
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if queries and not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        ensure_list_input(queries, "queries")
        validate_parallel_input_lengths(queries, gold_docs=gold_docs)
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k
        if not isinstance(num_to_retrieve, int) or num_to_retrieve < 1:
            raise ValueError("num_to_retrieve must be a positive integer.")

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            logger.info('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            result = self._build_retrieval_result(query, sorted_doc_ids, sorted_doc_scores, num_to_retrieve)
            retrieval_results.append(QuerySolution(question=result.query, docs=result.docs, doc_scores=result.scores,
                                                   doc_metadata=result.doc_metadata, graph_seeds=result.graph_seeds))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        ensure_list_input(queries, "queries")
        validate_parallel_input_lengths(queries, gold_docs=gold_docs, gold_answers=gold_answers)
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if queries and not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        ensure_list_input(queries, "queries")
        if not queries:
            return [], [], []
        #Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.warning(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        all_qa_results = [self.qa_llm.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        #Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if not hasattr(self, "_fact_edge_source_counts"):
            self._fact_edge_source_counts = {}
        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            for triple in triples:
                triple = tuple(triple)

                node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                entities_in_chunk.add(node_key)
                entities_in_chunk.add(node_2_key)

                if chunk_key not in current_graph_nodes:
                    fact_edges = ((node_key, node_2_key), (node_2_key, node_key)) if self.graph.is_directed() else (tuple(sorted((node_key, node_2_key))),)
                    for fact_edge in fact_edges:
                        self.node_to_node_stats[fact_edge] = self.node_to_node_stats.get(fact_edge, 0.0) + 1
                        source_counts = self._fact_edge_source_counts.setdefault(fact_edge, {})
                        source_counts[chunk_key] = source_counts.get(chunk_key, 0) + 1

            for node in entities_in_chunk:
                self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union({chunk_key})

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if not hasattr(self, "_passage_edge_sources"):
            self._passage_edge_sources = {}
        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    passage_edge = (chunk_key, node_key)
                    self.node_to_node_stats[passage_edge] = 1.0
                    self._passage_edge_sources[passage_edge] = chunk_key

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self, query_node_keys: Optional[List[str]] = None):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")
        if not hasattr(self, "_synonym_edge_scores"):
            self._synonym_edge_scores = {}

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())
        if query_node_keys is None and hasattr(self, "_pending_synonymy_entity_ids"):
            query_node_keys = self._pending_synonymy_entity_ids
            del self._pending_synonymy_entity_ids
        if query_node_keys is None:
            query_node_keys = entity_node_keys
        else:
            query_node_keys = [node_key for node_key in query_node_keys if node_key in self.entity_id_to_row]

        logger.info(f"Performing KNN retrieval for {len(query_node_keys)} new phrase nodes against {len(entity_node_keys)} total nodes.")
        if not query_node_keys or not entity_node_keys:
            return

        query_entity_embs = self.entity_embedding_store.get_embeddings(query_node_keys)
        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=query_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=query_entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns >= 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        synonyms.append((nn, score))
                        num_synonym_triple += 1
                        similarity_edges = ((node_key, nn), (nn, node_key)) if self.graph.is_directed() else (tuple(sorted((node_key, nn))),)
                        for sim_edge in similarity_edges:
                            self.node_to_node_stats[sim_edge] = max(self.node_to_node_stats.get(sim_edge, 0.0), score)
                            self._synonym_edge_scores[sim_edge] = max(self._synonym_edge_scores.get(sim_edge, 0.0), float(score))
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str], force_reextract: bool = False) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        existing_path = next((path for path in (self.openie_state_path, self.openie_results_path) if os.path.isfile(path)), None)
        provenance = self._current_openie_provenance()
        if not force_reextract and self._openie_info is not None:
            all_openie_info = list(self._openie_info)
            provenance = self._openie_provenance or provenance
        elif not force_reextract and existing_path is not None:
            try:
                with open(existing_path, encoding="utf-8") as openie_file:
                    openie_results = json.load(openie_file)
            except (OSError, json.JSONDecodeError) as exc:
                raise StateConsistencyError(f"Cannot read OpenIE state from {existing_path}: {exc}") from exc
            all_openie_info = openie_results.get('docs', [])
            if not isinstance(all_openie_info, list):
                raise StateConsistencyError(f"Invalid OpenIE state in {existing_path}: 'docs' must be a list.")
            if all_openie_info:
                provenance = self._validate_openie_provenance(openie_results.get("provenance"), existing_path)

            # Normalize IDs from older files that did not store canonical chunk hashes.
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
        else:
            all_openie_info = []

        existing_openie_keys = [info['idx'] for info in all_openie_info]
        if len(existing_openie_keys) != len(set(existing_openie_keys)):
            raise StateConsistencyError("OpenIE state contains duplicate chunk IDs.")
        chunk_keys_to_save = set(chunk_keys).difference(existing_openie_keys)
        current_provenance = self._current_openie_provenance()
        if all_openie_info and chunk_keys_to_save and provenance.get("producer") != current_provenance["producer"]:
            raise StateConsistencyError("Cannot append OpenIE rows with a different producer. Use a fresh save_dir or force a full re-extraction.")
        self._openie_provenance = current_provenance if chunk_keys_to_save else provenance

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        expected_keys = set(chunks_to_save)
        ner_keys = set(ner_results_dict)
        triple_keys = set(triple_results_dict)
        if ner_keys != expected_keys or triple_keys != expected_keys:
            raise StateConsistencyError(
                f"Incomplete OpenIE batch: expected={sorted(expected_keys)}, ner_missing={sorted(expected_keys - ner_keys)}, "
                f"ner_extra={sorted(ner_keys - expected_keys)}, triple_missing={sorted(expected_keys - triple_keys)}, "
                f"triple_extra={sorted(triple_keys - expected_keys)}."
            )

        merged_rows = []
        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            entities = ner_results_dict[chunk_key].unique_entities
            triples = triple_results_dict[chunk_key].triples
            if not isinstance(entities, list) or not all(isinstance(entity, str) for entity in entities):
                raise StateConsistencyError(f"Invalid NER result for chunk {chunk_key}.")
            if not isinstance(triples, list) or not all(isinstance(triple, (list, tuple)) and len(triple) == 3 and all(isinstance(value, str) for value in triple) for triple in triples):
                raise StateConsistencyError(f"Invalid triple result for chunk {chunk_key}.")
            merged_rows.append({'idx': chunk_key, 'passage': passage, 'extracted_entities': entities, 'extracted_triples': triples})

        all_openie_info.extend(merged_rows)

        return all_openie_info

    def _save_openie_state(self, all_openie_info: List[dict]) -> None:
        self.save_openie_results(all_openie_info, self.openie_state_path)
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info, self.openie_results_path)

    def save_openie_results(self, all_openie_info: List[dict], output_path: Optional[str] = None):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if num_phrases > 0:
            avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
            avg_ent_words = round(sum_phrase_words / num_phrases, 4)
        else:
            avg_ent_chars = 0
            avg_ent_words = 0

        openie_dict = {
            'docs': all_openie_info,
            'avg_ent_chars': avg_ent_chars,
            'avg_ent_words': avg_ent_words,
            'provenance': getattr(self, "_openie_provenance", None) or self._current_openie_provenance(),
        }

        output_path = output_path or self.openie_results_path
        temporary_path = output_path + ".tmp"
        with FileLock(output_path + ".lock"):
            try:
                with open(temporary_path, 'w', encoding="utf-8") as openie_file:
                    json.dump(openie_dict, openie_file, ensure_ascii=False)
                os.replace(temporary_path, output_path)
            finally:
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)
        logger.info(f"OpenIE results saved to {output_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Merge typed edge contributions into the graph without creating parallel copies.

        `node_to_node_stats` only contains contributions from the current indexing pass.
        Existing fact sources and synonym scores therefore have to be merged by their
        logical `(source_key, target_key)` identity to make incremental and one-shot
        indexing produce the same weighted graph. Undirected edges use a canonical key
        so reverse entries cannot become parallel physical edges.
        """
        attribute_names = ("weight", "edge_kind", "fact_source_counts", "synonym_score", "passage_source", "source_key", "target_key")

        def logical_key(source_key, target_key):
            return (source_key, target_key) if self.graph.is_directed() else tuple(sorted((source_key, target_key)))

        def compose_metadata(source_key, target_key, fact_source_counts=None, synonym_score=0.0, passage_source=None):
            if fact_source_counts is None:
                fact_source_counts = {}
            if not isinstance(fact_source_counts, dict):
                raise StateConsistencyError(f"Invalid fact provenance for graph edge {source_key} -> {target_key}.")
            normalized_counts = {}
            for source, count in fact_source_counts.items():
                if not isinstance(source, str) or not isinstance(count, int) or isinstance(count, bool) or count < 1:
                    raise StateConsistencyError(f"Invalid fact source count for graph edge {source_key} -> {target_key}.")
                normalized_counts[source] = count
            synonym_score = float(synonym_score or 0.0)
            if synonym_score < 0:
                raise StateConsistencyError(f"Invalid synonym score for graph edge {source_key} -> {target_key}.")
            edge_kinds = []
            if normalized_counts:
                edge_kinds.append("fact")
            if passage_source is not None:
                edge_kinds.append("passage")
            if synonym_score > 0:
                edge_kinds.append("synonym")
            if not edge_kinds:
                raise StateConsistencyError(f"Graph edge {source_key} -> {target_key} has no typed contribution.")
            return {
                "weight": max(float(sum(normalized_counts.values())), 1.0 if passage_source is not None else 0.0, synonym_score),
                "edge_kind": "+".join(edge_kinds),
                "fact_source_counts": normalized_counts,
                "synonym_score": synonym_score,
                "passage_source": passage_source,
                "source_key": source_key,
                "target_key": target_key,
            }

        def merge_metadata(left, right):
            merged_counts = dict(left["fact_source_counts"])
            for source, count in right["fact_source_counts"].items():
                merged_counts[source] = max(merged_counts.get(source, 0), count)
            left_passage = left["passage_source"]
            right_passage = right["passage_source"]
            if left_passage is not None and right_passage is not None and left_passage != right_passage:
                raise StateConsistencyError(f"Conflicting passage provenance for graph edge {left['source_key']} -> {left['target_key']}.")
            return compose_metadata(
                left["source_key"],
                left["target_key"],
                merged_counts,
                max(left["synonym_score"], right["synonym_score"]),
                left_passage if left_passage is not None else right_passage,
            )

        current_node_ids = set(self.graph.vs["name"])
        pending_metadata = {}
        for edge in self.node_to_node_stats:
            if edge[0] == edge[1]:
                continue
            if edge[0] not in current_node_ids or edge[1] not in current_node_ids:
                logger.warning(f"Edge {edge[0]} -> {edge[1]} is not valid.")
                continue
            canonical_edge = logical_key(edge[0], edge[1])
            metadata = compose_metadata(
                canonical_edge[0],
                canonical_edge[1],
                dict(getattr(self, "_fact_edge_source_counts", {}).get(edge, {})),
                getattr(self, "_synonym_edge_scores", {}).get(edge, 0.0),
                getattr(self, "_passage_edge_sources", {}).get(edge),
            )
            pending_metadata[canonical_edge] = merge_metadata(pending_metadata[canonical_edge], metadata) if canonical_edge in pending_metadata else metadata

        existing_edges = {}
        duplicate_edge_indices = []
        required_existing_attributes = set(attribute_names).difference({"weight", "edge_kind"})
        for graph_edge in self.graph.es:
            if not required_existing_attributes.issubset(graph_edge.attributes()):
                raise StateConsistencyError("The graph is missing source-aware edge metadata. Rebuild with force_index_from_scratch=True.")
            source_key = graph_edge["source_key"]
            target_key = graph_edge["target_key"]
            if not isinstance(source_key, str) or not isinstance(target_key, str):
                raise StateConsistencyError("Graph edge source_key and target_key must be strings.")
            physical_edge = (self.graph.vs[graph_edge.source]["name"], self.graph.vs[graph_edge.target]["name"])
            endpoints_match = physical_edge == (source_key, target_key) if self.graph.is_directed() else set(physical_edge) == {source_key, target_key}
            if not endpoints_match:
                raise StateConsistencyError(f"Graph edge metadata {source_key} -> {target_key} does not match its physical endpoints {physical_edge}.")
            canonical_edge = logical_key(source_key, target_key)
            metadata = compose_metadata(
                canonical_edge[0],
                canonical_edge[1],
                graph_edge["fact_source_counts"],
                graph_edge["synonym_score"],
                graph_edge["passage_source"],
            )
            if canonical_edge in existing_edges:
                canonical_index, canonical_metadata = existing_edges[canonical_edge]
                existing_edges[canonical_edge] = (canonical_index, merge_metadata(canonical_metadata, metadata))
                duplicate_edge_indices.append(graph_edge.index)
            else:
                existing_edges[canonical_edge] = (graph_edge.index, metadata)

        new_edges = []
        new_metadata = []
        for logical_edge, metadata in pending_metadata.items():
            if logical_edge in existing_edges:
                canonical_index, existing_metadata = existing_edges[logical_edge]
                existing_edges[logical_edge] = (canonical_index, merge_metadata(existing_metadata, metadata))
            else:
                new_edges.append(logical_edge)
                new_metadata.append(metadata)

        for edge_index, metadata in existing_edges.values():
            for attribute_name in attribute_names:
                self.graph.es[edge_index][attribute_name] = metadata[attribute_name]
        if duplicate_edge_indices:
            self.graph.delete_edges(duplicate_edge_indices)
        if new_edges:
            self.graph.add_edges(new_edges, attributes={name: [metadata[name] for metadata in new_metadata] for name in attribute_names})

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        if self._graph_edge_schema_available:
            self.graph["hipporag_edge_schema"] = 1
        temporary_path = self._graph_pickle_filename + ".tmp"
        with FileLock(self._graph_pickle_filename + ".lock"):
            try:
                self.graph.write_pickle(temporary_path)
                os.replace(temporary_path, self._graph_pickle_filename)
            finally:
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)
        self._graph_state_available = True
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        passage_nodes_set = set(passage_nodes_keys)
        vertex_names = self.graph.vs["name"] if self.graph.vcount() and "name" in self.graph.vs.attribute_names() else []
        num_triples_with_passage_node = sum(1 for edge in self.graph.es if vertex_names[edge.source] in passage_nodes_set or vertex_names[edge.target] in passage_nodes_set)
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        if "synonym_score" in self.graph.es.attribute_names():
            graph_info['num_synonymy_triples'] = sum(1 for edge in self.graph.es if float(edge["synonym_score"] or 0.0) > 0)
        else:
            graph_info['num_synonymy_triples'] = None

        # get # of total triples
        graph_info["num_total_triples"] = self.graph.ecount()

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()
        if expected_node_count > 0 and not self._graph_state_available:
            raise StateConsistencyError("Graph state is missing while embedding stores are non-empty. Re-run index() to rebuild it explicitly.")
        if expected_node_count != actual_node_count:
            raise StateConsistencyError(f"Graph/store node count mismatch: expected {expected_node_count}, got {actual_node_count}.")
        if actual_node_count > 0 and "name" not in self.graph.vs.attribute_names():
            raise StateConsistencyError("Graph vertices are missing their name attribute.")

        igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
        expected_node_ids = set(self.entity_node_keys).union(self.passage_node_keys)
        if set(igraph_name_to_idx) != expected_node_ids:
            raise StateConsistencyError("Graph vertex IDs do not match embedding store IDs.")
        self.node_name_to_vertex_idx = igraph_name_to_idx
        self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys]
        self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys]

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        all_openie_info = self._openie_info
        if all_openie_info is None:
            all_openie_info, _ = self.load_existing_openie([])
        openie_chunk_ids = {item['idx'] for item in all_openie_info}
        if openie_chunk_ids != set(self.passage_node_keys):
            missing = sorted(set(self.passage_node_keys).difference(openie_chunk_ids))
            extra = sorted(openie_chunk_ids.difference(self.passage_node_keys))
            raise StateConsistencyError(f"OpenIE provenance does not match chunk storage (missing={missing}, extra={extra}).")

        self.proc_triples_to_docs = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

            # prepare data_store
            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in self.passage_node_keys]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        query_strings = []
        seen_queries = set()
        for query in queries:
            query_text = query.question if isinstance(query, QuerySolution) else query
            if not isinstance(query_text, str):
                raise TypeError(f"Queries must be strings or QuerySolution instances, got {type(query).__name__}.")
            if query_text not in seen_queries:
                query_strings.append(query_text)
                seen_queries.add(query_text)

        triple_queries = [
            query for query in query_strings
            if self.global_config.linking_top_k > 0 and self.fact_node_keys and query not in self.query_to_embedding['triple']
        ]
        passage_queries = [query for query in query_strings if self.passage_node_keys and query not in self.query_to_embedding['passage']]

        instruction_mode = getattr(self.embedding_model, "query_instruction_mode", "ignored")
        if instruction_mode != "distinct":
            triple_set = set(triple_queries)
            passage_set = set(passage_queries)
            shared_queries = []
            for query in query_strings:
                if query not in triple_set and query not in passage_set:
                    continue
                existing_embedding = self.query_to_embedding['triple'].get(query)
                if existing_embedding is None:
                    existing_embedding = self.query_to_embedding['passage'].get(query)
                if existing_embedding is None:
                    shared_queries.append(query)
                    continue
                if query in triple_set:
                    self.query_to_embedding['triple'][query] = existing_embedding
                if query in passage_set:
                    self.query_to_embedding['passage'][query] = existing_embedding
            if shared_queries:
                logger.info(f"Encoding {len(shared_queries)} shared query embeddings.")
                encode_kwargs = {"norm": True}
                if instruction_mode == "shared":
                    encode_kwargs["instruction"] = get_query_instruction('query_to_passage')
                shared_embeddings = self.embedding_model.batch_encode(shared_queries, **encode_kwargs)
                for query, embedding in zip(shared_queries, shared_embeddings):
                    if query in triple_set:
                        self.query_to_embedding['triple'][query] = embedding
                    if query in passage_set:
                        self.query_to_embedding['passage'][query] = embedding
            return

        if triple_queries:
            logger.info(f"Encoding {len(triple_queries)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(triple_queries,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(triple_queries, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

        if passage_queries:
            logger.info(f"Encoding {len(passage_queries)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(passage_queries,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(passage_queries, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        if self.global_config.linking_top_k <= 0:
            return np.asarray([], dtype=np.float32)

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])

        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)
            
        query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
        query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
        return min_max_normalize(query_fact_scores)

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        if len(self.passage_embeddings) == 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """

        #Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))

        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase_key])

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                    phrases_and_ids.add((phrase, phrase_id))

        phrase_weights = np.divide(phrase_weights, number_of_occurs, out=np.zeros_like(phrase_weights), where=number_of_occurs != 0)

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(phrase_weights[phrase_id])

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k

        #Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        #Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        #Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        if sum(node_weights) <= 0:
            raise StateConsistencyError(f'No positive graph seeds were found for facts: {top_k_facts}')

        #Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        self.ppr_time += (ppr_end - ppr_start)

        if len(ppr_sorted_doc_ids) != len(self.passage_node_idxs):
            raise StateConsistencyError(f"PPR returned {len(ppr_sorted_doc_ids)} documents for a corpus of {len(self.passage_node_idxs)}.")

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores


    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k

        if link_top_k <= 0:
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                # If we have fewer facts than requested, use all of them
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # Otherwise get the top k
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [ast.literal_eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
            
            # Rerank the facts
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)
            
            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}
    
    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            # HippoRAG propagates relevance over the undirected projection of its provenance graph.
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
