import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""
    # LLM specific attributes 
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the LLM model, if none, means using OPENAI service."}
    )
    embedding_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for an OpenAI compatible embedding model, if none, means using OPENAI service."}
    )
    azure_endpoint: str = field(
        default=None,
        metadata={"help": "Azure OpenAI resource endpoint. Legacy full chat-completions URLs are accepted with a warning."}
    )
    azure_embedding_endpoint: str = field(
        default=None,
        metadata={"help": "Azure OpenAI resource endpoint for embeddings. Legacy full embedding URLs are accepted with a warning."}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "Random seed."}
    )
    temperature: Optional[float] = field(
        default=0,
        metadata={"help": "Temperature for sampling. Set to None for models or endpoints that do not accept it."}
    )
    response_format: Union[dict, None] = field(
        default=None,
        metadata={"help": "Optional Chat Completions response_format used for OpenIE requests. Direct LLM calls can override it per request."}
    )
    bedrock_mantle_auth: Literal["api_key", "aws_credentials"] = field(
        default="api_key",
        metadata={"help": "Authentication method for the Amazon Bedrock Mantle endpoint."}
    )
    bedrock_aws_profile: Optional[str] = field(
        default=None,
        metadata={"help": "AWS profile used when Bedrock Mantle authentication is aws_credentials."}
    )
    bedrock_region: Optional[str] = field(
        default=None,
        metadata={"help": "AWS region used to sign Bedrock Mantle requests."}
    )
    
    ## LLM specific attributes -> Async hyperparameters
    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "Retries after the initial SDK request; total HTTP attempts can be this value plus one."}
    )
    # Storage specific attributes
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, ignores existing OpenIE state. For an already-derived index, use a fresh save_dir so graph/entity/fact state cannot become stale."}
    )

    # Storage specific attributes 
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, rebuilds graph state while reusing compatible embedding/OpenIE stores. Use a fresh save_dir for a full storage rebuild."}
    )
    rerank_dspy_file_path: str = field(
        default=None,
        metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "Multiplicative factor that modified the passage node weights in PPR."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, writes an additional human-readable OpenIE export. Canonical provenance required for incremental indexing is always persisted."}
    )
    
    # Preprocessing specific attributes
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "Name of the text-based preprocessor to use in preprocessing."}
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={"help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "Number of overlap tokens between neighbouring chunks."}
    )
    preprocess_chunk_max_token_size: int = field(
        default=None,
        metadata={"help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default='by_token')
    
    
    # Information extraction specific attributes
    information_extraction_model_name: Literal["openie_openai_gpt", ] = field(
        default="openie_openai_gpt",
        metadata={"help": "Class name indicating which information extraction model to use."}
    )
    openie_mode: Literal["offline", "online", "Transformers-offline"] = field(
        default="online",
        metadata={"help": "Mode of the OpenIE model to use."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "Whether to skip graph construction or not. Set it to be true when running vllm offline indexing for the first time."}
    )
    
    
    # Embedding specific attributes
    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "Data type for local embedding model."}
    )
    
    
    
    # Graph construction specific attributes
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )
    
    
    
    # Retrieval specific attributes
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "Damping factor for ppr algorithm."}
    )
    
    
    # QA specific attributes
    max_qa_steps: int = field(
        default=1,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )
    
    # Save dir (highest level directory)
    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )

    # Vector store backend
    vector_store_type: Literal["parquet", "qdrant", "chroma", "milvus"] = field(
        default="parquet",
        metadata={"help": "Which embedding store backend to use. "
                  "'parquet' (default) stores embeddings in local Parquet files. "
                  "'qdrant' uses a Qdrant vector database (local file or remote). "
                  "'chroma' uses a ChromaDB collection (local file or remote HTTP). "
                  "'milvus' uses Milvus Lite, Milvus server, or Zilliz Cloud."}
    )

    # Qdrant-specific settings (only used when vector_store_type='qdrant')
    qdrant_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL of a remote Qdrant server (e.g. 'http://localhost:6333'). "
                  "If None, a local file-based Qdrant store is used inside save_dir."}
    )
    qdrant_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "API key for Qdrant Cloud or a secured remote Qdrant instance."}
    )

    # ChromaDB-specific settings (only used when vector_store_type='chroma')
    chroma_host: Optional[str] = field(
        default=None,
        metadata={"help": "Hostname of a remote ChromaDB HTTP server. "
                  "If None, a local persistent ChromaDB store is used inside save_dir."}
    )
    chroma_port: int = field(
        default=8000,
        metadata={"help": "Port of the remote ChromaDB HTTP server."}
    )

    # Milvus-specific settings (only used when vector_store_type='milvus')
    milvus_uri: Optional[str] = field(
        default=None,
        metadata={"help": "Milvus URI. If None, MILVUS_URI is used when set; otherwise "
                  "a local Milvus Lite database is created inside save_dir."}
    )
    milvus_token: Optional[str] = field(
        default=None,
        metadata={"help": "Milvus or Zilliz Cloud token. If None, MILVUS_TOKEN is used when set."}
    )
    milvus_db_name: Optional[str] = field(
        default=None,
        metadata={"help": "Milvus database name. If None, MILVUS_DB_NAME is used when set."}
    )
    milvus_consistency_level: Optional[Literal["Strong", "Session", "Bounded", "Eventually"]] = field(
        default=None,
        metadata={"help": "Milvus consistency level. If None, MILVUS_CONSISTENCY_LEVEL is used "
                  "when set; otherwise the Milvus client default is used."}
    )
    
    
    
    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )
    ## Dataset running specific attributes -> Graph
    graph_type: Literal[
        'dpr_only', 
        'entity', 
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation', 
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the corpus to use."}
    )

    # Additive settings are kept after the original fields to preserve positional compatibility.
    llm_supports_max_completion_tokens: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether a chat-completions endpoint accepts max_completion_tokens. When unset, official OpenAI and Azure endpoints and registered gateways that document it use max_completion_tokens; other endpoints use max_tokens."}
    )
    azure_api_version: Optional[str] = field(
        default=None,
        metadata={"help": "Azure OpenAI API version for chat completions."}
    )
    azure_chat_deployment: Optional[str] = field(
        default=None,
        metadata={"help": "Azure OpenAI deployment name for chat completions; defaults to llm_name."}
    )
    azure_embedding_api_version: Optional[str] = field(
        default=None,
        metadata={"help": "Azure OpenAI API version for embeddings; defaults to azure_api_version."}
    )
    azure_embedding_deployment: Optional[str] = field(
        default=None,
        metadata={"help": "Azure OpenAI deployment name for embeddings; defaults to embedding_model_name."}
    )
    openie_max_workers: int = field(
        default=8,
        metadata={"help": "Maximum number of concurrent online OpenIE requests."}
    )
    openie_ner_max_tokens: int = field(
        default=2048,
        metadata={"help": "Maximum output tokens for each online OpenIE NER request. Sized to leave room for reasoning tokens."}
    )
    openie_triple_max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum output tokens for each online OpenIE triple request. Sized to leave room for reasoning tokens."}
    )
    embedding_request_timeout: float = field(
        default=60.0,
        metadata={"help": "Timeout in seconds for HTTP embedding requests."}
    )
    embedding_provider: Optional[Literal["openai", "transformers", "vllm", "gritlm", "nvembed", "contriever", "cohere"]] = field(
        default=None,
        metadata={"help": "Explicit embedding provider. When unset, the legacy model-name routing rules are used."}
    )
    vector_store_namespace: Optional[str] = field(
        default=None,
        metadata={"help": "Stable, unique index namespace for vector database collections. Defaults to a fingerprint of the working directory."}
    )
    llm_api_key_env: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the environment variable holding the LLM API key. When unset, OpenAI-compatible endpoints use OPENAI_API_KEY and named gateways use their registered variable."}
    )
    embedding_api_key_env: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the environment variable holding the OpenAI-compatible embedding API key. When unset, OPENAI_API_KEY is used."}
    )
    
    
    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.openie_mode not in {"online", "offline", "Transformers-offline"}:
            raise ValueError(f"Unsupported openie_mode: {self.openie_mode}")
        if self.embedding_provider not in {None, "openai", "transformers", "vllm", "gritlm", "nvembed", "contriever", "cohere"}:
            raise ValueError(f"Unsupported embedding_provider: {self.embedding_provider}")
        if self.vector_store_namespace is not None and (not isinstance(self.vector_store_namespace, str) or not self.vector_store_namespace.strip()):
            raise ValueError("vector_store_namespace must be a non-empty string when set.")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1 when set.")
        if self.max_retry_attempts < 0:
            raise ValueError("max_retry_attempts cannot be negative.")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2 when set.")
        if (self.azure_api_version or self.azure_chat_deployment) and not self.azure_endpoint:
            raise ValueError("azure_endpoint is required when Azure chat settings are configured.")
        if (self.azure_embedding_api_version or self.azure_embedding_deployment) and not self.azure_embedding_endpoint:
            raise ValueError("azure_embedding_endpoint is required when Azure embedding settings are configured.")
        if self.openie_max_workers < 1:
            raise ValueError("openie_max_workers must be at least 1.")
        if self.openie_ner_max_tokens < 1 or self.openie_triple_max_tokens < 1:
            raise ValueError("OpenIE token limits must be at least 1.")
        if self.num_gen_choices != 1:
            raise ValueError("num_gen_choices must be 1 because HippoRAG consumes one response per inference call.")
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be at least 1.")
        if self.embedding_request_timeout <= 0:
            raise ValueError("embedding_request_timeout must be greater than 0.")
        if self.linking_top_k < 0:
            raise ValueError("linking_top_k cannot be negative; use 0 to disable graph fact linking.")
        if self.synonymy_edge_topk < 1 or self.synonymy_edge_query_batch_size < 1 or self.synonymy_edge_key_batch_size < 1:
            raise ValueError("Synonym KNN top-k and batch sizes must be at least 1.")
        if not -1 <= self.synonymy_edge_sim_threshold <= 1:
            raise ValueError("synonymy_edge_sim_threshold must be between -1 and 1.")
        if self.retrieval_top_k < 1 or self.qa_top_k < 1:
            raise ValueError("retrieval_top_k and qa_top_k must be at least 1.")
        if self.max_qa_steps < 1:
            raise ValueError("max_qa_steps must be at least 1.")
        if self.passage_node_weight < 0:
            raise ValueError("passage_node_weight cannot be negative.")
        if not 0 < self.damping < 1:
            raise ValueError("damping must be strictly between 0 and 1.")
        if self.save_dir is None: # If save_dir not given
            if self.dataset is None: self.save_dir = 'outputs' # running freely
            else: self.save_dir = os.path.join('outputs', self.dataset) # customize your dataset's output dir here
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")
