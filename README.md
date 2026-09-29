<h1 align="center">HippoRAG 2: From RAG to Memory</h1>
<p align="center">
    <img src="https://github.com/OSU-NLP-Group/HippoRAG/raw/main/images/hippo_brain.png" width="55%" style="max-width: 300px;">
</p>

[<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1nuelysWsXL8F5xH6q4JYJI8mvtlmeM9O#scrollTo=TjHdNe2KC81K)

[<img align="center" src="https://img.shields.io/badge/arXiv-2502.14802 HippoRAG 2-b31b1b" />](https://arxiv.org/abs/2502.14802)
[<img align="center" src="https://img.shields.io/badge/🤗 Dataset-HippoRAG 2-yellow" />](https://huggingface.co/datasets/osunlp/HippoRAG_2/tree/main)
[<img align="center" src="https://img.shields.io/badge/arXiv-2405.14831 HippoRAG 1-b31b1b" />](https://arxiv.org/abs/2405.14831)
[<img align="center" src="https://img.shields.io/badge/GitHub-HippoRAG 1-blue" />](https://github.com/OSU-NLP-Group/HippoRAG/tree/legacy)

HippoRAG 2 is a memory framework for LLMs that recognizes and uses connections in new knowledge, mirroring a key function of human long-term memory.

Our experiments show that HippoRAG 2 improves associativity (multi-hop retrieval) and sense-making (the process of integrating large and complex contexts) in even the most advanced RAG systems, without sacrificing their performance on simpler tasks.

Like its predecessor, HippoRAG 2 remains cost and latency efficient in online processes, while using significantly fewer resources for offline indexing compared to other graph-based solutions such as GraphRAG, RAPTOR, and LightRAG.

<p align="center">
  <img align="center" src="https://github.com/OSU-NLP-Group/HippoRAG/raw/main/images/intro.png" />
</p>
<p align="center">
  <b>Figure 1:</b> Evaluation of continual learning capabilities across three key dimensions: factual memory (NaturalQuestions, PopQA), sense-making (NarrativeQA), and associativity (MuSiQue, 2Wiki, HotpotQA, and LV-Eval). HippoRAG 2 surpasses other methods across all
categories, bringing it one step closer to true long-term memory.
</p>

<p align="center">
  <img align="center" src="https://github.com/OSU-NLP-Group/HippoRAG/raw/main/images/methodology.png" />
</p>
<p align="center">
  <b>Figure 2:</b> HippoRAG 2 methodology.
</p>

### Papers

* [**HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models**](https://arxiv.org/abs/2405.14831) [NeurIPS '24].
* [**From RAG to Memory: Non-Parametric Continual Learning for Large Language Models**](https://arxiv.org/abs/2502.14802) [ICML '25].

----

## Installation

Use Conda or `uv` to create a Python 3.10 environment. A project-local `.venv` is recommended for development.

```sh
conda create -n hipporag python=3.10
conda activate hipporag
pip install hipporag
```
Set only the environment variables required by the models you use:

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<path to Huggingface home directory>
export OPENAI_API_KEY=<your openai api key>   # if you want to use OpenAI model

conda activate hipporag
```

For a project-local environment managed by `uv`:

```sh
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e .
```

Install optional local-model support only when needed, for example `uv pip install -e '.[vllm]'` or `uv pip install -e '.[gritlm]'`.

HippoRAG supports `openai>=3.3.1,<4`: compatible 3.x client updates are accepted,
while a new major version requires an explicit compatibility review. To reproduce
the minimum supported SDK baseline, apply the optional constraint without changing
the library's public dependency range:

```sh
uv pip install -r requirements.txt -c constraints/openai-tested.txt
```

SDK compatibility should be checked in separate clean environments against both
the minimum baseline above and the newest allowed release. The latter can be
installed and tested with:

```sh
uv pip install -r requirements.txt --upgrade-package openai
PYTHONPATH=src python -m unittest tests.test_openai_sdk_compat
```

### Upgrading existing indexes

Version 2.0.0a5 binds persisted vectors and OpenIE state to the endpoint, deployment, model, normalization, and component identity that produced them. Existing indexes without an `index_manifest.json`, or indexes whose identity no longer matches the active configuration, are rejected rather than mixed silently. Re-index into a fresh `save_dir`; copying or fabricating only the manifest is not a safe migration. When injecting a custom embedding model, extraction LLM, or text preprocessor, set `index_identity` to a stable version string so configuration changes cannot reuse incompatible state. The default OpenIE output limits are `openie_ner_max_tokens=2048` and `openie_triple_max_tokens=4096`, sized to leave room for reasoning tokens; they are part of the OpenIE identity, so an index built with the earlier development defaults (512 and 2048) must either be re-indexed or loaded with those values set explicitly.

## Quick Start

### OpenAI

The complete runnable version is [`examples/demo_openai.py`](examples/demo_openai.py). A minimal workflow is:

Set `OPENAI_API_KEY` before using OpenAI models:

```sh
export OPENAI_API_KEY=<your OpenAI API key>
```

```python
from hipporag import HippoRAG

docs = ["George Rankin is a politician."]
queries = ["What is George Rankin's occupation?"]
with HippoRAG(save_dir="outputs", llm_model_name="gpt-4o-mini", embedding_model_name="text-embedding-3-small") as hipporag:
    hipporag.index(docs=docs)
    results = hipporag.rag_qa(queries=queries)
```

#### OpenAI-compatible endpoints

Pass custom base URLs for OpenAI-compatible LLM and embedding servers, such as a local vLLM server or a hosted gateway. No registration is needed for any OpenAI-compatible endpoint:

```python
hipporag = HippoRAG(
    save_dir=save_dir,
    llm_model_name="your-llm",
    llm_base_url="http://localhost:8000/v1",
    llm_api_key_env="MY_LLM_API_KEY",              # optional; defaults to OPENAI_API_KEY
    embedding_model_name="your-embedding-model",
    embedding_provider="openai",
    embedding_base_url="http://localhost:8001/v1",
    embedding_api_key_env="MY_EMBEDDING_API_KEY",  # optional; defaults to OPENAI_API_KEY
)
```

API keys are read from `OPENAI_API_KEY` by default. Set `llm_api_key_env` or `embedding_api_key_env` to the name of another environment variable when the LLM and embedding endpoints need different keys, for example a gateway LLM with OpenAI embeddings. When the relevant variable is unset and the endpoint is a loopback address that does not require authentication, HippoRAG supplies a client-local placeholder without changing the process-wide `OPENAI_API_KEY` environment variable.

Official OpenAI and Azure endpoints receive `max_completion_tokens`, the gateways listed under [Gateway prefixes](#gateway-prefixes) receive the parameter they document (also when reached through `llm_base_url`), and other endpoints receive the more widely supported `max_tokens`. Set `llm_supports_max_completion_tokens=True` in `BaseConfig` if your endpoint requires `max_completion_tokens`, for example for some reasoning models.

OpenAI-compatible chat and embedding endpoints must return standard `usage` data. HippoRAG fails closed instead of caching a response whose token cost cannot be accounted for.

To send a single test request through an endpoint:

```sh
python examples/demo_gateway.py --llm_name <vendor/model> --llm_base_url <base URL> --llm_api_key_env MY_LLM_API_KEY
```

#### Gateway prefixes

A few gateways also have a model-name prefix that fills in their endpoint, key variable and token parameter. The prefix is stripped before the request is sent, and the request is identical to the generic configuration above; `llm_base_url` and `llm_api_key_env` still override the defaults.

| Prefix | Equivalent `llm_base_url` | Default key variable |
| --- | --- | --- |
| `orcarouter/` | `https://api.orcarouter.ai/v1` | `ORCAROUTER_API_KEY` |
| `atlascloud/` | `https://api.atlascloud.ai/v1` | `ATLASCLOUD_API_KEY` |

These prefixes are kept for compatibility and are listed in the order they were added; they are not an endorsement, and HippoRAG is not affiliated with any gateway. New OpenAI-compatible gateways should use the generic configuration rather than a new prefix (see [CONTRIBUTING.md](CONTRIBUTING.md)).

### Amazon Bedrock

Models available through the standard Bedrock Runtime endpoint use the existing LiteLLM route. Prefix the Bedrock model ID with `bedrock/`, as shown in `examples/demo_bedrock.py`.

OpenAI models such as GPT-5.5 use Amazon Bedrock Mantle and its Responses API instead. Create a Bedrock API key, then run:

```sh
export AWS_BEARER_TOKEN_BEDROCK=<your Bedrock API key>
python examples/demo_bedrock_mantle.py
```

The corresponding configuration is:

```python
hipporag = HippoRAG(
    save_dir='outputs/bedrock-mantle',
    llm_model_name='bedrock-mantle/openai.gpt-5.5',
    llm_base_url='https://bedrock-mantle.us-east-2.api.aws/openai/v1',
    embedding_model_name=embedding_model_name,
)
```

The Mantle endpoint and model availability are region-specific. HippoRAG requires an explicit endpoint and raises an error if the Bedrock API key is missing. To use an existing AWS profile instead, construct a `BaseConfig` with `bedrock_mantle_auth='aws_credentials'`, `bedrock_aws_profile='<profile>'`, and `bedrock_region='<region>'`; this explicitly enables SigV4 authentication. Mantle response storage is disabled by default (`store=False`); pass `store=True` to `infer` only when server-side conversation state is required.

### Local Deployment (vLLM)

This simple example will illustrate how to use `hipporag` with any vLLM-compatible locally deployed LLM.

1. Run a local [OpenAI-compatible vLLM server](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#quickstart-online) with specified GPUs (make sure you leave enough memory for your embedding model).

Keep `VLLM_WORKER_MULTIPROC_METHOD=spawn` when running vLLM with multiple GPUs; it makes the required multiprocessing mode explicit and avoids CUDA initialization problems with forked workers.

```sh
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=<path to Huggingface home directory>

conda activate hipporag  # vllm should be in this environment

# Tune gpu-memory-utilization or max_model_len to fit your GPU memory, if OOM occurs
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --max_model_len 4096 --gpu-memory-utilization 0.95 
```

2. Now you can use very similar code to the one above to use `hipporag`: 

```python
save_dir = 'outputs'# Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
llm_model_name = # Any OpenAI model name
embedding_model_name = # Embedding model name (NV-Embed, GritLM or Contriever for now)
llm_base_url= # Base url for your deployed LLM (i.e. http://localhost:8000/v1)

hipporag = HippoRAG(save_dir=save_dir,
                    llm_model_name=llm_model,
                    embedding_model_name=embedding_model_name,
                    llm_base_url=llm_base_url)

# Same Indexing, Retrieval and QA as running OpenAI models above
```

## Vector Store Backends

HippoRAG stores embeddings in local Parquet files by default. It can also use
Qdrant, ChromaDB, or Milvus through `BaseConfig.vector_store_type`.
Remote collections are isolated by a per-index namespace and the backend endpoint
is fingerprinted in `index_manifest.json`. Set a stable, unique
`vector_store_namespace` when an index must keep the same remote collection after
its local `save_dir` is moved; do not reuse a namespace for unrelated indexes.

### Milvus

Install the optional Milvus dependency when you want to use this backend:

```sh
pip install "hipporag[milvus]"
```

For a source checkout, install the optional extra from the repository root:

```sh
pip install -e ".[milvus]"
```

Milvus Lite is used by default and stores its local database inside the
HippoRAG working directory, so no separate server is required:

```python
from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

config = BaseConfig(vector_store_type="milvus")

hipporag = HippoRAG(
    global_config=config,
    save_dir=save_dir,
    llm_model_name=llm_model_name,
    embedding_model_name=embedding_model_name,
)
```

To connect to Milvus server or Zilliz Cloud, set `milvus_uri` and
`milvus_token` directly or use `MILVUS_URI` and `MILVUS_TOKEN` environment
variables. `MILVUS_DB_NAME` and `MILVUS_CONSISTENCY_LEVEL` are also honored
when set:

```python
config = BaseConfig(
    vector_store_type="milvus",
    milvus_uri="http://localhost:19530",
    milvus_token=None,
    milvus_db_name=None,
    milvus_consistency_level="Session",
)
```

## Testing

Run the offline unit tests before submitting changes:

```sh
python -m unittest discover -s tests -p 'test_*.py'
python tests/integration/run_vector_stores.py
```

Provider integration scripts exercise indexing, graph reload, incremental updates, and deletion. They require the corresponding API or local model service:

| Provider | Command |
| --- | --- |
| OpenAI | `python tests/integration/run_openai.py` |
| Azure OpenAI | `python tests/integration/run_azure.py --azure_endpoint <resource-url> --azure_api_version <version> --azure_embedding_endpoint <resource-url>` |
| OpenAI-compatible endpoints | `python tests/integration/run_gateway.py --llm_model_name <vendor/model> --llm_base_url <base URL> --llm_api_key_env <key variable>` |
| Local vLLM | `python tests/integration/run_local.py` |
| Transformers | `python tests/integration/run_transformers.py` |

The local integration script expects an OpenAI-compatible server at `http://localhost:6578/v1`. See [Local Deployment](#local-deployment-vllm) for server setup.

## Reproducing our Experiments

To use our code to run experiments we recommend you clone this repository and follow the structure of the `main.py` script.

### Data for Reproducibility

We evaluated several sampled datasets in our paper, some of which are already included in the `reproduce/dataset` directory of this repo. For the complete set of datasets, please visit
our [HuggingFace dataset](https://huggingface.co/datasets/osunlp/HippoRAG_v2) and place them under `reproduce/dataset`. We also provide the OpenIE results for both `gpt-4o-mini` and `Llama-3.3-70B-Instruct` for our `musique` sample under `outputs/musique`.

To test your environment is properly set up, you can use the small dataset `reproduce/dataset/sample.json` for debugging as shown below.

### Running indexing and QA

Initialize the environmental variables and activate the environment:

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<path to Huggingface home directory>
export OPENAI_API_KEY=<your openai api key>   # if you want to use OpenAI model

conda activate hipporag
```

### OpenAI

```sh
dataset=sample  # or any other dataset under `reproduce/dataset`

# Run OpenAI model
python main.py --dataset $dataset --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2
```

Azure OpenAI uses the same entry point:

```sh
export AZURE_OPENAI_API_KEY=<your Azure OpenAI API key>
python main.py --dataset sample --llm_name gpt-4o-mini --embedding_name text-embedding-3-small \
  --azure_endpoint https://<resource>.openai.azure.com --azure_api_version <api-version> \
  --azure_chat_deployment <chat-deployment> \
  --azure_embedding_endpoint https://<resource>.openai.azure.com \
  --azure_embedding_deployment <embedding-deployment>
```

If embeddings use a different API version, also set `--azure_embedding_api_version`. Legacy full operation URLs containing the deployment and `api-version` remain accepted temporarily, but emit a migration warning.

To run the standard dense-retrieval/DPR-style baseline, add `--rag_type standard`. Both methods share the same loading and evaluation logic:

```sh
python main.py --dataset sample --rag_type standard --embedding_batch_size 1
```

### Run with vLLM (Llama)

1. As above, run a local [OpenAI-compatible vLLM server](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#quickstart-online) with specified GPU.

Do not omit `VLLM_WORKER_MULTIPROC_METHOD=spawn` when using multiple GPUs.

```sh
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=<path to Huggingface home directory>

conda activate hipporag  # vllm should be in this environment

# Tune gpu-memory-utilization or max_model_len to fit your GPU memory, if OOM occurs
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --max_model_len 4096 --gpu-memory-utilization 0.95 
```

2. Use another GPUs to run the main program in another terminal.

```sh
export CUDA_VISIBLE_DEVICES=2,3  # set another GPUs while vLLM server is running
export HF_HOME=<path to Huggingface home directory>
dataset=sample

python main.py --dataset $dataset --llm_base_url http://localhost:8000/v1 --llm_name meta-llama/Llama-3.3-70B-Instruct --embedding_name nvidia/NV-Embed-v2
```

#### Advanced: vLLM offline batch

vLLM offers an [offline batch mode](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#offline-batched-inference) for faster inference, which could bring us more than 3x faster indexing compared to vLLM online server. 

The offline implementation also sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` internally, and the export below documents the required multiprocessing mode explicitly.

1. Use the following command to run the main program with vLLM offline batch mode.

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3 # use all GPUs for faster offline indexing
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=<path to Huggingface home directory>
export OPENAI_API_KEY=''
dataset=sample

python main.py --dataset $dataset --llm_name meta-llama/Llama-3.3-70B-Instruct --openie_mode offline
```

2. After the first step, OpenIE result is saved to file. Go back to run vLLM online server and main program as described in the `Run with vLLM (Llama)` main section.

## Debugging Note

- `/reproduce/dataset/sample.json` is a small dataset specifically for debugging.
- When debugging vLLM offline mode, set `tensor_parallel_size` as `1` in `hipporag/llm/vllm_offline.py`.
- If you want to rerun a particular experiment, remember to clear the saved files, including OpenIE results and knowledge graph, e.g.,

```sh
rm reproduce/dataset/openie_results/openie_sample_results_ner_meta-llama_Llama-3.3-70B-Instruct_3.json
rm -rf outputs/sample/sample_meta-llama_Llama-3.3-70B-Instruct_nvidia_NV-Embed-v2
```
### Custom Datasets

To setup your own custom dataset for evaluation, follow the format and naming convention shown in `reproduce/dataset/sample_corpus.json` (your dataset's name should be followed by `_corpus.json`). If running an experiment with pre-defined questions, organize your query corpus according to the query file `reproduce/dataset/sample.json`, be sure to also follow our naming convention.

The corpus and optional query JSON files should have the following format:

#### Retrieval Corpus JSON

```json
[
  {
    "title": "FIRST PASSAGE TITLE",
    "text": "FIRST PASSAGE TEXT",
    "idx": 0
  },
  {
    "title": "SECOND PASSAGE TITLE",
    "text": "SECOND PASSAGE TEXT",
    "idx": 1
  }
]
```

#### (Optional) Query JSON

```json

[
  {
    "id": "sample/question_1.json",
    "question": "QUESTION",
    "answer": [
      "ANSWER"
    ],
    "answerable": true,
    "paragraphs": [
      {
        "title": "{FIRST SUPPORTING PASSAGE TITLE}",
        "text": "{FIRST SUPPORTING PASSAGE TEXT}",
        "is_supporting": true,
        "idx": 0
      },
      {
        "title": "{SECOND SUPPORTING PASSAGE TITLE}",
        "text": "{SECOND SUPPORTING PASSAGE TEXT}",
        "is_supporting": true,
        "idx": 1
      }
    ]
  }
]
```

#### (Optional) Chunking Corpus

When preparing your data, you may need to chunk each passage, as longer passage may be too complex for the OpenIE process.

## Code Structure

```
📦 .
│-- 📂 src/hipporag
│   ├── 📂 embedding_model          # Implementation of all embedding models
│   │   ├── __init__.py             # Getter function for get specific embedding model classes
|   |   ├── base.py                 # Base embedding model class `BaseEmbeddingModel` to inherit and `EmbeddingConfig`
|   |   ├── NVEmbedV2.py            # Implementation of NV-Embed-v2 model
|   |   ├── ...
│   ├── 📂 evaluation               # Implementation of all evaluation metrics
│   │   ├── __init__.py
|   |   ├── base.py                 # Base evaluation metric class `BaseMetric` to inherit
│   │   ├── qa_eval.py              # Eval metrics for QA
│   │   ├── retrieval_eval.py       # Eval metrics for retrieval
│   ├── 📂 information_extraction  # Implementation of all information extraction models
│   │   ├── __init__.py
|   |   ├── openie_openai.py        # Online OpenIE through any chat LLM class
|   |   ├── openie_vllm_offline.py  # Model for OpenIE with LLMs deployed offline with vLLM
|   |   ├── openie_transformers_offline.py  # Offline OpenIE with Transformers
│   ├── 📂 llm                      # Classes for inference with large language models
│   │   ├── __init__.py             # Getter function
|   |   ├── base.py                 # Config class for LLM inference and base LLM inference class to inherit
|   |   ├── openai_gpt.py           # OpenAI, Azure OpenAI and any OpenAI-compatible endpoint (e.g. a vLLM server)
|   |   ├── gateways.py             # Frozen registry of gateway prefixes and the shared max-token parameter rule
|   |   ├── gateway_llm.py          # Class for the registered gateway prefixes
|   |   ├── bedrock_llm.py          # Amazon Bedrock Runtime through LiteLLM
|   |   ├── bedrock_mantle.py       # Amazon Bedrock Mantle (Responses API)
|   |   ├── transformers_llm.py     # Local inference with Transformers
|   |   ├── vllm_offline.py         # Class for inference using the vLLM API directly
│   ├── 📂 prompts                  # Prompt templates and prompt template manager class
|   │   ├── 📂 dspy_prompts         # Prompts for filtering
|   │   │   ├── ...
|   │   ├── 📂 templates            # All prompt templates for template manager to load
|   │   │   ├── README.md           # Documentations of usage of prompte template manager and prompt template files
|   │   │   ├── __init__.py
|   │   │   ├── triple_extraction.py
|   │   │   ├── ...
│   │   ├── __init__.py
|   |   ├── linking.py              # Instruction for linking
|   |   ├── prompt_template_manager.py  # Implementation of prompt template manager
│   ├── 📂 utils                    # All utility functions used across this repo (the file name indicates its relevant usage)
│   │   ├── config_utils.py         # We use only one config across all modules and its setup is specified here
|   |   ├── ...
│   ├── __init__.py
│   ├── HippoRAG.py          # Highest level class for initiating retrieval, question answering, and evaluations
│   ├── embedding_store.py   # Storage database to load, manage and save embeddings for passages, entities and facts.
│   ├── rerank.py            # Reranking and filtering methods
│-- 📂 examples              # Minimal provider-specific usage examples
│-- 📂 tests
│   ├── 📂 integration       # Manual provider and vector-store integration checks
│   ├── test_gateway_llm.py
│   ├── test_regressions.py
│   ├── ...
│-- 📂 reproduce/dataset     # Sample and paper evaluation datasets
│-- 📜 main.py               # Unified HippoRAG, Azure, and standard-RAG experiment entry point
│-- 📜 README.md
│-- 📜 requirements.txt   # Dependencies list
│-- 📜 .gitignore         # Files to exclude from Git
```

## Contact

Questions or issues? File an issue or contact 
[Bernal Jiménez Gutiérrez](mailto:jimenezgutierrez.1@osu.edu),
[Yiheng Shu](mailto:shu.251@osu.edu),
[Yu Su](mailto:su.809@osu.edu),
The Ohio State University

## Citation

If you find this work useful, please consider citing our papers:

### HippoRAG 2
```
@misc{gutiérrez2025ragmemorynonparametriccontinual,
      title={From RAG to Memory: Non-Parametric Continual Learning for Large Language Models}, 
      author={Bernal Jiménez Gutiérrez and Yiheng Shu and Weijian Qi and Sizhe Zhou and Yu Su},
      year={2025},
      eprint={2502.14802},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14802}, 
}
```

### HippoRAG

```
@inproceedings{gutiérrez2024hipporag,
      title={HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models}, 
      author={Bernal Jiménez Gutiérrez and Yiheng Shu and Yu Gu and Michihiro Yasunaga and Yu Su},
      booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
      year={2024},
      url={https://openreview.net/forum?id=hkujvAPVsg}
 ```

## TODO:

- [x] Add support for more embedding models
- [x] Add support for embedding endpoints
- [x] Add support for vector database integration

Please feel free to open an issue or PR if you have any questions or suggestions.
