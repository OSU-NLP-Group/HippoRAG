import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="hipporag",
    version="2.0.0a5",
    author="Bernal Jimenez Gutierrez",
    author_email="jimenezgutierrez.1@osu.edu",
    description="A powerful graph-based RAG framework that enables LLMs to identify and leverage connections within new knowledge for improved retrieval.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OSU-NLP-Group/HippoRAG",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.5.1",
        "transformers==4.45.2",
        "openai>=3.3.1,<4",
        "litellm==1.73.1",
        "networkx==3.4.2",
        "python_igraph==0.11.8",
        "tiktoken==0.7.0",
        "pydantic>=2.10.4,<3",
        "tenacity==8.5.0",
        "einops", # No version specified
        "tqdm", # No version specified
        "boto3", # No version specified
        "nest_asyncio",
        "numpy",
        "pandas",
        "pyarrow",
        "requests",
        "scipy",
        "filelock",
        "httpx>=0.27,<1",
    ],
    extras_require={
        "milvus": ["pymilvus[milvus_lite]>=2.4.2"],
        "qdrant": ["qdrant-client>=1.9"],
        "chroma": ["chromadb>=0.5"],
        "transformers-embedding": ["sentence-transformers>=3.0"],
        "gritlm": ["gritlm==1.0.2"],
        "vllm": ["vllm==0.6.6.post1", "outlines"],
        "mcp": ["mcp>=2.2,<3"],
    },
    entry_points={"console_scripts": ["hipporag-mcp=hipporag.mcp_server:main"]},
    package_data={"hipporag": ["prompts/dspy_prompts/*.json"]},
    include_package_data=True,
)
