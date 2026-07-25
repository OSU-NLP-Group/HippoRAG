import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="hipporag",
    version="2.0.0-alpha.4",
    author="Bernal Jimenez Gutierrez",
    author_email="jimenezgutierrez.1@osu.edu",
    description="A powerful graph-based RAG framework that enables LLMs to identify and leverage connections within new knowledge for improved retrieval.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OSU-NLP-Group/HippoRAG",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    include_package_data=True,
    package_data={"hipporag": ["prompts/dspy_prompts/*.json"]},
    python_requires=">=3.10",
    install_requires=[
        "openai>=2.0",
        "litellm>=1.73",
        "vllm>=0.10",
        "gritlm>=1.0.2",
        "torch>=2.5",
        "transformers>=4.45",
        "networkx>=3.4.2",
        "pydantic>=2.10",
        "python_igraph>=0.11.8",
        "tenacity>=8.5",
        "tiktoken>=0.7",
        "nest_asyncio",
        "numpy",
        "scipy",
        "tqdm",
        "einops",
        "boto3",
        "pyarrow",
        "pandas",
        "outlines>=1.0",
        "requests",
    ],
    extras_require={
        "milvus": ["pymilvus[milvus_lite]>=2.4.2"],
        "dev": ["pytest>=8"],
    },
)
