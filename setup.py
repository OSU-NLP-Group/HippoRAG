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
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "vllm>=0.4.0",
        "openai>=1.0.0",
        "litellm>=1.0.0",
        "gritlm>=1.0.0",
        "networkx>=3.0",
        "python_igraph>=0.10.0",
        "tiktoken>=0.5.0",
        "pydantic>=2.0.0",
        "tenacity>=8.0.0",
        "einops", # No version specified
        "tqdm", # No version specified
        "boto3", # No version specified
    ]
)