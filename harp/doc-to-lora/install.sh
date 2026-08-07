curl -LsSf https://astral.sh/uv/install.sh | sh
uv self update
uv venv --python 3.10 --seed
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --torch-backend=cu124
uv sync
uv pip install tokenizers==0.21.0
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install flashinfer-python==0.2.2 -i https://flashinfer.ai/whl/cu124/torch2.6

# download squad dataset
HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download --repo-type dataset rajpurkar/squad --local-dir data/raw_datasets/squad
uv run data/build_drop_compact.py
uv run data/build_pwc_compact.py
uv run data/build_ropes_compact.py
uv run data/build_squad_compact.py

# optional: needed for gated models
# uv run huggingface-cli login

# optional: needed for logging with wandb
# wandb login

# optional: dev
# uv run pre-commit install
