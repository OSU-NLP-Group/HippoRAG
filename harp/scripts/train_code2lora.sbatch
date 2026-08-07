#!/usr/bin/env bash
#SBATCH --job-name=harp-code2lora-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
set -euo pipefail

HARP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
READY="${HARP_DATA_READY:?set HARP_DATA_READY to the frozen corpus READY.json}"
EMBEDDING_READY="${HARP_BASELINE_DATA_READY:?set HARP_BASELINE_DATA_READY to the embedding-index READY.json}"
OUTPUT="${HARP_OUTPUT_ROOT:?set HARP_OUTPUT_ROOT outside the source tree}"
ENV_DIR="${HARP_CODE2LORA_ENV:?set HARP_CODE2LORA_ENV to the Code2LoRA environment}"
STORAGE="${HARP_STORAGE_ROOT:?set HARP_STORAGE_ROOT to AD-local shared storage}"

for path in "${READY}" "${EMBEDDING_READY}" "${ENV_DIR}/bin/python"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 3; }
done

SCRATCH="/local/users/${USER}/harp-code2lora-${SLURM_JOB_ID:-manual}"
case "${SCRATCH}" in /local/users/"${USER}"/harp-code2lora-*) ;; *) exit 4 ;; esac
cleanup() { rm -rf -- "${SCRATCH}"; }
trap cleanup EXIT INT TERM
mkdir -p "${SCRATCH}/tmp" "${OUTPUT}"

export TMPDIR="${SCRATCH}/tmp"
export HF_DATASETS_CACHE="${SCRATCH}/hf-datasets"
export HF_HOME="${HF_HOME:-${STORAGE}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${STORAGE}/models/huggingface}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${HARP_ROOT}/code2lora/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

args=(
  --repoqa-ready "${READY}"
  --baseline-data-ready "${EMBEDDING_READY}"
  --output-dir "${OUTPUT}"
  --target-qas "${HARP_TARGET_QAS:-10000000}"
  --token-budget "${HARP_TOKEN_BUDGET:-4096}"
  --eval-token-budget "${HARP_EVAL_TOKEN_BUDGET:-4096}"
  --ce-chunk-tokens "${HARP_CE_CHUNK_TOKENS:-512}"
  --checkpoint-qas "${HARP_CHECKPOINT_QAS:-1450506,3589498,6490000,10000000}"
  --lr "${HARP_LEARNING_RATE:-1e-4}"
  --weight-decay "${HARP_WEIGHT_DECAY:-0.01}"
  --warmup-ratio "${HARP_WARMUP_RATIO:-0.03}"
  --max-grad-norm "${HARP_MAX_GRAD_NORM:-1.0}"
  --seed "${HARP_SEED:-3407}"
  --local-files-only
)
if [[ -n "${HARP_RESUME_CHECKPOINT:-}" ]]; then
  args+=(--resume "${HARP_RESUME_CHECKPOINT}")
fi

"${ENV_DIR}/bin/python" -m repotune_issuefix.train_code2lora_repoqa "${args[@]}"
