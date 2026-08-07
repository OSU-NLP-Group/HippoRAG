#!/usr/bin/env bash
#SBATCH --job-name=harp-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail

HARP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${HARP_EVAL_MODE:?set HARP_EVAL_MODE}"
DATASET="${HARP_EVAL_DATASET:?set HARP_EVAL_DATASET}"
OUTPUT="${HARP_EVAL_OUTPUT:?set HARP_EVAL_OUTPUT outside the source tree}"
ENV_DIR="${HARP_EVAL_ENV:-${HARP_D2L_ENV:?set HARP_EVAL_ENV or HARP_D2L_ENV}}"
STORAGE="${HARP_STORAGE_ROOT:?set HARP_STORAGE_ROOT to AD-local shared storage}"

case "${MODE}" in base|concat|ties|streaming_ties_exact|code2lora) ;; *) echo "Unsupported mode: ${MODE}" >&2; exit 2 ;; esac
SCRATCH="/local/users/${USER}/harp-eval-${SLURM_JOB_ID:-manual}"
case "${SCRATCH}" in /local/users/"${USER}"/harp-eval-*) ;; *) exit 4 ;; esac
cleanup() { rm -rf -- "${SCRATCH}"; }
trap cleanup EXIT INT TERM
mkdir -p "${SCRATCH}/tmp" "${OUTPUT}"

export TMPDIR="${SCRATCH}/tmp"
export HF_HOME="${HF_HOME:-${STORAGE}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${STORAGE}/models/huggingface}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${HARP_ROOT}/doc-to-lora/src:${HARP_ROOT}/code2lora/src:${HARP_ROOT}/pipelines/evaluation:${PYTHONPATH:-}"

args=(
  --mode "${MODE}"
  --dataset-jsonl "${DATASET}"
  --output-dir "${OUTPUT}"
  --formats open mcq
  --retrieval-budgets ${HARP_EVAL_RETRIEVAL_BUDGETS:-none}
  --max-new-tokens-open "${HARP_EVAL_MAX_NEW_TOKENS:-256}"
  --snapshot-shard-index "${HARP_EVAL_SHARD_INDEX:-0}"
  --num-snapshot-shards "${HARP_EVAL_NUM_SHARDS:-1}"
  --local-files-only
)
if [[ "${MODE}" != base ]]; then
  args+=(--checkpoint "${HARP_EVAL_CHECKPOINT:?set HARP_EVAL_CHECKPOINT}")
fi
if [[ -n "${HARP_EVAL_RETRIEVAL:-}" ]]; then
  args+=(--retrieval-jsonl "${HARP_EVAL_RETRIEVAL}")
fi
if [[ "${MODE}" == concat || "${MODE}" == ties || "${MODE}" == streaming_ties_exact ]]; then
  args+=(--chunk-root "${HARP_EVAL_CHUNK_ROOT:?set HARP_EVAL_CHUNK_ROOT}")
fi
if [[ "${MODE}" == code2lora ]]; then
  args+=(--baseline-data-ready "${HARP_BASELINE_DATA_READY:?set HARP_BASELINE_DATA_READY}")
fi

"${ENV_DIR}/bin/python" "${HARP_ROOT}/pipelines/evaluation/run_doc_to_lora_broad_eval.py" "${args[@]}"
