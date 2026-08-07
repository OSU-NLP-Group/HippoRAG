#!/usr/bin/env bash
#SBATCH --job-name=harp-doc2lora-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=24:00:00
set -euo pipefail

HARP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${HARP_D2L_VARIANT:-concat}"
READY="${HARP_DATA_READY:?set HARP_DATA_READY to the frozen corpus READY.json}"
OUTPUT="${HARP_OUTPUT_ROOT:?set HARP_OUTPUT_ROOT outside the source tree}"
ENV_DIR="${HARP_D2L_ENV:?set HARP_D2L_ENV to the Doc-to-LoRA environment}"
STORAGE="${HARP_STORAGE_ROOT:?set HARP_STORAGE_ROOT to AD-local shared storage}"
D2L_DIR="${HARP_ROOT}/doc-to-lora"
GPU_COUNT="${SLURM_GPUS_ON_NODE:-8}"

case "${VARIANT}" in
  concat) CONFIG="${D2L_DIR}/configs/repoqa/gemma4_e2b_snapshot_memory_64k_k9_10m.yaml" ;;
  ties) CONFIG="${D2L_DIR}/configs/repoqa/gemma4_e2b_snapshot_memory_64k_k9_10m_ties_r64.yaml" ;;
  streaming_ties_exact) CONFIG="${D2L_DIR}/configs/repoqa/gemma4_e2b_snapshot_memory_64k_k9_10m_streaming_ties_exact_r64.yaml" ;;
  *) echo "Unknown HARP_D2L_VARIANT: ${VARIANT}" >&2; exit 2 ;;
esac

for path in "${READY}" "${CONFIG}" "${ENV_DIR}/bin/accelerate"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 3; }
done

SCRATCH="/local/users/${USER}/harp-doc2lora-${SLURM_JOB_ID:-manual}"
case "${SCRATCH}" in /local/users/"${USER}"/harp-doc2lora-*) ;; *) exit 4 ;; esac
cleanup() { rm -rf -- "${SCRATCH}"; }
trap cleanup EXIT INT TERM
mkdir -p "${SCRATCH}/tmp" "${SCRATCH}/datasets" "${OUTPUT}"

export TMPDIR="${SCRATCH}/tmp"
export DOC_TO_LORA_DATA_DIR="${SCRATCH}/datasets"
export HF_DATASETS_CACHE="${SCRATCH}/hf-datasets"
export HF_HOME="${HF_HOME:-${STORAGE}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${STORAGE}/models/huggingface}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${D2L_DIR}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

stage_steps() {
  jq -r ".partitions[\"$1/train\"].optimizer_steps" "${READY}"
}

latest_checkpoint() {
  find "$1" -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-
}

run_stage() {
  local stage="$1" learning_rate="$2" init_checkpoint="${3:-}"
  local stage_output="${OUTPUT}/${stage}"
  local max_steps warmup resume
  max_steps="$(stage_steps "${stage}")"
  warmup="$(( (max_steps + 9) / 10 ))"
  mkdir -p "${stage_output}"
  resume="$(latest_checkpoint "${stage_output}" || true)"
  local extra=()
  if [[ -n "${resume}" ]]; then
    extra+=("--resume_from_checkpoint=${resume}")
  elif [[ -n "${init_checkpoint}" ]]; then
    extra+=("--from_pretrained_checkpoint=${init_checkpoint}")
  fi
  (
    cd "${D2L_DIR}"
    "${ENV_DIR}/bin/accelerate" launch \
      --config_file accelerate_config.yaml \
      --num_processes="${GPU_COUNT}" --gpu_ids=all \
      --main_process_port="${MASTER_PORT:-29560}" \
      train.py "${CONFIG}" \
      "--train_ds_names=${READY}" "--val_ds_names=${READY}" \
      "--repoqa_stage=${stage}" "--output_root=${stage_output}" \
      "--max_steps=${max_steps}" "--learning_rate=${learning_rate}" \
      "--warmup_steps=${warmup}" --ignore_data_skip=false "${extra[@]}"
  )
}

run_stage stage1 4e-5 "${HARP_RESUME_CHECKPOINT:-}"
STAGE1_WEIGHTS="$(find "${OUTPUT}/stage1" -name pytorch_model.bin -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
[[ -n "${STAGE1_WEIGHTS}" ]] || { echo "No completed Stage 1 weights found" >&2; exit 5; }
run_stage stage2a 2e-5 "${STAGE1_WEIGHTS}"
