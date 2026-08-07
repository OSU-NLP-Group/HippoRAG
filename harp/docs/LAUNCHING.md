# Launching jobs

All substantial data processing, training, and inference should run through
Slurm. The checked-in launchers request CPUs and GPUs explicitly and clean their
job-local scratch directories on exit.

## Required external paths

Set these for the relevant workload:

```bash
export HARP_STORAGE_ROOT=/mnt/shared_ad3_mt1/$USER/harp
export HARP_D2L_ENV=/mnt/shared_ad3_mt1/$USER/envs/doc-to-lora
export HARP_CODE2LORA_ENV=/mnt/shared_ad3_mt1/$USER/envs/code2lora
export HARP_DATA_READY=/mnt/shared_ad3_mt1/$USER/harp/data/READY.json
export HARP_OUTPUT_ROOT=/mnt/shared_ad3_mt1/$USER/harp/runs/my-run
```

The model and dataset caches should be on the same AD storage as the compute
allocation. Generated outputs must not be written into this source repository.

## Doc-to-LoRA

Choose `concat`, `ties`, or `streaming_ties_exact`:

```bash
mkdir -p "${HARP_OUTPUT_ROOT}"
sbatch --output="${HARP_OUTPUT_ROOT}/slurm-%j.out" \
  --export=ALL,HARP_D2L_VARIANT=concat \
  scripts/train_doc_to_lora.sbatch
```

Set `HARP_RESUME_CHECKPOINT` only when explicitly resuming a known checkpoint.
Stage 2 otherwise initializes from the completed Stage 1 model weights.

## Code2LoRA

Code2LoRA needs the frozen Doc-to-LoRA corpus manifest and the validated
embedding-index manifest:

```bash
export HARP_BASELINE_DATA_READY=/mnt/shared_ad3_mt1/$USER/harp/data/code2lora/READY.json
mkdir -p "${HARP_OUTPUT_ROOT}"
sbatch --output="${HARP_OUTPUT_ROOT}/slurm-%j.out" scripts/train_code2lora.sbatch
```

## Inference

The shared evaluator supports base, Doc-to-LoRA composition modes, and
Code2LoRA. Required inputs are explicit:

```bash
export HARP_EVAL_MODE=concat
export HARP_EVAL_CHECKPOINT=/path/to/pytorch_model.bin
export HARP_EVAL_DATASET="$PWD/benchmarks/repoqa_104/swefixer_targeted_retrieval_104_step500.jsonl"
export HARP_EVAL_CHUNK_ROOT=/path/to/canonical/64k/chunks
export HARP_EVAL_OUTPUT=/mnt/shared_ad3_mt1/$USER/harp/eval/my-run
mkdir -p "${HARP_EVAL_OUTPUT}"
sbatch --output="${HARP_EVAL_OUTPUT}/slurm-%j.out" scripts/evaluate.sbatch
```

Optionally set `HARP_EVAL_RETRIEVAL` to a frozen retrieval JSONL and
`HARP_EVAL_RETRIEVAL_BUDGETS` to a space-separated list such as
`"none 500 1k 2k 8k"`.
