# Code2LoRA for HARP

This repository contains the checkpoint-compatible Code2LoRA implementation
used by HARP. The frozen Gemma model receives LoRA factors generated from a
validated repository embedding.

Install the package in an isolated environment:

```bash
pip install -e .
```

Training is launched from the parent HARP repository with
`scripts/train_code2lora.sbatch`. Data, embeddings, checkpoints, and logs are
external artifacts and are not stored here.

