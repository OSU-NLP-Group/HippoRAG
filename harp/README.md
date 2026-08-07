# HARP

HARP contains the reusable implementation used to train repository-conditioned
LoRA adapters with Doc-to-LoRA and Code2LoRA. Generated datasets, checkpoints,
logs, predictions, analyses, and presentation material are intentionally kept
outside this repository.

The source tree has two implementation directories tracked by the single HARP
Git repository:

- `doc-to-lora/`: the Doc-to-LoRA fork with Concat, TIES, and exact streaming
  TIES repository composition.
- `code2lora/`: the repository-embedding-conditioned Code2LoRA implementation.

Shared corpus-building and inference programs live under `pipelines/`. Generic
Slurm launchers live under `scripts/`.

The canonical repository-to-64K-chunk builder and its freeze, repack, audit,
and test utilities live under `pipelines/chunking/`.

The frozen 104-question evaluation panel, including open-answer references,
MCQ choices, and evidence spans, is stored under `benchmarks/repoqa_104/`.

Checkpoint weights are intentionally excluded from version control. Publish
them through an approved model registry or artifact store rather than ordinary
Git history.

Start with:

- [Data pipeline](docs/DATA_PIPELINE.md)
- [Doc-to-LoRA implementation](docs/DOC_TO_LORA.md)
- [Code2LoRA implementation](docs/CODE2LORA.md)
- [Launching jobs](docs/LAUNCHING.md)
- [Reproducibility contract](docs/REPRODUCIBILITY.md)

The launchers require explicit external paths. They do not assume a personal
home directory, allocation ID, checkpoint, or dated output location.
