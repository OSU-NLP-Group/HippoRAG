# Doc-to-LoRA for HARP

This fork contains the repository-memory Doc-to-LoRA implementation used by
HARP. It extends the upstream context-to-LoRA generator with complete-snapshot
training and three repository composition modes: Concat, TIES, and exact
streaming TIES.

Install the package using the checked-in `pyproject.toml`, then launch training
from the parent HARP repository with `scripts/train_doc_to_lora.sbatch`.

The production configurations are under `configs/repoqa/`. Datasets,
checkpoints, logs, predictions, and reports are external artifacts and are not
stored in this repository.

