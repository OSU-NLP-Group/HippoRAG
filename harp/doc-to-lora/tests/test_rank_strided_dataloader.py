from types import SimpleNamespace

from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import IterableDataset
from transformers import Trainer

from ctx_to_lora.trainer import ModulatedModelTrainer


class _RankStridedFixture(IterableDataset):
    rank_strided_assignment = True

    def __iter__(self):
        yield {"value": 1}
        yield {"value": 2}


def test_rank_strided_iterable_is_not_sharded_twice():
    trainer = object.__new__(ModulatedModelTrainer)
    trainer.args = SimpleNamespace(
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_drop_last=False,
        dataloader_prefetch_factor=None,
        process_index=3,
    )
    trainer.data_collator = lambda rows: rows
    trainer._get_collator_with_removed_columns = (
        lambda collator, description: collator
    )
    dataset = _RankStridedFixture()

    dataloader = ModulatedModelTrainer._get_dataloader(
        trainer,
        dataset=dataset,
        description="rank-strided fixture",
        batch_size=1,
        is_training=True,
    )

    assert dataloader.dataset is dataset
    assert list(dataloader) == [[{"value": 1}], [{"value": 2}]]


def test_snapshot_training_uses_ddp_gradient_bucket_views(monkeypatch):
    handler = DistributedDataParallelKwargs(find_unused_parameters=True)
    monkeypatch.setattr(
        Trainer,
        "_build_accelerator_args",
        lambda self, **kwargs: {"kwargs_handlers": [handler], **kwargs},
    )
    trainer = object.__new__(ModulatedModelTrainer)
    trainer._snapshot_ddp_gradient_bucket_views = True

    accelerator_args = trainer._build_accelerator_args(mixed_precision="bf16")

    assert accelerator_args["mixed_precision"] == "bf16"
    assert handler.gradient_as_bucket_view is True
