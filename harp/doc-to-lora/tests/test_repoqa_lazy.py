import json
from pathlib import Path
from tempfile import TemporaryDirectory

import datasets
import pyarrow as pa
import pyarrow.parquet as pq

from ctx_to_lora.data.definitions import IGNORE_INDEX
from ctx_to_lora.data.repoqa_lazy import (
    FrozenRepoQADataset,
    load_frozen_repoqa_dataset,
)


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        ids = [10 + i for i in range(sum(len(item["content"]) for item in messages) % 7 + 3)]
        if messages[-1]["role"] == "assistant":
            masks = [0] * (len(ids) - 2) + [1, 1]
            return {"input_ids": ids, "assistant_masks": masks}
        return {"input_ids": ids}


def make_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    chunks = pa.Table.from_pylist(
        [
            {
                "chunk_id": "c0",
                "payload_text": '<<<FILE path="a.py">>>\na = 1\n<<<END FILE>>>\n',
            },
            {
                "chunk_id": "c1",
                "payload_text": '<<<FILE path="b.py">>>\nb = 2\n<<<END FILE>>>\n',
            },
        ]
    )
    pq.write_table(chunks, repo / "chunks.parquet", row_group_size=1)
    snapshots = pa.Table.from_pylist(
        [
            {"commit_sha": "abc", "chunk_index": 0, "chunk_id": "c0"},
            {"commit_sha": "abc", "chunk_index": 1, "chunk_id": "c1"},
        ]
    )
    pq.write_table(snapshots, repo / "snapshots.parquet")
    return repo


def test_lazy_repoqa_hydrates_complete_snapshot_without_storing_all_chunk_ids():
    with TemporaryDirectory() as directory:
        repo = make_repo(Path(directory))
        rows = datasets.Dataset.from_dict(
            {
                "repo_id": ["owner/repo"],
                "commit_sha": ["abc"],
                "repo_dir": [str(repo)],
                "question": ["What values are assigned?"],
                "answer": ["One and two."],
                "chunk_ids": [[]],
                "bm25_chunk_ids": [["c1"]],
                "use_all_chunks": [True],
            }
        )
        dataset = FrozenRepoQADataset(
            rows,
            TinyTokenizer(),
            TinyTokenizer(),
            "concat",
            max_qas_len=64,
            max_context_tokens=64,
            chunk_cache_mb=1,
        )
        item = dataset[0]
        assert item["n_ctx_chunks"] == [2]
        assert list(item["ctx_position_ids"]).count(0) == 2
        assert sum(value != IGNORE_INDEX for value in item["labels"]) == 2


def test_lazy_repoqa_bm25_uses_only_preselected_chunks():
    with TemporaryDirectory() as directory:
        repo = make_repo(Path(directory))
        rows = datasets.Dataset.from_dict(
            {
                "repo_id": ["owner/repo"],
                "commit_sha": ["abc"],
                "repo_dir": [str(repo)],
                "question": ["Where is b?"],
                "answer": ["b.py"],
                "chunk_ids": [["c0"]],
                "bm25_chunk_ids": [["c1"]],
                "use_all_chunks": [False],
            }
        )
        dataset = FrozenRepoQADataset(
            rows,
            TinyTokenizer(),
            TinyTokenizer(),
            "bm25_topk_ties",
            max_qas_len=64,
            max_context_tokens=64,
            chunk_cache_mb=1,
        )
        assert dataset[0]["n_ctx_chunks"] == [1]


def test_lazy_repoqa_groups_questions_with_the_exact_same_context():
    with TemporaryDirectory() as directory:
        repo = make_repo(Path(directory))
        rows = datasets.Dataset.from_dict(
            {
                "repo_id": ["owner/repo"] * 3,
                "commit_sha": ["abc"] * 3,
                "repo_dir": [str(repo)] * 3,
                "question": ["Question one?", "Question two?", "Question three?"],
                "answer": ["One.", "Two.", "Three."],
                "chunk_ids": [["c0"]] * 3,
                "bm25_chunk_ids": [["c0"]] * 3,
                "use_all_chunks": [False] * 3,
            }
        )
        dataset = FrozenRepoQADataset(
            rows,
            TinyTokenizer(),
            TinyTokenizer(),
            "concat",
            max_qas_len=64,
            max_context_tokens=64,
            chunk_cache_mb=1,
            max_qas_per_sample=2,
        )
        assert dataset.group_indices == [(0, 1), (2,)]
        first = dataset[0]
        assert first["n_queries"] == [2]
        assert list(first["position_ids"]).count(0) == 2
        assert sum(value != IGNORE_INDEX for value in first["labels"]) == 4


def test_lazy_repoqa_training_filters_are_explicit_and_eval_safe():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        repo = make_repo(root)
        index = root / "rows.repoqa.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "repo_id": "owner/repo",
                        "commit_sha": "abc",
                        "repo_dir": str(repo),
                        "question": "Where is a?",
                        "answer": "a.py",
                        "chunk_ids": [],
                        "bm25_chunk_ids": ["c0"],
                        "use_all_chunks": True,
                        "num_repo_chunks": 2,
                        "bm25_evidence_recall": 1.0,
                    },
                    {
                        "repo_id": "owner/repo",
                        "commit_sha": "abc",
                        "repo_dir": str(repo),
                        "question": "Where is b?",
                        "answer": "b.py",
                        "chunk_ids": [],
                        "bm25_chunk_ids": ["c1"],
                        "use_all_chunks": True,
                        "num_repo_chunks": 9,
                        "bm25_evidence_recall": 0.0,
                    },
                ]
            ),
            index,
        )
        tokenizer = TinyTokenizer()
        dataset = load_frozen_repoqa_dataset(
            [str(index)],
            tokenizer,
            tokenizer,
            "bm25_topk_ties",
            max_qas_len=64,
            max_context_tokens=64,
            chunk_cache_mb=1,
            split="train",
            max_repo_chunks=8,
            require_bm25_full_evidence=True,
        )
        assert len(dataset) == 1
        try:
            load_frozen_repoqa_dataset(
                [str(index)],
                tokenizer,
                tokenizer,
                "bm25_topk_ties",
                max_qas_len=64,
                max_context_tokens=64,
                chunk_cache_mb=1,
                split="validation",
                require_bm25_full_evidence=True,
            )
        except ValueError as error:
            assert "training-only" in str(error)
        else:
            raise AssertionError("BM25 evidence filtering must be rejected for eval")
