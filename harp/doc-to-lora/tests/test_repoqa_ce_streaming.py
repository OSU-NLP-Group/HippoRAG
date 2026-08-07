import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from ctx_to_lora.data.repoqa_ce_streaming import (
    READY_FORMAT,
    SNAPSHOT_MEMORY_READY_FORMAT,
    RepoQACEStreamingDataset,
    canonical_context_hash,
    file_sha256,
)


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": list(range(3, 3 + len(messages[-1]["content"]))) }


def build_fixture(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    payload = "abcde"
    payload_hash = hashlib.sha256(payload.encode()).hexdigest()
    pq.write_table(
        pa.Table.from_pylist(
            [{"chunk_id": "c0", "payload_text": payload, "payload_sha256": payload_hash}]
        ),
        repo / "chunks.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"commit_sha": "abc", "chunk_index": 0, "chunk_id": "c0"}]
        ),
        repo / "snapshots.parquet",
    )
    qa_path = root / "qas.parquet"
    qa_rows = [
        {
            "logical_example_id": f"logical-{index}",
            "source_qa_id": f"source-{index}",
            "input_ids": [1, 2, 10 + index, 20 + index],
            "response_start": 2,
            "response_end": 4,
            "duplicate_multiplicity": 1,
            "source_family": "unit",
            "qa_family": "ast",
            "task_category": "lookup",
        }
        for index in range(5)
    ]
    pq.write_table(pa.Table.from_pylist(qa_rows), qa_path)
    context_hash = canonical_context_hash(
        "owner/repo", "abc", ["c0"], ["c0"], [payload_hash]
    )
    groups_path = root / "groups.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "context_group_id": f"group-{index}",
                    "stage": "stage1",
                    "repo_id": "owner/repo",
                    "commit_sha": "abc",
                    "repo_dir": str(repo),
                    "selected_chunk_ids": ["c0"],
                    "selected_context_hash": context_hash,
                    "qa_file": str(qa_path),
                    "qa_row_group": 0,
                    "qa_start": start,
                    "qa_count": count,
                }
                for index, (start, count) in enumerate(((0, 2), (2, 2), (4, 1)))
            ]
        ),
        groups_path,
    )
    ready_path = root / "READY.json"
    ready_path.write_text(
        json.dumps(
            {
                "format": READY_FORMAT,
                "loss": "answer_token_ce",
                "qa_token_budget": 8,
                "partitions": {
                    "stage1/train": {
                        "logical_qas": 5,
                        "physical_qa_rows": 5,
                        "group_manifests": [str(groups_path)],
                        "group_manifest_sha256": {
                            str(groups_path): file_sha256(groups_path)
                        },
                    }
                },
            }
        )
    )
    return ready_path


def dataset(ready_path: Path) -> RepoQACEStreamingDataset:
    return RepoQACEStreamingDataset(
        ready_path,
        "stage1",
        "train",
        TinyTokenizer(),
        max_context_tokens=512,
        qa_token_budget=8,
        cache_mb=1,
        seed=17,
    )


def test_streaming_token_budget_and_exact_kill_resume_cursor():
    with TemporaryDirectory() as directory:
        ready_path = build_fixture(Path(directory))
        first = dataset(ready_path)
        iterator = iter(first)
        consumed = next(iterator)
        assert 1 <= consumed["n_queries"][0] <= 2
        state = first.state_dict()
        assert state["within_group_qa_offset"] == 0

        resumed = dataset(ready_path)
        resumed.load_state_dict(state)
        remaining = [logical for item in resumed for logical in item["logical_example_ids"]]
        all_ids = consumed["logical_example_ids"] + remaining
        assert set(all_ids) == {f"logical-{index}" for index in range(5)}
        assert len(all_ids) == len(set(all_ids)) == 5
        assert resumed.state_dict()["logical_qas_consumed"] == 5


def test_final_yield_marks_cursor_exhausted_without_extra_read():
    with TemporaryDirectory() as directory:
        ready_path = build_fixture(Path(directory))
        stream = dataset(ready_path)
        iterator = iter(stream)

        next(iterator)
        next(iterator)
        assert not stream.exhausted
        next(iterator)
        assert stream.exhausted
        assert stream.permutation_position >= stream.total_groups


def test_two_rank_streams_have_equal_physical_rounds_with_zero_weight_padding():
    with TemporaryDirectory() as directory:
        ready_path = build_fixture(Path(directory))
        old_rank = os.environ.get("RANK")
        old_world = os.environ.get("WORLD_SIZE")
        try:
            per_rank = []
            for rank in (0, 1):
                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = "2"
                items = list(dataset(ready_path))
                per_rank.append(items)
            assert len(per_rank[0]) == len(per_rank[1]) == 2
            real_ids = [
                logical_id
                for items in per_rank
                for item in items
                if not item["is_ddp_padding"]
                for logical_id in item["logical_example_ids"]
            ]
            assert set(real_ids) == {f"logical-{index}" for index in range(5)}
            assert len(real_ids) == len(set(real_ids)) == 5
            padding = [
                item for items in per_rank for item in items if item["is_ddp_padding"]
            ]
            assert len(padding) == 1
            assert padding[0]["logical_qa_count"] == [0]
            assert padding[0]["qa_weights"].sum() == 0
        finally:
            if old_rank is None:
                os.environ.pop("RANK", None)
            else:
                os.environ["RANK"] = old_rank
            if old_world is None:
                os.environ.pop("WORLD_SIZE", None)
            else:
                os.environ["WORLD_SIZE"] = old_world


def test_snapshot_memory_reuses_one_context_for_multiple_frozen_qa_packs():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ready_path = build_fixture(root)
        ready = json.loads(ready_path.read_text())
        ready["format"] = SNAPSHOT_MEMORY_READY_FORMAT
        ready["physical_qa_packs"] = 2
        groups_path = Path(
            ready["partitions"]["stage1/train"]["group_manifests"][0]
        )
        group = pq.read_table(groups_path).to_pylist()[0]
        group.update(
            {
                "qa_count": 5,
                "qa_pack_starts": [0, 2],
                "qa_pack_counts": [2, 3],
                "qa_pack_token_counts": [8, 12],
            }
        )
        pq.write_table(pa.Table.from_pylist([group]), groups_path)
        ready["qa_token_budget"] = 12
        ready["partitions"]["stage1/train"].update(
            {
                "logical_qas": 5,
                "physical_qa_rows": 5,
                "physical_qa_packs": 2,
                "group_manifest_sha256": {
                    str(groups_path): file_sha256(groups_path)
                },
            }
        )
        ready_path.write_text(json.dumps(ready))

        stream = RepoQACEStreamingDataset(
            ready_path,
            "stage1",
            "train",
            TinyTokenizer(),
            max_context_tokens=512,
            qa_token_budget=12,
            cache_mb=1,
            seed=17,
        )
        items = list(stream)
        assert len(items) == 1
        assert items[0]["n_queries"] == [5]
        assert items[0]["qa_pack_starts"] == [0, 2]
        assert items[0]["qa_pack_counts"] == [2, 3]
        assert items[0]["physical_pack_count"] == 2
        assert stream.physical_packs_consumed == 2


def test_snapshot_family_loss_weights_do_not_change_logical_counts():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ready_path = build_fixture(root)
        ready = json.loads(ready_path.read_text())
        ready["format"] = SNAPSHOT_MEMORY_READY_FORMAT
        ready["qa_loss_weights"] = {"ast": 1.0, "llm": 1.8}
        groups_path = Path(
            ready["partitions"]["stage1/train"]["group_manifests"][0]
        )
        group = pq.read_table(groups_path).to_pylist()[0]
        group.update(
            {
                "qa_count": 5,
                "qa_pack_starts": [0],
                "qa_pack_counts": [5],
                "qa_pack_token_counts": [20],
            }
        )
        pq.write_table(pa.Table.from_pylist([group]), groups_path)
        qa_path = Path(group["qa_file"])
        qas = pq.read_table(qa_path).to_pylist()
        qas[1]["qa_family"] = "llm"
        pq.write_table(pa.Table.from_pylist(qas), qa_path)
        ready["qa_token_budget"] = 20
        ready["partitions"]["stage1/train"].update(
            {
                "logical_qas": 5,
                "physical_qa_rows": 5,
                "group_manifest_sha256": {
                    str(groups_path): file_sha256(groups_path)
                },
            }
        )
        ready_path.write_text(json.dumps(ready))

        item = next(
            iter(
                RepoQACEStreamingDataset(
                    ready_path,
                    "stage1",
                    "train",
                    TinyTokenizer(),
                    max_context_tokens=512,
                    qa_token_budget=20,
                    cache_mb=1,
                    seed=17,
                )
            )
        )
        assert item["logical_qa_count"] == [5]
        assert item["qa_weights"].sum() == 5
        assert abs(float(item["qa_loss_weights"].sum()) - 5.8) < 1e-6
        assert abs(float(item["logical_qa_loss_weight"][0]) - 5.8) < 1e-6
