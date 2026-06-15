from __future__ import annotations

import json
from pathlib import Path

import torch

from tools.rank_identity_candidate_pairs import rank_candidate_pairs, write_rank_outputs


class FakeEmbedder:
    def __init__(self, vectors: dict[str, torch.Tensor]) -> None:
        self._vectors = vectors

    def encode_image(self, image_path: Path) -> torch.Tensor:
        return self._vectors[image_path.name]


def _write_manifest(path: Path) -> None:
    rows = [
        {
            "pair_id": "cand0000",
            "anchor_id": "root/a",
            "candidate_id": "root/b",
            "sg_page": "SG-001-01",
        },
        {
            "pair_id": "cand0001",
            "anchor_id": "root/a",
            "candidate_id": "root/c",
            "sg_page": "SG-001-01",
        },
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_rank_candidate_pairs_orders_pairs_by_similarity(tmp_path: Path) -> None:
    manifest_path = tmp_path / "candidates.jsonl"
    _write_manifest(manifest_path)

    result = rank_candidate_pairs(
        manifest_path,
        data_root=tmp_path,
        embedder=FakeEmbedder(
            {
                "a.jpg": torch.tensor([1.0, 0.0]),
                "b.jpg": torch.tensor([0.0, 1.0]),
                "c.jpg": torch.tensor([1.0, 0.0]),
            }
        ),
        top_k=1,
    )

    assert [row.pair_id for row in result.rows] == ["cand0001", "cand0000"]
    assert result.summary.input_pairs == 2
    assert result.summary.top_pairs == 1


def test_write_rank_outputs_writes_scored_and_top_rows(tmp_path: Path) -> None:
    manifest_path = tmp_path / "candidates.jsonl"
    _write_manifest(manifest_path)
    result = rank_candidate_pairs(
        manifest_path,
        data_root=tmp_path,
        embedder=FakeEmbedder(
            {
                "a.jpg": torch.tensor([1.0, 0.0]),
                "b.jpg": torch.tensor([0.0, 1.0]),
                "c.jpg": torch.tensor([1.0, 0.0]),
            }
        ),
        top_k=1,
    )

    scored_path = tmp_path / "scored.jsonl"
    top_path = tmp_path / "top.jsonl"
    write_rank_outputs(result, scored_output_path=scored_path, top_output_path=top_path)

    assert sum(1 for _ in scored_path.open()) == 2
    top_rows = [json.loads(line) for line in top_path.open()]
    assert top_rows[0]["pair_id"] == "cand0001"
