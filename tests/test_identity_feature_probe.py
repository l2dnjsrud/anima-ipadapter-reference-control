from __future__ import annotations

import json
from pathlib import Path

import torch

from tools.build_identity_pair_probe_manifest import build_probe_rows
from tools.score_identity_pair_probe import score_pair_probe_manifest


class FakeEmbedder:
    def __init__(self, embeddings: dict[str, torch.Tensor]) -> None:
        self._embeddings = embeddings

    def encode_image(self, image_path: Path) -> torch.Tensor:
        return self._embeddings[image_path.name]


def _write_image_pair(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")
    path.with_suffix(".txt").write_text("caption", encoding="utf-8")


def test_build_probe_rows_emits_balanced_same_group_and_cross_group_pairs(tmp_path: Path) -> None:
    for image_id in (
        "A/SG-001/one.jpg",
        "A/SG-001/two.jpg",
        "A/SG-002/one.jpg",
        "A/SG-002/two.jpg",
    ):
        _write_image_pair(tmp_path / image_id)

    rows = build_probe_rows(tmp_path, pairs_per_label=2)

    labels = [row.label for row in rows]
    assert labels == ["positive", "negative", "positive", "negative"]
    assert rows[0].anchor_group == rows[0].candidate_group
    assert rows[1].anchor_group != rows[1].candidate_group


def test_score_pair_probe_manifest_reports_margin_and_auc(tmp_path: Path) -> None:
    manifest_path = tmp_path / "pairs.jsonl"
    rows = [
        {
            "pair_id": "p0",
            "label": "positive",
            "anchor_id": "root/a",
            "candidate_id": "root/b",
            "anchor_group": "root",
            "candidate_group": "root",
        },
        {
            "pair_id": "n0",
            "label": "negative",
            "anchor_id": "root/a",
            "candidate_id": "other/c",
            "anchor_group": "root",
            "candidate_group": "other",
        },
    ]
    manifest_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    for image_id in ("root/a.jpg", "root/b.jpg", "other/c.jpg"):
        _write_image_pair(tmp_path / image_id)

    result = score_pair_probe_manifest(
        manifest_path,
        data_root=tmp_path,
        embedder=FakeEmbedder(
            {
                "a.jpg": torch.tensor([1.0, 0.0]),
                "b.jpg": torch.tensor([1.0, 0.0]),
                "c.jpg": torch.tensor([0.0, 1.0]),
            }
        ),
        encoder_name="fake",
    )

    assert result["summary"]["positive_mean"] == 1.0
    assert result["summary"]["negative_mean"] == 0.0
    assert result["summary"]["separation_margin"] == 1.0
    assert result["summary"]["pairwise_auc"] == 1.0
    assert result["group_summaries"]["root"]["positive_mean"] == 1.0
    assert result["group_summaries"]["root"]["negative_mean"] == 0.0
    assert result["group_summaries"]["root"]["pairwise_auc"] == 1.0
