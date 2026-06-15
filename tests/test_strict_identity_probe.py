from __future__ import annotations

import json
from pathlib import Path

import torch

from tools.build_strict_panel_pair_probe_manifest import (
    build_strict_probe_rows,
)
from tools.score_siglip_token_pair_probe import score_token_pair_probe_manifest


class FakeTokenEmbedder:
    def __init__(self, embeddings: dict[str, torch.Tensor]) -> None:
        self._embeddings = embeddings

    def encode_image(self, image_path: Path) -> torch.Tensor:
        return self._embeddings[image_path.name]


def _write_image_pair(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")
    path.with_suffix(".txt").write_text("caption", encoding="utf-8")


def _name(prefix: str, panel: str) -> str:
    return f"{prefix}_candidate_{panel}.jpg"


def test_build_strict_probe_rows_uses_duplicate_panel_positive_and_same_group_negative(
    tmp_path: Path,
) -> None:
    panel_a = "00001_SG-001-01_page_100x200_s01"
    panel_b = "00002_SG-001-02_page_100x200_s01"
    _write_image_pair(tmp_path / "001-100/SG-001" / _name("v4", panel_a))
    _write_image_pair(tmp_path / "001-100/SG-001" / _name("v5", panel_a))
    _write_image_pair(tmp_path / "001-100/SG-001" / _name("v5", panel_b))

    rows = build_strict_probe_rows(tmp_path, pairs_per_label=1)

    assert [row.label for row in rows] == ["positive", "negative"]
    assert rows[0].anchor_panel_key == rows[0].candidate_panel_key
    assert rows[1].anchor_group == rows[1].candidate_group
    assert rows[1].anchor_panel_key != rows[1].candidate_panel_key


def test_score_token_pair_probe_reports_token_metric_separation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "pairs.jsonl"
    rows = [
        {
            "pair_id": "p0",
            "label": "positive",
            "anchor_id": "root/a",
            "candidate_id": "root/b",
        },
        {
            "pair_id": "n0",
            "label": "negative",
            "anchor_id": "root/a",
            "candidate_id": "root/c",
        },
    ]
    manifest_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    for image_id in ("root/a.jpg", "root/b.jpg", "root/c.jpg"):
        _write_image_pair(tmp_path / image_id)

    result = score_token_pair_probe_manifest(
        manifest_path,
        data_root=tmp_path,
        embedder=FakeTokenEmbedder(
            {
                "a.jpg": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "b.jpg": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "c.jpg": torch.tensor([[-1.0, 0.0], [0.0, -1.0]]),
            }
        ),
        encoder_name="fake",
        layer=-1,
        topk=1,
    )

    assert result["summaries"]["topk_token"]["positive_mean"] == 1.0
    assert result["summaries"]["topk_token"]["negative_mean"] < 1.0
    assert result["summaries"]["topk_token"]["pairwise_auc"] == 1.0
