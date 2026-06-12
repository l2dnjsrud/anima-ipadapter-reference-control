from __future__ import annotations

import json
from pathlib import Path

import torch

from tools.probe_failure_attribute_embeddings import score_failure_attribute_probe


class FakeEmbedder:
    def __init__(self, embeddings: dict[str, torch.Tensor]) -> None:
        self._embeddings = embeddings

    def encode_image(self, image_path: Path) -> torch.Tensor:
        return self._embeddings[image_path.name]


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


def test_score_failure_attribute_probe_ranks_candidates_by_reference_cosine(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "ref.jpg"
    no_ip = tmp_path / "no_ip.png"
    blend = tmp_path / "blend.png"
    c063 = tmp_path / "c063.png"
    for path in (reference, no_ip, blend, c063):
        _touch(path)
    manifest_path = tmp_path / "probe.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "sample": "heldout-test",
                "split": "heldout",
                "failure_attribute": "beard-hat-crop",
                "reference_path": str(reference),
                "candidates": {
                    "no_ip": str(no_ip),
                    "blend_species_face": str(blend),
                    "c063_calibrator_only_w14": str(c063),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = score_failure_attribute_probe(
        manifest_path,
        embedder=FakeEmbedder(
            {
                "ref.jpg": torch.tensor([1.0, 0.0]),
                "no_ip.png": torch.tensor([0.0, 1.0]),
                "blend.png": torch.tensor([1.0, 0.0]),
                "c063.png": torch.tensor([0.5, 0.5]),
            }
        ),
        encoder_name="fake",
    )

    summary = result["summary"]
    assert isinstance(summary, dict)
    assert summary["cases"] == 1
    assert summary["supported_cases"] == 1
    assert summary["decision"] == "encoder_space_has_partial_supervised_signal"
    rows = result["rows"]
    assert isinstance(rows, list)
    best = rows[0]
    assert isinstance(best, dict)
    assert best["variant"] == "blend_species_face"
    assert best["rank"] == 1
    assert best["uplift"] == 1.0
