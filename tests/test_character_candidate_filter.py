from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from tools.filter_character_candidate_pairs import filter_candidate_pairs


class FakeScorer:
    def __init__(self, scores: Mapping[str, tuple[float, ...]]) -> None:
        self._scores = scores

    def score(self, image_path: Path, candidate_texts: tuple[str, ...]) -> tuple[float, ...]:
        return self._scores[image_path.name]


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


def _write_pair_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "pair_id": "cand0000",
                "anchor_id": "root/a",
                "candidate_id": "root/b",
                "sg_page": "SG-001-01",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_filter_candidate_pairs_keeps_pairs_when_both_images_are_character_centered(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "pairs.jsonl"
    _write_pair_manifest(manifest)
    _write_image(tmp_path / "root/a.jpg")
    _write_image(tmp_path / "root/b.jpg")

    result = filter_candidate_pairs(
        manifest,
        data_root=tmp_path,
        scorer=FakeScorer(
            {
                "a.jpg": (0.8, 0.7, 0.6, 0.5, 0.2, 0.1, 0.0, -0.1),
                "b.jpg": (0.6, 0.5, 0.4, 0.3, 0.3, 0.2, 0.1, 0.0),
            }
        ),
        threshold=0.15,
    )

    assert result.summary.kept_pairs == 1
    assert result.rows[0].decision == "keep_character_pair_candidate"


def test_filter_candidate_pairs_keeps_pairs_at_threshold_boundary(tmp_path: Path) -> None:
    manifest = tmp_path / "pairs.jsonl"
    _write_pair_manifest(manifest)
    _write_image(tmp_path / "root/a.jpg")
    _write_image(tmp_path / "root/b.jpg")

    result = filter_candidate_pairs(
        manifest,
        data_root=tmp_path,
        scorer=FakeScorer(
            {
                "a.jpg": (0.75, 0.1, 0.0, -0.1, 0.6, 0.2, 0.1, 0.0),
                "b.jpg": (0.65, 0.1, 0.0, -0.1, 0.5, 0.2, 0.1, 0.0),
            }
        ),
        threshold=0.15,
    )

    assert result.summary.kept_pairs == 1
    assert result.rows[0].decision == "keep_character_pair_candidate"


def test_filter_candidate_pairs_rejects_pairs_with_non_character_side(tmp_path: Path) -> None:
    manifest = tmp_path / "pairs.jsonl"
    _write_pair_manifest(manifest)
    _write_image(tmp_path / "root/a.jpg")
    _write_image(tmp_path / "root/b.jpg")

    result = filter_candidate_pairs(
        manifest,
        data_root=tmp_path,
        scorer=FakeScorer(
            {
                "a.jpg": (0.8, 0.7, 0.6, 0.5, 0.2, 0.1, 0.0, -0.1),
                "b.jpg": (0.2, 0.1, 0.0, -0.1, 0.7, 0.6, 0.5, 0.4),
            }
        ),
        threshold=0.15,
    )

    assert result.summary.kept_pairs == 0
    assert result.rows[0].decision == "reject_non_character_pair"


def test_filter_candidate_pairs_rejects_pairs_below_threshold(tmp_path: Path) -> None:
    manifest = tmp_path / "pairs.jsonl"
    _write_pair_manifest(manifest)
    _write_image(tmp_path / "root/a.jpg")
    _write_image(tmp_path / "root/b.jpg")

    result = filter_candidate_pairs(
        manifest,
        data_root=tmp_path,
        scorer=FakeScorer(
            {
                "a.jpg": (0.8, 0.7, 0.6, 0.5, 0.2, 0.1, 0.0, -0.1),
                "b.jpg": (0.64, 0.1, 0.0, -0.1, 0.5, 0.2, 0.1, 0.0),
            }
        ),
        threshold=0.15,
    )

    assert result.summary.kept_pairs == 0
    assert result.rows[0].decision == "reject_non_character_pair"
