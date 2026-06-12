from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_reviewed_identity_manifest import (
    ReviewInputError,
    build_reviewed_rows,
    summarize_reviewed_rows,
)


def _write_candidate(path: Path, pair_id: str = "cand0000") -> None:
    path.write_text(
        json.dumps(
            {
                "pair_id": pair_id,
                "anchor_id": "root/a",
                "candidate_id": "root/b",
                "sg_page": "SG-001-01",
                "anchor_character_score": 0.2,
                "candidate_character_score": 0.16,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_label(path: Path, *, pair_id: str = "cand0000", label: str = "same_character") -> None:
    path.write_text(
        json.dumps(
            {
                "pair_id": pair_id,
                "review_label": label,
                "positive_usable": True,
                "review_notes": "same clear face pair",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_reviewed_rows_merges_manual_labels(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.jsonl"
    label_path = tmp_path / "labels.jsonl"
    _write_candidate(candidate_path)
    _write_label(label_path)

    rows = build_reviewed_rows(candidate_path, label_path)
    summary = summarize_reviewed_rows(rows)

    assert rows[0].review_label == "same_character"
    assert rows[0].positive_usable is True
    assert summary.same_character == 1
    assert summary.positive_usable == 1


def test_build_reviewed_rows_rejects_missing_label(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.jsonl"
    label_path = tmp_path / "labels.jsonl"
    _write_candidate(candidate_path)
    _write_label(label_path, pair_id="cand9999")

    with pytest.raises(ReviewInputError, match="missing label"):
        build_reviewed_rows(candidate_path, label_path)


def test_build_reviewed_rows_rejects_invalid_label(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.jsonl"
    label_path = tmp_path / "labels.jsonl"
    _write_candidate(candidate_path)
    _write_label(label_path, label="maybe")

    with pytest.raises(ReviewInputError, match="invalid review_label"):
        build_reviewed_rows(candidate_path, label_path)
