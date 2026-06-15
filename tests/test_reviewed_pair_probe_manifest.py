from __future__ import annotations

import json
from pathlib import Path

from tools.build_reviewed_pair_probe_manifest import (
    build_pair_probe_rows,
    summarize_pair_probe_rows,
)


def _row(pair_id: str, review_label: str, positive_usable: bool) -> str:
    return json.dumps(
        {
            "pair_id": pair_id,
            "anchor_id": f"root/{pair_id}/a",
            "candidate_id": f"root/{pair_id}/b",
            "sg_page": "SG-001-01",
            "review_label": review_label,
            "positive_usable": positive_usable,
        }
    )


def test_build_pair_probe_rows_uses_only_safe_positive_and_different_negative(
    tmp_path: Path,
) -> None:
    reviewed_path = tmp_path / "reviewed.jsonl"
    reviewed_path.write_text(
        "\n".join(
            (
                _row("same_usable", "same_character", True),
                _row("same_noisy", "same_character", False),
                _row("negative", "different_character", False),
                _row("unclear", "unclear", False),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_pair_probe_rows(reviewed_path)
    summary = summarize_pair_probe_rows(input_rows=4, rows=rows)

    assert [row.pair_id for row in rows] == ["same_usable", "negative"]
    assert [row.label for row in rows] == ["positive", "negative"]
    assert rows[0].anchor_group == "root/same_usable"
    assert summary.input_rows == 4
    assert summary.output_rows == 2
    assert summary.positive_rows == 1
    assert summary.negative_rows == 1
