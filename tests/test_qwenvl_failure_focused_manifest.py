from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_qwenvl_failure_focused_manifest import (
    FailureManifestConfig,
    ManifestInputError,
    build_failure_focused_manifest,
)


def test_build_failure_focused_manifest_repeats_train_failures_without_heldout(
    tmp_path: Path,
) -> None:
    clean_manifest = tmp_path / "clean.jsonl"
    positive_manifest = tmp_path / "positive.jsonl"
    gate_summary = tmp_path / "summary.json"
    output_manifest = tmp_path / "out.jsonl"
    output_summary = tmp_path / "out.summary.json"

    _write_jsonl(
        clean_manifest,
        (
            {"ref_id": "train/pose", "tgt_id": "train/pose", "prompt": "base pose"},
            {"ref_id": "train/plain", "tgt_id": "train/plain", "prompt": "base plain"},
        ),
    )
    _write_jsonl(
        positive_manifest,
        (
            {"ref_id": "positive/a", "tgt_id": "positive/b", "prompt": "positive pair"},
        ),
    )
    gate_summary.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "split": "train",
                        "ref_id": "train/pose",
                        "selected_attributes": [
                            "side profile portrait",
                            "folding fan in hand",
                        ],
                    },
                    {
                        "split": "train",
                        "ref_id": "train/plain",
                        "selected_attributes": ["calm scholar"],
                    },
                    {
                        "split": "heldout",
                        "ref_id": "heldout/demon",
                        "selected_attributes": ["green-skinned demon"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = build_failure_focused_manifest(
        FailureManifestConfig(
            clean_manifest_path=clean_manifest,
            positive_manifest_path=positive_manifest,
            gate_summary_path=gate_summary,
            output_manifest_path=output_manifest,
            output_summary_path=output_summary,
            repeat_per_failure_row=2,
        )
    )

    rows = _read_jsonl(output_manifest)
    assert [row["ref_id"] for row in rows] == [
        "train/pose",
        "train/plain",
        "train/pose",
        "train/pose",
        "positive/a",
    ]
    assert summary.clean32_rows == 2
    assert summary.failure_source_rows == 1
    assert summary.failure_repeated_rows == 2
    assert summary.c052_positive_rows == 1
    assert summary.total_rows == 5
    assert summary.heldout_rows_used == 0
    assert summary.failure_keyword_counts["side profile portrait"] == 1
    assert summary.failure_keyword_counts["folding fan in hand"] == 1
    assert "heldout/demon" not in {row["ref_id"] for row in rows}

    saved_summary = json.loads(output_summary.read_text(encoding="utf-8"))
    assert saved_summary["total_rows"] == 5
    assert saved_summary["heldout_rows_used"] == 0


def test_build_failure_focused_manifest_rejects_malformed_clean_row(
    tmp_path: Path,
) -> None:
    clean_manifest = tmp_path / "clean.jsonl"
    positive_manifest = tmp_path / "positive.jsonl"
    gate_summary = tmp_path / "summary.json"

    _write_jsonl(clean_manifest, ({"ref_id": "train/bad", "prompt": "missing target"},))
    _write_jsonl(positive_manifest, ())
    gate_summary.write_text(json.dumps({"samples": []}), encoding="utf-8")

    with pytest.raises(ManifestInputError, match="missing tgt_id"):
        build_failure_focused_manifest(
            FailureManifestConfig(
                clean_manifest_path=clean_manifest,
                positive_manifest_path=positive_manifest,
                gate_summary_path=gate_summary,
                output_manifest_path=tmp_path / "out.jsonl",
                output_summary_path=tmp_path / "out.summary.json",
            )
        )


def _write_jsonl(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]
