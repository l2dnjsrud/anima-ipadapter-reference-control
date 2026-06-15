from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.single_character_manifest import (
    SingleCharacterSelectionConfig,
    select_single_character_rows,
    write_candidate_sheet,
    write_pair_rows,
)
from tools.validate_reference_suite import validate_reference_suite


def _write_image_pair(
    dataset_root: Path,
    relative_stem: str,
    *,
    size: tuple[int, int],
    caption: str = "mrcolor_panel_style, character panel, close-up panel",
) -> None:
    image_path = dataset_root / f"{relative_stem}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 80, 40)).save(image_path)
    image_path.with_suffix(".txt").write_text(caption, encoding="utf-8")


def test_select_single_character_rows_filters_and_splits_deterministically(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    for index in range(6):
        _write_image_pair(dataset_root, f"SG-{index:03d}/portrait", size=(720, 960))
    _write_image_pair(
        dataset_root,
        "SG-999/wide",
        size=(1600, 500),
        caption="mrcolor_panel_style, character panel, wide panel",
    )
    _write_image_pair(
        dataset_root,
        "SG-998/background",
        size=(720, 960),
        caption="mrcolor_panel_style, background panel",
    )

    result = select_single_character_rows(
        SingleCharacterSelectionConfig(
            dataset_root=dataset_root,
            train_count=4,
            heldout_count=2,
        )
    )

    assert [row.ref_id for row in result.train_rows] == [
        "SG-001/portrait",
        "SG-002/portrait",
        "SG-004/portrait",
        "SG-005/portrait",
    ]
    assert [row.ref_id for row in result.heldout_rows] == [
        "SG-000/portrait",
        "SG-003/portrait",
    ]
    assert result.summary.candidates_scanned == 8
    assert result.summary.candidates_kept == 6


def test_write_rows_and_candidate_sheet(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    for index in range(3):
        _write_image_pair(dataset_root, f"SG-{index:03d}/portrait", size=(720, 960))
    result = select_single_character_rows(
        SingleCharacterSelectionConfig(
            dataset_root=dataset_root,
            train_count=2,
            heldout_count=1,
        )
    )
    manifest_path = tmp_path / "train.jsonl"
    sheet_path = tmp_path / "sheet.jpg"

    write_pair_rows(result.train_rows, manifest_path)
    write_candidate_sheet(result.candidates, sheet_path)

    loaded = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert loaded == [
        {
            "ref_id": "SG-001/portrait",
            "tgt_id": "SG-001/portrait",
            "prompt": "mrcolor_panel_style, character panel, close-up panel",
        },
        {
            "ref_id": "SG-002/portrait",
            "tgt_id": "SG-002/portrait",
            "prompt": "mrcolor_panel_style, character panel, close-up panel",
        },
    ]
    assert sheet_path.exists()


def test_validate_reference_suite_reports_row_and_missing_counts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_image_pair(dataset_root, "SG-001/portrait", size=(720, 960))
    manifest_path = tmp_path / "suite.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "ref_id": "SG-001/portrait",
                "tgt_id": "SG-001/portrait",
                "prompt": "mrcolor_panel_style, character panel",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_reference_suite(manifest_path, dataset_root)

    assert result.rows == 1
    assert result.missing_images == ()
