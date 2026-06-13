from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.c097_hard_shape_data_expansion import (
    C097ExpansionConfig,
    C097ExpansionError,
    build_c097_hard_shape_expansion,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c097_builds_balanced_manifest_with_cross_group_negatives(tmp_path: Path) -> None:
    # Given: four hard-shape groups with same-group positive crop pairs.
    source_manifest = tmp_path / "source.jsonl"
    source_root = tmp_path / "source_root"
    rows = _same_group_pairs(source_root, groups=("frog", "goblin", "oni", "lizard"), crops=4)
    _write_jsonl(source_manifest, rows)

    # When: the C097 builder expands them into explicit positive/negative rows.
    config = _config(tmp_path, source_manifest, source_root, max_rows_per_group=3)
    summary = build_c097_hard_shape_expansion(config)

    # Then: the manifest is balanced enough for the next SigLIP encoder stage.
    manifest_rows = _read_jsonl(config.output_manifest)
    assert summary.decision == "data_gate_pass_for_deeper_siglip_encoder_training"
    assert summary.selected_rows == 12
    assert summary.explicit_negative_rows == 12
    assert summary.heldout_rows_used == 0
    assert set(summary.selected_group_counts.values()) == {3}
    assert len(manifest_rows) == 12
    assert all(row["shape_group"] != row["negative_shape_group"] for row in manifest_rows)
    assert all(str(row["neg_id"]).startswith("c097_hard_shape/") for row in manifest_rows)
    assert (config.output_root / "c097_hard_shape/pair_000_ref.jpg").is_file()
    assert (config.output_root / "c097_hard_shape/pair_000_target.txt").is_file()
    assert config.review_sheet.is_file()
    assert config.report_path.is_file()


def test_c097_excludes_heldout_rows_and_rejects_malformed_ids(tmp_path: Path) -> None:
    # Given: one valid pair and one malformed heldout-looking row.
    source_manifest = tmp_path / "source.jsonl"
    source_root = tmp_path / "source_root"
    rows = _same_group_pairs(source_root, groups=("frog", "goblin"), crops=3)
    rows.append(
        {
            "ref_id": "external/c084_sheet_crop_pairs/heldout07_bad_id",
            "tgt_id": "external/c084_sheet_crop_pairs/heldout07_bad_id_2",
            "prompt": "prompt",
        }
    )
    _write_jsonl(source_manifest, rows)

    # When: heldout rows are present, they are counted as rejected rather than used.
    config = _config(tmp_path, source_manifest, source_root, max_rows_per_group=2, minimum_groups=2)
    summary = build_c097_hard_shape_expansion(config)

    # Then: no selected training row contains heldout ids.
    assert summary.heldout_rows_rejected == 1
    assert summary.heldout_rows_used == 0
    assert all("heldout" not in json.dumps(row) for row in _read_jsonl(config.output_manifest))

    # And malformed non-heldout ids block the data gate clearly.
    _write_jsonl(
        source_manifest,
        (
            {
                "ref_id": "external/c084_sheet_crop_pairs/bad",
                "tgt_id": "external/c084_sheet_crop_pairs/also_bad",
                "prompt": "prompt",
            },
        ),
    )
    with pytest.raises(C097ExpansionError, match="cannot parse hard-shape id"):
        build_c097_hard_shape_expansion(config)


def test_c097_rejects_missing_source_image(tmp_path: Path) -> None:
    # Given: metadata whose corresponding image is absent from the source root.
    source_manifest = tmp_path / "source.jsonl"
    source_root = tmp_path / "source_root"
    _write_jsonl(
        source_manifest,
        (
            _row("frog", "front", 1, "frog", "profile", 1),
            _row("goblin", "front", 1, "goblin", "profile", 1),
        ),
    )

    # When/Then: missing material is reported before a broken manifest is written.
    with pytest.raises(C097ExpansionError, match="missing source image"):
        build_c097_hard_shape_expansion(
            _config(tmp_path, source_manifest, source_root, minimum_groups=2)
        )


def _config(
    tmp_path: Path,
    source_manifest: Path,
    source_root: Path,
    *,
    max_rows_per_group: int = 4,
    minimum_groups: int = 4,
    minimum_rows: int = 4,
) -> C097ExpansionConfig:
    return C097ExpansionConfig(
        source_manifest=source_manifest,
        source_root=source_root,
        output_root=tmp_path / "c097_root",
        output_manifest=tmp_path / "out" / "c097.jsonl",
        output_summary=tmp_path / "out" / "c097.summary.json",
        review_sheet=tmp_path / "eval" / "pair_review_sheet.jpg",
        report_path=tmp_path / "eval" / "report.md",
        max_rows_per_group=max_rows_per_group,
        max_rows_per_source_pose_pair=4,
        minimum_groups=minimum_groups,
        minimum_rows=minimum_rows,
    )


def _same_group_pairs(source_root: Path, *, groups: tuple[str, ...], crops: int) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for group in groups:
        for index in range(1, crops + 1):
            _write_source_image(source_root, _image_id(group, "front", index))
            _write_source_image(source_root, _image_id(group, "profile", index))
            rows.append(_row(group, "front", index, group, "profile", index))
    return rows


def _row(
    ref_group: str,
    ref_pose: str,
    ref_index: int,
    tgt_group: str,
    tgt_pose: str,
    tgt_index: int,
) -> JsonObject:
    return {
        "ref_id": _image_id(ref_group, ref_pose, ref_index),
        "tgt_id": _image_id(tgt_group, tgt_pose, tgt_index),
        "prompt": "mrcolor_panel_style, hard shape prompt",
    }


def _image_id(group: str, pose: str, index: int) -> str:
    return f"external/c084_sheet_crop_pairs/c083_c082_{group}_{pose}_crop{index:02d}"


def _write_source_image(root: Path, image_id: str) -> None:
    path = root / f"{image_id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 40), "green").save(path)


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...] | list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    parsed: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            parsed.append(raw)
    return tuple(parsed)
