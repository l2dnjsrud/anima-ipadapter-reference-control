from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.c086_generated_hard_negative_manifest import (
    C086ManifestConfig,
    C086ManifestError,
    build_c086_generated_hard_negative_manifest,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c086_builds_generated_hard_negative_rows_without_heldout(
    tmp_path: Path,
) -> None:
    # Given: train, heldout, and crop-focus c085 gate outputs.
    config = _fixture_config(tmp_path)
    _write_training_sources(tmp_path)
    _write_c085_manifest(config.c085_manifest_path)
    _write_gate_summary(config.c085_gate_summary_path)
    _write_crop_summary(config.c085_crop_summary_path)
    _write_png(config.c085_gate_dir / "train00_c085_anchored_full_adapter_w14.png")
    _write_png(config.c085_gate_dir / "heldout00_c085_anchored_full_adapter_w14.png")
    _write_png(config.c085_gate_dir / "crop_pair00_c085_anchored_full_adapter_w14.png")

    # When: the c086 manifest is built.
    summary = build_c086_generated_hard_negative_manifest(config)

    # Then: only train/crop rows are emitted, with generated failures as neg_id.
    rows = _read_jsonl(config.output_manifest_path)
    assert summary.total_rows == 2
    assert summary.train_negative_rows == 1
    assert summary.crop_negative_rows == 1
    assert summary.heldout_rows_used == 0
    assert summary.generated_negative_rows == 2
    assert [row["ref_id"] for row in rows] == [
        "clean/ref",
        "external/c084_sheet_crop_pairs/crop_ref",
    ]
    assert [row["neg_id"] for row in rows] == [
        "external/c086_generated_hard_negatives/train00_c085_anchored_full_adapter_w14",
        "external/c086_generated_hard_negatives/crop_pair00_c085_anchored_full_adapter_w14",
    ]
    assert not any(str(row["ref_id"]).startswith("heldout") for row in rows)
    assert (
        config.output_image_root
        / "external/c086_generated_hard_negatives/train00_c085_anchored_full_adapter_w14.jpg"
    ).is_file()
    assert (config.output_image_root / "clean/ref.txt").is_symlink()
    assert (
        config.output_image_root / "external/c084_sheet_crop_pairs/crop_target.txt"
    ).is_symlink()


def test_c086_rejects_missing_generated_negative(tmp_path: Path) -> None:
    # Given: a train sample without its generated c085 negative PNG.
    config = _fixture_config(tmp_path)
    _write_training_sources(tmp_path)
    _write_c085_manifest(config.c085_manifest_path)
    _write_gate_summary(config.c085_gate_summary_path)

    # When/Then: the builder blocks instead of writing a partial manifest.
    with pytest.raises(C086ManifestError, match="missing generated negative"):
        build_c086_generated_hard_negative_manifest(config)


def _fixture_config(tmp_path: Path) -> C086ManifestConfig:
    return C086ManifestConfig(
        c085_manifest_path=tmp_path / "c085.jsonl",
        c085_image_root=tmp_path / "c085_root",
        c085_gate_dir=tmp_path / "gate",
        c085_gate_summary_path=tmp_path / "gate" / "summary.json",
        c085_crop_summary_path=tmp_path / "gate" / "crop_pair_summary.json",
        output_image_root=tmp_path / "out_root",
        output_manifest_path=tmp_path / "out" / "c086.jsonl",
        output_summary_path=tmp_path / "out" / "c086.summary.json",
    )


def _write_training_sources(tmp_path: Path) -> None:
    for image_id in (
        "clean/ref",
        "external/c084_sheet_crop_pairs/crop_ref",
        "external/c084_sheet_crop_pairs/crop_target",
    ):
        _write_jpg(tmp_path / "c085_root" / f"{image_id}.jpg")
        (tmp_path / "c085_root" / f"{image_id}.txt").parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        (tmp_path / "c085_root" / f"{image_id}.txt").write_text(
            "caption\n",
            encoding="utf-8",
        )


def _write_c085_manifest(path: Path) -> None:
    _write_jsonl(
        path,
        (
            {
                "ref_id": "external/c084_sheet_crop_pairs/crop_ref",
                "tgt_id": "external/c084_sheet_crop_pairs/crop_target",
                "prompt": "crop prompt",
            },
        ),
    )


def _write_gate_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: JsonObject = {
        "samples": [
            {
                "label": "train00",
                "split": "train",
                "ref_id": "clean/ref",
                "prompt": "train prompt",
            },
            {
                "label": "heldout00",
                "split": "heldout",
                "ref_id": "heldout/ref",
                "prompt": "heldout prompt",
            },
        ]
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _write_crop_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: JsonObject = {
        "samples": [
            {
                "label": "crop_pair00",
                "split": "direct_green",
                "ref_id": "external/c084_sheet_crop_pairs/crop_ref",
                "prompt": "crop prompt",
            },
        ]
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
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


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(80, 120, 160)).save(path)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(180, 40, 80)).save(path)
