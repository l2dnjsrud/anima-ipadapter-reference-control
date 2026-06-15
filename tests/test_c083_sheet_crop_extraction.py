from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from tools.c083_sheet_crop_extraction import C083Config, extract_c083_crops, review_c083_crops
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c083_extracts_component_crops_and_preserves_source_metadata(tmp_path: Path) -> None:
    # Given: a generated sheet image with two separated green foreground figures.
    out_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    source_image = scratch_dir / "source.png"
    source_image.parent.mkdir(parents=True, exist_ok=True)
    _two_figure_image(source_image)
    source_manifest = out_dir / "generation_manifest.jsonl"
    _write_jsonl(
        source_manifest,
        (
            {
                "candidate_id": "c082_group_a_front",
                "group_id": "c082_group_a",
                "view_id": "front",
                "status": "generated",
                "local_image_path": str(source_image),
                "blank": False,
            },
        ),
    )

    # When: c083 extracts crop candidates.
    summary = extract_c083_crops(C083Config(out_dir=out_dir, scratch_dir=scratch_dir, source_manifest_path=source_manifest))

    # Then: it writes reviewable crop rows while preserving the c082 source identity metadata.
    rows = _read_jsonl(out_dir / "crop_candidate_manifest.jsonl")
    assert summary["source_generated_rows"] == 1
    assert summary["crop_candidate_rows"] == 2
    assert summary["heldout_rows_used"] == 0
    assert summary["training_started"] is False
    assert summary["raw_crop_images_committed"] is False
    assert {row["group_id"] for row in rows} == {"c082_group_a"}
    assert {row["source_candidate_id"] for row in rows} == {"c082_group_a_front"}
    assert all(Path(str(row["local_image_path"])).is_file() for row in rows)
    assert (out_dir / "contact_sheet.jpg").is_file()


def test_c083_review_builds_cross_source_pairs_without_self_pairs(tmp_path: Path) -> None:
    # Given: four groups with four visually-approved crop candidates from distinct sources.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rows = tuple(_crop_row(group_index, source_index) for group_index in range(4) for source_index in range(4))
    _write_jsonl(out_dir / "crop_candidate_manifest.jsonl", rows)
    labels_path = _labels(out_dir, rows, "target_positive")

    # When: the reviewed crop rows are converted into training pairs.
    summary = review_c083_crops(C083Config(out_dir=out_dir, scratch_dir=tmp_path / "scratch", labels_path=labels_path))

    # Then: only cross-source pairs are approved and the acquisition gate advances.
    pairs = _read_jsonl(out_dir / "approved_pair_manifest.jsonl")
    assert summary["approved_group_count"] == 4
    assert summary["approved_pair_rows"] == 48
    assert summary["direct_self_pair_rows"] == 0
    assert summary["decision"] == "ready_for_c084_paired_training_manifest"
    assert all(row["ref_id"] != row["tgt_id"] for row in pairs)


def test_c083_review_blocks_single_source_groups(tmp_path: Path) -> None:
    # Given: target-positive crops that all came from the same source sheet.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rows = tuple(_crop_row(0, 0, crop_index=index) for index in range(3))
    _write_jsonl(out_dir / "crop_candidate_manifest.jsonl", rows)
    labels_path = _labels(out_dir, rows, "target_positive")

    # When: c083 reviews the crop manifest.
    summary = review_c083_crops(C083Config(out_dir=out_dir, scratch_dir=tmp_path / "scratch", labels_path=labels_path))

    # Then: the same-source crops do not form training pairs.
    assert summary["approved_group_count"] == 0
    assert summary["approved_pair_rows"] == 0
    assert summary["decision"] == "more_pairs_required"


def _two_figure_image(path: Path) -> None:
    image = Image.new("RGB", (240, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((28, 32, 82, 130), fill=(52, 170, 82), outline=(20, 70, 40), width=4)
    draw.rectangle((152, 26, 212, 136), fill=(70, 190, 96), outline=(20, 70, 40), width=4)
    image.save(path)


def _crop_row(group_index: int, source_index: int, *, crop_index: int = 1) -> JsonObject:
    candidate_id = f"c083_group{group_index}_source{source_index}_crop{crop_index}"
    return {
        "candidate_id": candidate_id,
        "group_id": f"c082_group{group_index}",
        "source_candidate_id": f"c082_group{group_index}_view{source_index}",
        "source_view_id": f"view{source_index}",
        "crop_index": crop_index,
        "local_image_path": f".tmp/test/{candidate_id}.png",
        "heldout_excluded": True,
    }


def _labels(out_dir: Path, rows: tuple[JsonObject, ...], label: str) -> Path:
    path = out_dir / "manual_visual_labels.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("candidate_id", "manual_label", "manual_note"), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"candidate_id": row["candidate_id"], "manual_label": label, "manual_note": "fixture visual decision"})
    return path


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    parsed: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            parsed.append(raw)
    return tuple(parsed)
