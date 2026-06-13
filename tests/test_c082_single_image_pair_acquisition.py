from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from tools.c082_single_image_pair_acquisition import (
    C082Config,
    build_c082_prompt_package,
    review_c082_generation,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue

SINGLE_IMAGE_TERMS = (
    "exactly one character",
    "one pose",
    "one single illustration",
    "no extra figure",
    "simple background",
)
FORBIDS = (
    "character sheet",
    "reference sheet",
    "turnaround",
    "model sheet",
    "lineup",
    "multiple poses",
    "multiple views",
    "split view",
    "collage",
    "duplicate character",
    "extra character",
)


def test_c082_prompt_package_requires_single_image_constraints(tmp_path: Path) -> None:
    # Given: a fresh c082 output area.
    config = C082Config(out_dir=tmp_path / "out", scratch_dir=tmp_path / "scratch")

    # When: the prompt package is built.
    summary = build_c082_prompt_package(config)

    # Then: it contains six c082 groups with strict single-image wording.
    rows = _read_jsonl(tmp_path / "out" / "prompt_manifest.jsonl")
    groups = {str(row["group_id"]) for row in rows}
    assert summary["prompt_count"] == 24
    assert len(groups) == 6
    assert all(group.startswith("c082_") for group in groups)
    assert all(_contains_all(row, SINGLE_IMAGE_TERMS) for row in rows)
    assert all(_contains_all(row, FORBIDS) for row in rows)
    assert summary["heldout_rows_used"] == 0
    assert summary["training_started"] is False
    assert summary["raw_generated_images_committed"] is False


def test_c082_review_builds_cross_view_pairs_without_self_pairs(tmp_path: Path) -> None:
    # Given: four full target-positive groups with generated image rows.
    out_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    build_c082_prompt_package(C082Config(out_dir=out_dir, scratch_dir=scratch_dir))
    rows = _read_jsonl(out_dir / "prompt_manifest.jsonl")[:16]
    generation_path = _generation_manifest(out_dir, scratch_dir, rows)
    labels_path = _labels(out_dir, rows)

    # When: the generation manifest is reviewed.
    summary = review_c082_generation(
        C082Config(out_dir=out_dir, scratch_dir=scratch_dir, labels_path=labels_path),
        generation_manifest_path=generation_path,
    )

    # Then: only cross-view target-positive pairs are approved and the decision advances.
    pairs = _read_jsonl(out_dir / "approved_pair_manifest.jsonl")
    assert summary["approved_group_count"] == 4
    assert summary["approved_pair_rows"] == 48
    assert summary["direct_self_pair_rows"] == 0
    assert summary["decision"] == "ready_for_c083_paired_training_manifest"
    assert all(row["ref_id"] != row["tgt_id"] for row in pairs)
    assert (scratch_dir / "contact_sheet.jpg").is_file()


def test_c082_review_blocks_single_view_groups(tmp_path: Path) -> None:
    # Given: every group has one target-positive generated image only.
    out_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    build_c082_prompt_package(C082Config(out_dir=out_dir, scratch_dir=scratch_dir))
    rows = _read_jsonl(out_dir / "prompt_manifest.jsonl")[::4]
    generation_path = _generation_manifest(out_dir, scratch_dir, rows)
    labels_path = _labels(out_dir, rows)

    # When: the single-view generations are reviewed.
    summary = review_c082_generation(
        C082Config(out_dir=out_dir, scratch_dir=scratch_dir, labels_path=labels_path),
        generation_manifest_path=generation_path,
    )

    # Then: no pairs are formed despite six approved groups.
    assert summary["approved_group_count"] == 0
    assert summary["approved_pair_rows"] == 0
    assert summary["decision"] == "more_identity_pairs_required"


def _contains_all(row: JsonObject, terms: tuple[str, ...]) -> bool:
    text = f"{row['prompt']} {row['negative']}".lower()
    return all(term in text for term in terms)


def _generation_manifest(out_dir: Path, scratch_dir: Path, rows: tuple[JsonObject, ...]) -> Path:
    path = out_dir / "generation_manifest.jsonl"
    image_dir = scratch_dir / "generated"
    image_dir.mkdir(parents=True, exist_ok=True)
    generated: list[JsonObject] = []
    for row in rows:
        image_path = image_dir / f"{row['candidate_id']}.png"
        Image.new("RGB", (96, 128), (60, 190, 80)).save(image_path)
        generated.append(dict(row) | {"status": "generated", "local_image_path": str(image_path), "image_width": 96, "image_height": 128, "blank": False})
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated), encoding="utf-8")
    return path


def _labels(out_dir: Path, rows: tuple[JsonObject, ...]) -> Path:
    path = out_dir / "manual_visual_labels.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("candidate_id", "manual_label", "manual_note"), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"candidate_id": row["candidate_id"], "manual_label": "target_positive", "manual_note": "fixture approved single image identity view"})
    return path


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    parsed: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            parsed.append(raw)
    return tuple(parsed)
