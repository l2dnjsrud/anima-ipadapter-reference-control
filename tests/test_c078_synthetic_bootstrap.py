from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from tools.c078_synthetic_bootstrap import C078BootstrapConfig, build_c078_prompt_package, review_c078_generation
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c078_prompt_package_writes_24_direct_green_prompts(tmp_path: Path) -> None:
    summary = build_c078_prompt_package(C078BootstrapConfig(out_dir=tmp_path / "out", scratch_dir=tmp_path / "scratch"))

    assert summary["prompt_count"] == 24
    assert summary["heldout_rows_used"] == 0
    assert summary["training_started"] is False
    assert summary["decision"] == "prompt_package_ready"
    rows = _read_jsonl(tmp_path / "out" / "prompt_manifest.jsonl")
    assert len(rows) == 24
    assert len({row["seed"] for row in rows}) == 24
    assert all("green" in str(row["prompt"]).lower() for row in rows)
    assert all("single character" in str(row["prompt"]).lower() for row in rows)


def test_c078_review_promotes_generation_only_after_threshold(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    build_c078_prompt_package(C078BootstrapConfig(out_dir=out_dir, scratch_dir=scratch_dir))
    generation_path = _generation_manifest(out_dir, scratch_dir, count=24)
    labels_path = _labels(out_dir, count=12)

    summary = review_c078_generation(
        C078BootstrapConfig(out_dir=out_dir, scratch_dir=scratch_dir, labels_path=labels_path),
        generation_manifest_path=generation_path,
    )

    assert summary["generated_count"] == 24
    assert summary["blank_count"] == 0
    assert summary["new_target_positive_confirmed_count"] == 12
    assert summary["decision"] == "ready_for_c079_training_manifest"
    assert (scratch_dir / "contact_sheet.jpg").is_file()
    assert (out_dir / "reviewed_synthetic_labels.jsonl").is_file()


def test_c078_review_blocks_when_labels_are_missing(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    build_c078_prompt_package(C078BootstrapConfig(out_dir=out_dir, scratch_dir=scratch_dir))
    generation_path = _generation_manifest(out_dir, scratch_dir, count=3)

    summary = review_c078_generation(
        C078BootstrapConfig(out_dir=out_dir, scratch_dir=scratch_dir),
        generation_manifest_path=generation_path,
    )

    assert summary["generated_count"] == 3
    assert summary["new_target_positive_confirmed_count"] == 0
    assert summary["decision"] == "manual_needed_review_synthetic_refs"
    assert (out_dir / "manual_needed_report.md").is_file()


def _generation_manifest(out_dir: Path, scratch_dir: Path, *, count: int) -> Path:
    path = out_dir / "generation_manifest.jsonl"
    image_dir = scratch_dir / "generated"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[JsonObject] = []
    for index in range(count):
        image_path = image_dir / f"c078_{index:02d}.png"
        Image.new("RGB", (96, 128), (60, 190, 80)).save(image_path)
        rows.append(
            {
                "candidate_id": f"c078_synth_{index:02d}",
                "prompt_id": f"prompt-{index}",
                "status": "generated",
                "local_image_path": str(image_path),
                "image_width": 96,
                "image_height": 128,
            }
        )
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def _labels(out_dir: Path, *, count: int) -> Path:
    path = out_dir / "manual_visual_labels.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("candidate_id", "manual_label", "manual_note"), lineterminator="\n")
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "candidate_id": f"c078_synth_{index:02d}",
                    "manual_label": "target_positive",
                    "manual_note": "fixture direct-green synthetic reference",
                }
            )
    return path


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)
