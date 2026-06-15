from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c069_direct_green_acquisition import (
    C069Config,
    build_c069_direct_green_acquisition,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_build_c069_acquisition_excludes_heldout_and_requires_new_data(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    _write_image(dataset_root / "train/leaf.jpg", (30, 170, 60))
    _write_image(dataset_root / "train/cup.jpg", (70, 150, 90))
    _write_image(dataset_root / "train/red_orb.jpg", (180, 30, 40))
    _write_image(dataset_root / "heldout/monster.jpg", (20, 190, 40))
    all_manifest = tmp_path / "all.jsonl"
    heldout_manifest = tmp_path / "heldout.jsonl"
    c067_topk_path = tmp_path / "topk.json"
    _write_jsonl(
        all_manifest,
        (
            {"ref_id": "train/leaf"},
            {"ref_id": "train/cup"},
            {"ref_id": "train/red_orb"},
            {"ref_id": "heldout/monster"},
        ),
    )
    _write_jsonl(heldout_manifest, ({"ref_id": "heldout/monster"},))
    c067_topk_path.write_text(
        json.dumps({"direct_green_non_human_face": [{"image_id": "train/leaf"}]}),
        encoding="utf-8",
    )

    summary = build_c069_direct_green_acquisition(
        C069Config(
            dataset_root=dataset_root,
            all_manifest_path=all_manifest,
            heldout_manifest_path=heldout_manifest,
            c067_topk_path=c067_topk_path,
            out_dir=tmp_path / "out",
            top_k_per_bucket=2,
        )
    )

    rows = _read_jsonl(tmp_path / "out" / "candidate_manifest.jsonl")
    labels = _read_jsonl(tmp_path / "out" / "reviewed_candidate_labels.jsonl")
    assert summary["scanned_image_count"] == 3
    assert summary["heldout_rows_used"] == 0
    assert summary["missing_paths"] == 0
    assert summary["scanned_beyond_c067_topk"] is True
    assert summary["direct_green_target_positive_count"] == 0
    assert summary["decision"] == "new_dataset_captioning_required"
    assert {row["image_id"] for row in rows}.isdisjoint({"heldout/monster"})
    assert {row["review_label"] for row in labels} == {"false_positive_background_object"}
    assert (tmp_path / "out" / "annotated_review_sheet.jpg").is_file()


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 128), color).save(path)


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return rows
