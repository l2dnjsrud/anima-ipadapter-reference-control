from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.build_c066_direct_green_candidates import (
    C066Config,
    build_c066_direct_green_candidates,
)
from tools.siglip_auto_caption_types import JsonValue


def test_build_c066_candidates_excludes_heldout_and_writes_balanced_pairs(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    train_manifest = tmp_path / "train.jsonl"
    heldout_manifest = tmp_path / "heldout.jsonl"
    gate_summary = tmp_path / "summary.json"
    candidate_manifest = tmp_path / "candidates.jsonl"
    candidate_summary = tmp_path / "candidates.summary.json"
    pair_manifest = tmp_path / "pairs.jsonl"

    _write_image(dataset_root, "train/green_a", (20, 180, 40))
    _write_image(dataset_root, "train/green_b", (30, 170, 50))
    _write_image(dataset_root, "train/red_a", (150, 30, 30))
    _write_image(dataset_root, "train/red_b", (160, 35, 35))
    _write_image(dataset_root, "train/human", (120, 110, 100))
    _write_image(dataset_root, "train/plain", (90, 90, 90))
    _write_image(dataset_root, "heldout/green", (20, 190, 40))
    _write_jsonl(
        train_manifest,
        (
            {"ref_id": "train/green_a"},
            {"ref_id": "train/green_b"},
            {"ref_id": "train/red_a"},
            {"ref_id": "train/red_b"},
            {"ref_id": "train/human"},
            {"ref_id": "train/plain"},
        ),
    )
    _write_jsonl(heldout_manifest, ({"ref_id": "heldout/green"},))
    gate_summary.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "split": "train",
                        "ref_id": "train/red_a",
                        "selected_attributes": ["red glowing demonic eye"],
                    },
                    {
                        "split": "train",
                        "ref_id": "train/red_b",
                        "selected_attributes": ["side profile portrait", "red glowing demonic eye"],
                    },
                    {
                        "split": "train",
                        "ref_id": "train/human",
                        "selected_attributes": ["human martial arts character", "old bearded martial arts master"],
                    },
                    {
                        "split": "heldout",
                        "ref_id": "heldout/green",
                        "selected_attributes": ["green monster face with red glowing eye"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = build_c066_direct_green_candidates(
        C066Config(
            dataset_root=dataset_root,
            train_manifest_path=train_manifest,
            heldout_manifest_path=heldout_manifest,
            gate_summary_path=gate_summary,
            c065_pair_manifest_path=None,
            output_manifest_path=candidate_manifest,
            output_summary_path=candidate_summary,
            output_pair_manifest_path=pair_manifest,
            max_per_bucket=10,
            green_ratio_min=0.2,
            strong_green_ratio_min=0.1,
        )
    )

    candidates = _read_jsonl(candidate_manifest)
    pairs = _read_jsonl(pair_manifest)
    image_ids = {str(row["image_id"]) for row in candidates}
    assert "heldout/green" not in image_ids
    assert {"train/green_a", "train/green_b"} <= image_ids
    assert summary.heldout_rows_used == 0
    assert summary.missing_paths == 0
    assert summary.sidecar_caption_keyword_hits == 0
    assert summary.direct_green_positive_count == 0
    assert summary.source_buckets["direct_green_pixel_candidate"] == 2
    assert summary.source_buckets["red_eye_proxy"] == 2
    assert summary.source_buckets["human_negative"] == 1
    assert summary.pair_rows == 8
    assert {row["label"] for row in pairs} == {"positive", "negative"}
    assert all("heldout/green" not in (row["anchor_id"], row["candidate_id"]) for row in pairs)


def _write_image(dataset_root: Path, image_id: str, color: tuple[int, int, int]) -> None:
    image_path = dataset_root / f"{image_id}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(image_path)
    image_path.with_suffix(".txt").write_text(
        "mrcolor_panel_style, full color manga panel, character panel",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]
