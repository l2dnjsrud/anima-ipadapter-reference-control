from __future__ import annotations

import json
from pathlib import Path

from tools.build_c065_failure_attribute_pairs import (
    C065Config,
    build_c065_failure_attribute_pairs,
)
from tools.siglip_auto_caption_types import JsonValue


def test_build_c065_pairs_excludes_heldout_and_reports_direct_green_gap(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    train_manifest = tmp_path / "train.jsonl"
    heldout_manifest = tmp_path / "heldout.jsonl"
    gate_summary = tmp_path / "summary.json"
    output_manifest = tmp_path / "pairs.jsonl"
    output_summary = tmp_path / "pairs.summary.json"

    for image_id in (
        "train/red_a",
        "train/red_b",
        "train/plain",
        "heldout/green",
    ):
        _write_image(dataset_root, image_id)
    _write_jsonl(
        train_manifest,
        (
            {"ref_id": "train/red_a", "tgt_id": "train/red_a", "prompt": "red a"},
            {"ref_id": "train/red_b", "tgt_id": "train/red_b", "prompt": "red b"},
            {"ref_id": "train/plain", "tgt_id": "train/plain", "prompt": "plain"},
        ),
    )
    _write_jsonl(
        heldout_manifest,
        ({"ref_id": "heldout/green", "tgt_id": "heldout/green", "prompt": "green"},),
    )
    gate_summary.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "split": "train",
                        "ref_id": "train/red_a",
                        "selected_attributes": [
                            "red glowing demonic eye",
                            "side profile portrait",
                        ],
                    },
                    {
                        "split": "train",
                        "ref_id": "train/red_b",
                        "selected_attributes": ["pale purple-skinned villain"],
                    },
                    {
                        "split": "train",
                        "ref_id": "train/plain",
                        "selected_attributes": ["calm young warrior"],
                    },
                    {
                        "split": "heldout",
                        "ref_id": "heldout/green",
                        "selected_attributes": ["green monster"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = build_c065_failure_attribute_pairs(
        C065Config(
            gate_summary_path=gate_summary,
            train_manifest_path=train_manifest,
            heldout_manifest_path=heldout_manifest,
            dataset_root=dataset_root,
            output_manifest_path=output_manifest,
            output_summary_path=output_summary,
        )
    )

    rows = _read_jsonl(output_manifest)
    assert [row["label"] for row in rows] == [
        "positive",
        "negative",
        "positive",
        "negative",
    ]
    assert {row["attribute_bucket"] for row in rows} == {
        "non_human_red_pale_profile_proxy"
    }
    assert {row["anchor_id"] for row in rows} == {"train/red_a", "train/red_b"}
    assert {row["candidate_id"] for row in rows if row["label"] == "negative"} == {
        "train/plain"
    }
    assert all("heldout/green" not in (row["anchor_id"], row["candidate_id"]) for row in rows)
    assert rows[0]["anchor_attributes"] == [
        "red glowing demonic eye",
        "side profile portrait",
    ]
    assert rows[1]["negative_reason"] == "candidate_not_in_attribute_bucket"
    assert rows[1]["source_split"] == "train"

    assert summary.heldout_rows_used == 0
    assert summary.total_pairs == 4
    assert summary.positive_pairs == 2
    assert summary.negative_pairs == 2
    assert summary.direct_green_monster_positive_count == 0
    assert summary.per_bucket_counts["non_human_red_pale_profile_proxy"] == {
        "source_rows": 2,
        "positive_pairs": 2,
        "negative_pairs": 2,
    }
    assert summary.path_verification_counts == {
        "train_rows": 3,
        "train_existing_images": 3,
        "train_missing_images": 0,
        "heldout_rows": 1,
        "heldout_existing_images": 1,
        "heldout_missing_images": 0,
        "pair_rows_with_existing_paths": 4,
        "pair_rows_with_missing_paths": 0,
    }
    saved_summary = json.loads(output_summary.read_text(encoding="utf-8"))
    assert saved_summary["direct_green_monster_positive_count"] == 0


def test_build_c065_pairs_resolves_path_like_sample_ids(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    train_manifest = tmp_path / "train.jsonl"
    heldout_manifest = tmp_path / "heldout.jsonl"
    gate_summary = tmp_path / "summary.json"

    for image_id in ("comic/a", "comic/b", "comic/plain"):
        _write_image(dataset_root, image_id)
    _write_jsonl(
        train_manifest,
        (
            {"ref_id": "comic/a", "tgt_id": "comic/a", "prompt": "a"},
            {"ref_id": "comic/b", "tgt_id": "comic/b", "prompt": "b"},
            {"ref_id": "comic/plain", "tgt_id": "comic/plain", "prompt": "plain"},
        ),
    )
    _write_jsonl(heldout_manifest, ())
    gate_summary.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": str(dataset_root / "comic" / "a.jpg"),
                        "selected_attributes": ["elderly"],
                    },
                    {
                        "sample_id": str(dataset_root / "comic" / "b.jpg"),
                        "selected_attributes": ["wrinkled"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    build_c065_failure_attribute_pairs(
        C065Config(
            gate_summary_path=gate_summary,
            train_manifest_path=train_manifest,
            heldout_manifest_path=heldout_manifest,
            dataset_root=dataset_root,
            output_manifest_path=tmp_path / "pairs.jsonl",
            output_summary_path=tmp_path / "pairs.summary.json",
        )
    )

    rows = _read_jsonl(tmp_path / "pairs.jsonl")
    assert {row["attribute_bucket"] for row in rows} == {"old_face_crop"}


def _write_image(dataset_root: Path, image_id: str) -> None:
    image_path = dataset_root / f"{image_id}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"jpg")


def _write_jsonl(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]
