from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c100_local_positive_gate import C100Config, build_c100_local_positive_package, config_for_cli
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c100_blocks_without_reviewed_local_positive(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _image(data_root / "train/green_a.jpg", (20, 180, 40))
    _image(data_root / "train/fang_a.jpg", (170, 190, 190))
    _image(data_root / "train/red_a.jpg", (190, 30, 30))
    _image(data_root / "heldout/green.jpg", (20, 180, 40))

    out_dir = tmp_path / "out"
    summary = build_c100_local_positive_package(
        _config(
            tmp_path,
            out_dir,
            c066_rows=(
                _c066("train/green_a", data_root / "train/green_a.jpg", "direct_green_pixel_candidate"),
                _c066("train/fang_a", data_root / "train/fang_a.jpg", "fang_profile_proxy"),
                _c066("train/red_a", data_root / "train/red_a.jpg", "red_eye_proxy"),
                _c066("heldout/green", data_root / "heldout/green.jpg", "direct_green_pixel_candidate"),
            ),
        )
    )

    rows = _read_jsonl(out_dir / "c100_candidate_manifest.jsonl")
    ids = {str(row["image_id"]) for row in rows}
    assert "heldout/green" not in ids
    assert summary["candidate_rows"] == 3
    assert summary["heldout_leakage_count"] == 0
    assert summary["missing_path_count"] == 0
    assert summary["reviewed_local_positive_count"] == 0
    assert summary["review_required_count"] == 3
    assert summary["decision"] == "c101_blocked_needs_manual_annotation_or_teacher"
    assert (out_dir / "c100_candidate_review_sheet.jpg").is_file()


def test_c100_greenlights_with_enough_reviewed_local_positive(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _image(data_root / "train/green_a.jpg", (20, 180, 40))
    _image(data_root / "train/fang_a.jpg", (170, 190, 190))
    out_dir = tmp_path / "out"

    summary = build_c100_local_positive_package(
        _config(
            tmp_path,
            out_dir,
            c066_rows=(
                _c066("train/green_a", data_root / "train/green_a.jpg", "direct_green_pixel_candidate"),
                _c066("train/fang_a", data_root / "train/fang_a.jpg", "fang_profile_proxy"),
            ),
            review_rows=(
                {"image_id": "train/green_a", "manual_label": "local_positive"},
                {"image_id": "train/fang_a", "manual_label": "local_positive"},
            ),
            min_reviewed_positive=2,
        )
    )

    assert summary["candidate_rows"] == 2
    assert summary["reviewed_local_positive_count"] == 2
    assert summary["decision"] == "c101_training_greenlit"
    assert "training/" in str(summary["next_c101_command_surface"])


def test_c100_cli_config_keeps_review_labels_inside_out_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "custom-out"

    config = config_for_cli(out_dir)

    assert config.out_dir == out_dir
    assert config.review_labels == out_dir / "reviewed_local_labels.jsonl"


def _config(
    tmp_path: Path,
    out_dir: Path,
    *,
    c066_rows: tuple[JsonObject, ...],
    review_rows: tuple[JsonObject, ...] = (),
    min_reviewed_positive: int = 8,
) -> C100Config:
    c066_manifest = tmp_path / "c066.jsonl"
    heldout_manifest = tmp_path / "heldout.jsonl"
    c066_summary = tmp_path / "c066.summary.json"
    c099_summary = tmp_path / "c099.summary.json"
    c099_inventory = tmp_path / "c099.inventory.json"
    review_labels = tmp_path / "review.jsonl"
    _write_jsonl(c066_manifest, c066_rows)
    _write_jsonl(heldout_manifest, ({"ref_id": "heldout/green"},))
    _write_jsonl(review_labels, review_rows)
    c066_summary.write_text(
        json.dumps(
            {
                "direct_green_positive_count": 0,
                "source_buckets": {"direct_green_pixel_candidate": 2},
            }
        ),
        encoding="utf-8",
    )
    c099_summary.write_text(
        json.dumps({"decision": "c100_blocked_needs_annotation_or_teacher"}),
        encoding="utf-8",
    )
    c099_inventory.write_text(
        json.dumps({"key_metrics": {"c066_direct_green_positive_count": 0}}),
        encoding="utf-8",
    )
    return C100Config(
        c066_manifest=c066_manifest,
        c066_summary=c066_summary,
        c099_summary=c099_summary,
        c099_inventory=c099_inventory,
        heldout_manifest=heldout_manifest,
        review_labels=review_labels,
        out_dir=out_dir,
        plan_path=tmp_path / "plan.md",
        min_reviewed_positive=min_reviewed_positive,
    )


def _c066(image_id: str, image_path: Path, bucket: str) -> JsonObject:
    return {
        "image_id": image_id,
        "label": "positive",
        "source_bucket": bucket,
        "candidate_source": "test",
        "caption": "caption",
        "image_path": str(image_path),
        "green_ratio": 0.2,
        "strong_green_ratio": 0.1,
        "red_ratio": 0.0,
        "path_exists": image_path.is_file(),
    }


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path)


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
