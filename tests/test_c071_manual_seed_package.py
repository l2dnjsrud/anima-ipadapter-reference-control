from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from tools.c071_import_manual_labels import C071ImportConfig, import_c071_manual_labels
from tools.c071_seed_package import C071PackageConfig, build_c071_seed_package
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c071_package_and_import_require_four_unique_target_positives(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    summary = build_c071_seed_package(
        C071PackageConfig(
            c068_reviewed_path=paths["c068"],
            c069_reviewed_path=paths["c069"],
            c070_reviewed_path=paths["c070"],
            heldout_manifest_path=paths["heldout"],
            out_dir=tmp_path / "out",
        )
    )
    assert summary["heldout_rows_used"] == 0
    assert summary["unique_candidate_count"] == 4
    assert summary["source_row_counts"] == {"c068": 1, "c069": 2, "c070": 2}
    assert summary["minimum_target_positive_required"] == 4
    assert "target_positive" in summary["label_schema"]
    assert (tmp_path / "out" / "annotation_template.csv").is_file()
    assert (tmp_path / "out" / "annotated_review_sheet.jpg").is_file()

    _write_labels(
        tmp_path / "labels.csv",
        (
            ("train/proxy_a", "target_positive"),
            ("train/proxy_b", "target_positive"),
            ("train/guard_human", "guard_false_positive_human"),
            ("train/guard_bg", "guard_false_positive_background_object"),
        ),
    )
    import_summary = import_c071_manual_labels(
        C071ImportConfig(
            annotation_candidates_path=tmp_path / "out" / "annotation_candidates.jsonl",
            manual_labels_path=tmp_path / "labels.csv",
            heldout_manifest_path=paths["heldout"],
            out_dir=tmp_path / "imported",
        )
    )
    assert import_summary["unique_target_positive_count"] == 2
    assert import_summary["decision"] == "external_manual_data_required"
    assert _read_jsonl(tmp_path / "imported" / "imported_confirmed_positives.jsonl")


def test_c071_import_rejects_unknown_label_heldout_and_duplicate_positive(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    build_c071_seed_package(
        C071PackageConfig(paths["c068"], paths["c069"], paths["c070"], paths["heldout"], tmp_path / "out")
    )

    _write_labels(tmp_path / "unknown.csv", (("train/proxy_a", "bad_label"),))
    with pytest.raises(ValueError, match="unknown label"):
        import_c071_manual_labels(
            C071ImportConfig(tmp_path / "out" / "annotation_candidates.jsonl", tmp_path / "unknown.csv", paths["heldout"], tmp_path / "bad")
        )

    _write_labels(tmp_path / "heldout.csv", (("heldout/monster", "target_positive"),))
    with pytest.raises(ValueError, match="heldout"):
        import_c071_manual_labels(
            C071ImportConfig(tmp_path / "out" / "annotation_candidates.jsonl", tmp_path / "heldout.csv", paths["heldout"], tmp_path / "bad")
        )

    _write_labels(
        tmp_path / "dup.csv",
        (("train/proxy_a", "target_positive"), ("train/proxy_a", "target_positive")),
    )
    with pytest.raises(ValueError, match="duplicate target_positive"):
        import_c071_manual_labels(
            C071ImportConfig(tmp_path / "out" / "annotation_candidates.jsonl", tmp_path / "dup.csv", paths["heldout"], tmp_path / "bad")
        )


def _fixtures(tmp_path: Path) -> dict[str, Path]:
    dataset = tmp_path / "dataset"
    _write_image(dataset / "train/proxy_a.jpg", (160, 30, 30))
    _write_image(dataset / "train/proxy_b.jpg", (30, 140, 70))
    _write_image(dataset / "train/guard_human.jpg", (120, 120, 120))
    _write_image(dataset / "train/guard_bg.jpg", (40, 170, 60))
    _write_image(dataset / "heldout/monster.jpg", (20, 180, 50))
    paths = {
        "c068": tmp_path / "c068.jsonl",
        "c069": tmp_path / "c069.jsonl",
        "c070": tmp_path / "c070.jsonl",
        "heldout": tmp_path / "heldout.jsonl",
    }
    _write_jsonl(
        paths["c068"],
        (
            _row("train/proxy_a", dataset / "train/proxy_a.jpg", "useful_proxy_positive", "side_profile_silhouette"),
            _row("heldout/monster", dataset / "heldout/monster.jpg", "useful_proxy_positive", "direct_green_non_human_face"),
        ),
    )
    _write_jsonl(
        paths["c069"],
        (
            _row("train/proxy_a", dataset / "train/proxy_a.jpg", "useful_proxy_non_human", "red_green_mix"),
            _row("train/proxy_b", dataset / "train/proxy_b.jpg", "useful_proxy_non_human", "red_green_mix"),
        ),
    )
    _write_jsonl(
        paths["c070"],
        (
            _row("train/guard_human", dataset / "train/guard_human.jpg", "false_positive_human", "semantic_target_fallback"),
            _row("train/guard_bg", dataset / "train/guard_bg.jpg", "false_positive_background_object", "background_green_guard"),
        ),
    )
    _write_jsonl(paths["heldout"], ({"ref_id": "heldout/monster"},))
    return paths


def _row(image_id: str, image_path: Path, review_label: str, source_bucket: str) -> JsonObject:
    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "review_label": review_label,
        "review_note": "fixture",
        "source_bucket": source_bucket,
        "rank": 1,
        "bucket_score": 0.5,
        "green_ratio": 0.1,
        "central_green_ratio": 0.1,
        "red_ratio": 0.1,
        "heldout_excluded": False,
        "path_exists": True,
    }


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 128), color).save(path)


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return rows


def _write_labels(path: Path, rows: tuple[tuple[str, str], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", "manual_label"))
        writer.writeheader()
        for image_id, manual_label in rows:
            writer.writerow({"image_id": image_id, "manual_label": manual_label})
