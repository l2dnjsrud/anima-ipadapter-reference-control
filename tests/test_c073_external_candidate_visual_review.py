from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from tools.c073_external_candidate_visual_review import (
    C073ReviewError,
    C073VisualReviewConfig,
    DownloadOutcome,
    build_c073_external_candidate_visual_review,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c073_downloads_candidates_and_requires_manual_visual_labels(tmp_path: Path) -> None:
    candidates = _write_candidates(tmp_path)
    config = C073VisualReviewConfig(candidates_path=candidates, out_dir=tmp_path / "out", scratch_dir=tmp_path / "scratch")

    summary = build_c073_external_candidate_visual_review(config, fetch_image=_fake_fetch_image)

    assert summary["candidate_count"] == 3
    assert summary["source_c072_commit"] == "ad60ea7"
    assert summary["downloaded_count"] == 2
    assert summary["failed_count"] == 1
    assert summary["decision"] == "visual_labels_pending"
    assert summary["committed_external_image_count"] == 0
    assert (tmp_path / "out" / "download_manifest.jsonl").is_file()
    assert (tmp_path / "out" / "visual_label_template.csv").is_file()
    assert (tmp_path / "scratch" / "contact_sheet.jpg").is_file()

    _write_labels(
        tmp_path / "out" / "manual_visual_labels.csv",
        (
            ("c073_a", "target_positive", "visually confirmed green creature"),
            ("c073_b", "guard_false_positive_human", "human fox-girl proxy only"),
        ),
    )
    reviewed_summary = build_c073_external_candidate_visual_review(config, fetch_image=_fake_fetch_image)

    assert reviewed_summary["reviewed_rows"] == 2
    assert reviewed_summary["label_counts"] == {"target_positive": 1, "guard_false_positive_human": 1}
    assert reviewed_summary["unique_target_positive_count"] == 1
    assert reviewed_summary["target_positive_confirmed_count"] == 1
    assert reviewed_summary["minimum_target_positive_required"] == 4
    assert "target_positive" in reviewed_summary["label_schema"]
    assert reviewed_summary["decision"] == "external_manual_data_required"
    reviewed_rows = _read_jsonl(tmp_path / "out" / "reviewed_external_labels.jsonl")
    assert reviewed_rows[0]["visual_confirmation"] is True
    assert reviewed_rows[1]["visual_confirmation"] is False


def test_c073_rejects_missing_or_unknown_visual_label(tmp_path: Path) -> None:
    candidates = _write_candidates(tmp_path)
    config = C073VisualReviewConfig(candidates_path=candidates, out_dir=tmp_path / "out", scratch_dir=tmp_path / "scratch")
    _write_labels(tmp_path / "out" / "manual_visual_labels.csv", (("c073_a", "bad_label", "bad"),))

    with pytest.raises(C073ReviewError, match="unknown visual label"):
        build_c073_external_candidate_visual_review(config, fetch_image=_fake_fetch_image)

    _write_labels(tmp_path / "out" / "manual_visual_labels.csv", (("c073_a", "target_positive", "only one"),))
    with pytest.raises(C073ReviewError, match="missing visual label"):
        build_c073_external_candidate_visual_review(config, fetch_image=_fake_fetch_image)


def _fake_fetch_image(url: str, destination: Path, _timeout_seconds: float, _max_image_bytes: int) -> DownloadOutcome:
    if "fail" in url:
        return DownloadOutcome("failed", 0, 0, 0, "fixture_failure")
    color = (40, 180, 70) if "green" in url else (180, 120, 80)
    image = Image.new("RGB", (96, 128), color)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return DownloadOutcome("downloaded", destination.stat().st_size, 96, 128, "")


def _write_candidates(tmp_path: Path) -> Path:
    path = tmp_path / "candidates.jsonl"
    rows = (
        _candidate("c073_a", "external/source/a", "https://example.test/green.jpg"),
        _candidate("c073_b", "external/source/b", "https://example.test/human.jpg"),
        _candidate("c073_c", "external/source/c", "https://example.test/fail.jpg"),
    )
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def _candidate(candidate_id: str, image_id: str, image_path: str) -> JsonObject:
    return {
        "candidate_id": candidate_id,
        "image_id": image_id,
        "image_path": image_path,
        "source_bucket": "external_metadata_potential_direct_green_non_human",
        "suggested_label": "useful_proxy_non_human",
        "source_experiments": ["c072_external_source_discovery"],
        "source_labels": ["metadata_potential_direct_green_non_human"],
        "source_buckets": ["fixture/source"],
        "review_notes": ["fixture"],
        "rank": 1,
        "bucket_score": 1.0,
        "green_ratio": 0.0,
        "central_green_ratio": 0.0,
        "red_ratio": 0.0,
        "heldout_excluded": False,
        "path_exists": False,
        "external_source_url": "https://huggingface.co/datasets/fixture/source",
        "external_license_note": "cc0-1.0",
    }


def _write_labels(path: Path, rows: tuple[tuple[str, str, str], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("candidate_id", "manual_label", "manual_note"), lineterminator="\n")
        writer.writeheader()
        for candidate_id, manual_label, manual_note in rows:
            writer.writerow({"candidate_id": candidate_id, "manual_label": manual_label, "manual_note": manual_note})


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)
