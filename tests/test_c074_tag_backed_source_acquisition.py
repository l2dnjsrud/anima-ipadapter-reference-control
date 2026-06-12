from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from tools.c074_tag_backed_source_acquisition import C074Config, build_c074_tag_backed_source_acquisition
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c074_builds_tag_backed_package_and_ready_gate(tmp_path: Path) -> None:
    config = C074Config(out_dir=tmp_path / "out", scratch_dir=tmp_path / "scratch")
    pending = build_c074_tag_backed_source_acquisition(config, fetch_image=_fake_fetch)

    assert pending["inspected_source_count"] == 4
    assert pending["candidate_count"] == 10
    assert pending["downloaded_count"] == 10
    assert pending["large_downloads_performed"] is False
    assert pending["decision"] == "external_manual_data_required"
    assert (tmp_path / "scratch" / "contact_sheet.jpg").is_file()
    assert (tmp_path / "out" / "source_manifest.jsonl").is_file()
    assert _read_jsonl(tmp_path / "out" / "external_candidates.jsonl")[0]["suggested_label"] == "target_positive"

    _write_labels(tmp_path / "out" / "manual_visual_labels.csv", target_count=4)
    ready = build_c074_tag_backed_source_acquisition(config, fetch_image=_fake_fetch)

    assert ready["reviewed_rows"] == 10
    assert ready["target_positive_confirmed_count"] == 4
    assert ready["decision"] == "ready_for_encoder_training"
    assert ready["committed_external_image_count"] == 0
    rows = _read_jsonl(tmp_path / "out" / "reviewed_external_labels.jsonl")
    assert sum(1 for row in rows if row["manual_label"] == "target_positive") == 4


def test_c074_rejects_unknown_or_missing_labels(tmp_path: Path) -> None:
    config = C074Config(out_dir=tmp_path / "out", scratch_dir=tmp_path / "scratch")
    build_c074_tag_backed_source_acquisition(config, fetch_image=_fake_fetch)

    _write_custom_labels(tmp_path / "out" / "manual_visual_labels.csv", (("c074_neeko_c1_0", "bad_label"),))
    with pytest.raises(ValueError, match="unknown c074 label"):
        build_c074_tag_backed_source_acquisition(config, fetch_image=_fake_fetch)

    _write_custom_labels(tmp_path / "out" / "manual_visual_labels.csv", (("c074_neeko_c1_0", "target_positive"),))
    with pytest.raises(ValueError, match="missing c074 label"):
        build_c074_tag_backed_source_acquisition(config, fetch_image=_fake_fetch)


def _fake_fetch(_url: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 128), (60, 170, 90)).save(destination)
    return True


def _write_labels(path: Path, *, target_count: int) -> None:
    rows: list[tuple[str, str]] = []
    for cluster in (1, 2):
        for sample in range(5):
            candidate_id = f"c074_neeko_c{cluster}_{sample}"
            label = "target_positive" if len(rows) < target_count else "useful_proxy_non_human"
            rows.append((candidate_id, label))
    _write_custom_labels(path, tuple(rows))


def _write_custom_labels(path: Path, rows: tuple[tuple[str, str], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("candidate_id", "manual_label", "manual_note"), lineterminator="\n")
        writer.writeheader()
        for candidate_id, label in rows:
            writer.writerow({"candidate_id": candidate_id, "manual_label": label, "manual_note": "fixture visual review"})


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)
