from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from tools.c077_hf_sample_sources import C077TreeProbe
from tools.c077_target_positive_acquisition import C077AcquisitionConfig, build_c077_target_positive_acquisition
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c077_promotes_only_when_new_target_threshold_is_met(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.csv"
    _write_labels(labels_path, target_count=12)

    summary = build_c077_target_positive_acquisition(
        C077AcquisitionConfig(
            out_dir=tmp_path / "out",
            scratch_dir=tmp_path / "scratch",
            c074_labels_path=_c074_labels(tmp_path, count=12),
            heldout_manifest_path=_heldout(tmp_path),
            labels_path=labels_path,
            max_new_downloads=12,
            max_per_source=12,
        ),
        probes=_tree_probes(count=12),
        fetch_image=_fake_fetch,
    )

    assert summary["candidate_count"] == 24
    assert summary["downloaded_count"] == 24
    assert summary["new_downloaded_count"] == 12
    assert summary["target_positive_confirmed_count"] == 24
    assert summary["new_target_positive_confirmed_count"] == 12
    assert summary["decision"] == "ready_for_c077_training_manifest"
    assert summary["heldout_rows_used"] == 0
    assert summary["raw_external_images_committed"] is False
    assert (tmp_path / "scratch" / "contact_sheet.jpg").is_file()
    assert (tmp_path / "out" / "reviewed_external_labels.jsonl").is_file()


def test_c077_reports_manual_needed_when_tree_candidates_are_not_targets(tmp_path: Path) -> None:
    summary = build_c077_target_positive_acquisition(
        C077AcquisitionConfig(
            out_dir=tmp_path / "out",
            scratch_dir=tmp_path / "scratch",
            c074_labels_path=_c074_labels(tmp_path, count=10),
            heldout_manifest_path=_heldout(tmp_path),
            max_new_downloads=3,
        ),
        probes=_tree_probes(count=3),
        fetch_image=_fake_fetch,
    )

    assert summary["candidate_count"] == 13
    assert summary["target_positive_confirmed_count"] == 10
    assert summary["new_target_positive_confirmed_count"] == 0
    assert summary["decision"] == "manual_needed_more_target_positives"
    assert summary["source_probe_decision"] == "source_probe_ready_for_review"
    assert (tmp_path / "out" / "manual_needed_report.md").is_file()


def test_c077_source_blocked_when_no_new_sample_assets_exist(tmp_path: Path) -> None:
    summary = build_c077_target_positive_acquisition(
        C077AcquisitionConfig(
            out_dir=tmp_path / "out",
            scratch_dir=tmp_path / "scratch",
            c074_labels_path=tmp_path / "missing.jsonl",
            heldout_manifest_path=_heldout(tmp_path),
        ),
        probes=(
            C077TreeProbe(
                repo="fixture/empty",
                official_url="https://huggingface.co/datasets/fixture/empty",
                access_status="public",
                license_note="mit",
                path_status="paths_ok",
                inspected_path_count=0,
                sample_paths=(),
                source_note="fixture",
            ),
        ),
        fetch_image=_fake_fetch,
    )

    assert summary["candidate_count"] == 0
    assert summary["new_candidate_count"] == 0
    assert summary["decision"] == "source_blocked_manual_needed"
    assert summary["source_probe_decision"] == "source_blocked"
    assert summary["contact_sheet_written"] is False
    assert (tmp_path / "out" / "source_blocked_report.md").is_file()


def _tree_probes(*, count: int) -> tuple[C077TreeProbe, ...]:
    paths = tuple(f"samples/0/clu0-sample{index}.png" for index in range(count))
    return (
        C077TreeProbe(
            repo="fixture/green_probe",
            official_url="https://huggingface.co/datasets/fixture/green_probe",
            access_status="public",
            license_note="mit",
            path_status="paths_ok",
            inspected_path_count=count,
            sample_paths=paths,
            source_note="fixture direct sample assets",
        ),
    )


def _fake_fetch(_url: str, destination: Path, _timeout_seconds: float, _max_image_bytes: int) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 128), (50, 180, 70)).save(destination)
    return True


def _c074_labels(tmp_path: Path, *, count: int) -> Path:
    path = tmp_path / "c074.jsonl"
    rows: list[JsonObject] = []
    for index in range(count):
        image_path = tmp_path / "c074" / f"seed-{index}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (96, 128), (60, 170, 90)).save(image_path)
        rows.append(
            {
                "candidate_id": f"c074_seed_{index}",
                "image_id": f"external/c074/seed-{index}",
                "local_image_path": str(image_path),
                "manual_label": "target_positive",
                "external_license_note": "mit/NFA caution",
            }
        )
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def _heldout(tmp_path: Path) -> Path:
    path = tmp_path / "heldout.jsonl"
    path.write_text(json.dumps({"ref_id": "external/fixture/green_probe/0/clu0-sample99"}) + "\n", encoding="utf-8")
    return path


def _write_labels(path: Path, *, target_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("candidate_id", "manual_label", "manual_note"), lineterminator="\n")
        writer.writeheader()
        for index in range(target_count):
            writer.writerow(
                {
                    "candidate_id": f"c077_fixture_green_probe_0_clu0_sample{index}",
                    "manual_label": "target_positive",
                    "manual_note": "fixture new target-positive direct-green non-human",
                }
            )


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)
