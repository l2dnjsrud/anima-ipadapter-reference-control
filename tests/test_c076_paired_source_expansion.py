from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c072_source_probe import DatasetProbe
from tools.c076_paired_source_expansion import C076SourceExpansionConfig, build_c076_paired_source_expansion
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c076_packages_prior_seeds_and_metadata_candidates(tmp_path: Path) -> None:
    summary = build_c076_paired_source_expansion(
        C076SourceExpansionConfig(
            out_dir=tmp_path / "out",
            scratch_dir=tmp_path / "scratch",
            c074_labels_path=_c074_labels(tmp_path, count=3),
            heldout_manifest_path=_heldout(tmp_path),
            max_new_downloads=2,
        ),
        probes=_probes(),
        fetch_image=_fake_fetch,
    )

    assert summary["inspected_source_count"] == 3
    assert summary["candidate_count"] == 5
    assert summary["downloaded_count"] == 5
    assert summary["network_downloaded_count"] == 2
    assert summary["reviewed_rows"] == 5
    assert summary["target_positive_confirmed_count"] == 3
    assert summary["new_target_positive_confirmed_count"] == 0
    assert summary["source_probe_decision"] == "source_probe_ready_for_review"
    assert summary["decision"] == "more_data_required"
    assert summary["heldout_rows_used"] == 0
    assert summary["large_downloads_performed"] is False
    assert summary["committed_external_image_count"] == 0
    assert (tmp_path / "scratch" / "contact_sheet.jpg").is_file()
    assert (tmp_path / "out" / "source_manifest.jsonl").is_file()
    assert (tmp_path / "out" / "feature_boundary_metrics.json").is_file()

    rows = _read_jsonl(tmp_path / "out" / "reviewed_external_labels.jsonl")
    assert [row["manual_label"] for row in rows[:3]] == ["target_positive", "target_positive", "target_positive"]
    assert {row["manual_label"] for row in rows[3:]} == {"useful_proxy_non_human"}


def test_c076_writes_source_blocked_report_when_no_candidates(tmp_path: Path) -> None:
    summary = build_c076_paired_source_expansion(
        C076SourceExpansionConfig(
            out_dir=tmp_path / "out",
            scratch_dir=tmp_path / "scratch",
            c074_labels_path=tmp_path / "missing.jsonl",
            heldout_manifest_path=_heldout(tmp_path),
        ),
        probes=(
            DatasetProbe(
                repo="blocked/plain",
                official_url="https://huggingface.co/datasets/blocked/plain",
                access_status="public",
                license_note="unknown",
                metadata_probe_status="rows_ok",
                features=("image", "caption"),
                inspected_row_count=1,
                rows=({"caption": "ordinary human portrait", "image": {"src": "https://example.test/plain.jpg"}},),
                probe_note="fixture",
            ),
        ),
        fetch_image=_fake_fetch,
    )

    assert summary["decision"] == "source_blocked"
    assert summary["source_probe_decision"] == "source_blocked"
    assert summary["candidate_count"] == 0
    assert summary["contact_sheet_written"] is False
    assert (tmp_path / "out" / "source_blocked_report.md").is_file()


def _probes() -> tuple[DatasetProbe, ...]:
    return (
        DatasetProbe(
            repo="fixture/green",
            official_url="https://huggingface.co/datasets/fixture/green",
            access_status="public",
            license_note="cc0-1.0",
            metadata_probe_status="rows_ok",
            features=("image", "caption"),
            inspected_row_count=2,
            rows=(
                {"caption": "green skin monster girl with tail and fangs", "image": {"src": "https://example.test/green-a.jpg"}},
                {"caption": "colored skin lizard creature green face", "image": {"src": "https://example.test/green-b.jpg"}},
            ),
            probe_note="fixture",
        ),
        DatasetProbe(
            repo="fixture/plain",
            official_url="https://huggingface.co/datasets/fixture/plain",
            access_status="public",
            license_note="cc-by-4.0",
            metadata_probe_status="rows_ok",
            features=("image", "caption"),
            inspected_row_count=1,
            rows=({"caption": "plain human with green background", "image": {"src": "https://example.test/plain.jpg"}},),
            probe_note="fixture",
        ),
    )


def _fake_fetch(_url: str, destination: Path, _timeout_seconds: float, _max_image_bytes: int) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 128), (40, 180, 70)).save(destination)
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
                "image_path": f"https://example.test/seed-{index}.jpg",
                "local_image_path": str(image_path),
                "manual_label": "target_positive",
                "source_buckets": ["CyberHarem/neeko_leagueoflegends"],
                "external_license_note": "mit/NFA caution",
            }
        )
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def _heldout(tmp_path: Path) -> Path:
    path = tmp_path / "heldout.jsonl"
    path.write_text(json.dumps({"ref_id": "heldout/local"}) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)
