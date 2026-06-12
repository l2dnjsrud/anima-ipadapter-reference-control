from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.c072_external_source_discovery import (
    C072DiscoveryConfig,
    DatasetProbe,
    build_c072_external_source_discovery,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c072_builds_metadata_only_external_candidate_package(tmp_path: Path) -> None:
    summary = build_c072_external_source_discovery(
        C072DiscoveryConfig(heldout_manifest_path=_heldout(tmp_path), out_dir=tmp_path / "out"),
        probes=_positive_probes(),
    )

    assert summary["inspected_source_count"] == 2
    assert summary["large_downloads_performed"] is False
    assert summary["heldout_rows_used"] == 0
    assert summary["candidate_package_status"] == "metadata_only_manual_review_required"
    assert summary["unique_potential_target_positive_count"] == 4
    assert summary["target_positive_confirmed_count"] == 0
    assert summary["decision"] == "external_candidates_found_manual_confirmation_required"

    candidates = _read_jsonl(tmp_path / "out" / "external_candidates.jsonl")
    assert len(candidates) == 4
    assert candidates[0]["image_id"] == "external/mrzjy/AniGamePersonaCaps/0"
    assert {str(row["suggested_label"]) for row in candidates} == {"useful_proxy_non_human"}
    assert all(str(row["image_id"]).startswith("external/") for row in candidates)
    assert all(row["path_exists"] is False for row in candidates)
    assert all("example.test" not in str(row["review_notes"][0]) for row in candidates)
    assert (tmp_path / "out" / "external_candidate_template.csv").is_file()
    assert (tmp_path / "out" / "source_manifest.jsonl").is_file()
    assert "external_candidates_found_manual_confirmation_required" in (tmp_path / "out" / "report.md").read_text(
        encoding="utf-8"
    )

    with (tmp_path / "out" / "external_candidate_template.csv").open(encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    assert rows[0]["manual_label"] == ""
    assert "target_positive" in rows[0]["allowed_labels"]


def test_c072_records_no_safe_package_when_sources_lack_candidates(tmp_path: Path) -> None:
    summary = build_c072_external_source_discovery(
        C072DiscoveryConfig(heldout_manifest_path=_heldout(tmp_path), out_dir=tmp_path / "out"),
        probes=(
            DatasetProbe(
                repo="blocked/source",
                official_url="https://huggingface.co/datasets/blocked/source",
                access_status="public",
                license_note="unknown",
                metadata_probe_status="rows_ok",
                features=("image", "caption"),
                inspected_row_count=1,
                rows=({"title": "plain human", "caption": "fair skin, black hair", "image": "https://example.test/a.jpg"},),
                probe_note="fixture",
            ),
        ),
    )

    assert summary["candidate_package_status"] == "no_safe_external_candidate_package"
    assert summary["unique_potential_target_positive_count"] == 0
    assert summary["decision"] == "external_manual_data_required"
    assert not (tmp_path / "out" / "external_candidates.jsonl").exists()
    assert "no_safe_external_candidate_package" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def _positive_probes() -> tuple[DatasetProbe, ...]:
    return (
        DatasetProbe(
            repo="alfredplpl/anime-with-caption-cc0",
            official_url="https://huggingface.co/datasets/alfredplpl/anime-with-caption-cc0",
            access_status="public",
            license_note="cc0-1.0",
            metadata_probe_status="rows_ok",
            features=("image", "prompt", "phi3_caption"),
            inspected_row_count=3,
            rows=(
                _row("green skin, colored skin, fangs, 1girl", "https://example.test/green_skin.jpg"),
                _row("furry, green background, wolf ears, solo", "https://example.test/furry_green_bg.jpg"),
                _row("green hair, rabbit ears, anthropomorphic appearance", "https://example.test/rabbit.jpg"),
            ),
            probe_note="fixture",
        ),
        DatasetProbe(
            repo="mrzjy/AniGamePersonaCaps",
            official_url="https://huggingface.co/datasets/mrzjy/AniGamePersonaCaps",
            access_status="public",
            license_note="cc-by-sa-4.0",
            metadata_probe_status="rows_ok",
            features=("image", "title", "caption"),
            inspected_row_count=1,
            rows=(
                {
                    "title": "Conway",
                    "caption": "unique glowing green skin and eyes; cartoonish anthropomorphic creature",
                    "image_url": "https://example.test/conway.jpg",
                },
            ),
            probe_note="fixture",
        ),
    )


def _row(prompt: str, image_url: str) -> JsonObject:
    return {"prompt": prompt, "phi3_caption": prompt, "image": {"src": image_url}}


def _heldout(tmp_path: Path) -> Path:
    path = tmp_path / "heldout.jsonl"
    path.write_text(json.dumps({"ref_id": "heldout/local"}) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value: JsonValue = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return tuple(rows)
