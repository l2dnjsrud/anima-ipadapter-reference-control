from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tools.c071_seed_package import LABEL_SCHEMA
from tools.c072_source_probe import SOURCES, DatasetProbe, fetch_probe
from tools.c076_source_expansion_report import blocked_report, feature_boundary_metrics, next_action, report
from tools.c076_source_expansion_io import (
    FetchImage,
    dimensions,
    fetch_image_to_path,
    read_ids,
    read_jsonl,
    read_label_map,
    write_jsonl,
    write_sheet,
    write_template,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue

OUT_DIR: Final = Path("eval/c076_paired_direct_green_source_expansion_20260612")
SCRATCH: Final = Path(".tmp/c076_paired_direct_green_source_expansion")
C074_LABELS: Final = Path("eval/c074_tag_backed_direct_green_source_acquisition_20260612/reviewed_external_labels.jsonl")
HELDOUT: Final = Path("training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl")
MINIMUM_TARGET_POSITIVES: Final = 24
MAX_BYTES: Final = 4_194_304
GREEN_TERMS: Final = ("green skin", "colored skin", "green body", "green face", "green creature", "green fur")
NON_HUMAN_TERMS: Final = ("monster", "creature", "demon", "alien", "anthro", "furry", "tail", "fang", "dragon", "lizard")


@dataclass(frozen=True, slots=True)
class C076SourceExpansionConfig:
    out_dir: Path = OUT_DIR
    scratch_dir: Path = SCRATCH
    c074_labels_path: Path = C074_LABELS
    heldout_manifest_path: Path = HELDOUT
    row_limit: int = 80
    timeout_seconds: float = 8.0
    max_image_bytes: int = MAX_BYTES
    max_new_downloads: int = 8
    labels_path: Path | None = None


def build_c076_paired_source_expansion(
    config: C076SourceExpansionConfig,
    *,
    probes: tuple[DatasetProbe, ...] | None = None,
    fetch_image: FetchImage | None = None,
) -> JsonObject:
    selected_fetch = fetch_image_to_path if fetch_image is None else fetch_image
    config.out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = config.scratch_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    heldout_ids = read_ids(config.heldout_manifest_path)
    source_probes = probes if probes is not None else tuple(
        fetch_probe(source, row_limit=config.row_limit, timeout_seconds=config.timeout_seconds) for source in SOURCES
    )
    c074_rows = read_jsonl(config.c074_labels_path) if config.c074_labels_path.is_file() else ()
    metadata_candidates = _metadata_candidates(source_probes, heldout_ids=heldout_ids)
    seed_candidates = _seed_candidates(c074_rows, heldout_ids=heldout_ids)
    downloads = _materialize(seed_candidates, metadata_candidates, images_dir, config, selected_fetch)
    labels_path = config.out_dir / "manual_visual_labels.csv" if config.labels_path is None else config.labels_path
    label_map = read_label_map(labels_path) if labels_path.is_file() else {}
    reviewed = _review_downloads(downloads, label_map)
    source_rows = _source_rows(source_probes, c074_rows, metadata_candidates)
    write_jsonl(config.out_dir / "source_manifest.jsonl", source_rows)
    write_jsonl(config.out_dir / "external_candidates.jsonl", seed_candidates + metadata_candidates)
    write_jsonl(config.out_dir / "download_manifest.jsonl", downloads)
    write_jsonl(config.out_dir / "reviewed_external_labels.jsonl", reviewed)
    write_template(config.out_dir / "visual_label_template.csv", downloads)
    sheet_path = config.scratch_dir / "contact_sheet.jpg"
    sheet_written = write_sheet(downloads, sheet_path)
    feature_metrics = feature_boundary_metrics(reviewed)
    (config.out_dir / "feature_boundary_metrics.json").write_text(json.dumps(feature_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = _summary(source_rows, downloads, reviewed, feature_metrics, sheet_path, sheet_written)
    (config.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (config.out_dir / "report.md").write_text(report(summary), encoding="utf-8")
    if summary["decision"] == "source_blocked":
        (config.out_dir / "source_blocked_report.md").write_text(blocked_report(summary), encoding="utf-8")
    return summary


def _metadata_candidates(probes: tuple[DatasetProbe, ...], *, heldout_ids: set[str]) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for probe in probes:
        for row_index, row in enumerate(probe.rows):
            score = _score(row)
            image_url = _image_url(row)
            image_id = f"external/{probe.repo}/{row_index}"
            if score < 6 or not image_url or image_id in heldout_ids:
                continue
            rows.append(
                {
                    "candidate_id": f"c076_meta_{len(rows):03d}",
                    "image_id": image_id,
                    "image_path": image_url,
                    "source_bucket": "metadata_direct_green_non_human_probe",
                    "suggested_label": "useful_proxy_non_human",
                    "source_experiments": ["c076_paired_direct_green_source_expansion"],
                    "source_labels": ["metadata_green_non_human"],
                    "source_buckets": [probe.repo],
                    "review_notes": [_excerpt(row)],
                    "rank": len(rows) + 1,
                    "bucket_score": float(score),
                    "heldout_excluded": False,
                    "external_source_url": probe.official_url,
                    "external_license_note": probe.license_note,
                    "review_source": "metadata_probe_needs_visual_confirmation",
                }
            )
    rows.sort(key=lambda row: float(row["bucket_score"]), reverse=True)
    return tuple(rows[:12])


def _seed_candidates(rows: tuple[JsonObject, ...], *, heldout_ids: set[str]) -> tuple[JsonObject, ...]:
    seeds: list[JsonObject] = []
    for row in rows:
        if row.get("manual_label") != "target_positive" or str(row.get("image_id", "")) in heldout_ids:
            continue
        seeds.append(dict(row) | {"candidate_id": f"c076_seed_{row['candidate_id']}", "review_source": "prior_c074_manual"})
    return tuple(seeds)


def _materialize(
    seeds: tuple[JsonObject, ...],
    metadata: tuple[JsonObject, ...],
    images_dir: Path,
    config: C076SourceExpansionConfig,
    fetch_image_fn: FetchImage,
) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for row in seeds:
        rows.append(_copy_seed(row, images_dir))
    for row in metadata[: config.max_new_downloads]:
        destination = images_dir / f"{row['candidate_id']}.jpg"
        ok = fetch_image_fn(str(row["image_path"]), destination, config.timeout_seconds, config.max_image_bytes)
        width, height = dimensions(destination) if ok else (0, 0)
        rows.append(dict(row) | {"download_status": "downloaded" if ok else "failed", "local_image_path": str(destination), "image_width": width, "image_height": height, "large_downloads_performed": False})
    return tuple(rows)


def _copy_seed(row: JsonObject, images_dir: Path) -> JsonObject:
    source = Path(str(row.get("local_image_path", "")))
    destination = images_dir / f"{row['candidate_id']}.jpg"
    if source.is_file():
        shutil.copyfile(source, destination)
        width, height = dimensions(destination)
        return dict(row) | {"download_status": "copied_prior_c074", "local_image_path": str(destination), "image_width": width, "image_height": height}
    return dict(row) | {"download_status": "missing_prior_c074", "local_image_path": str(destination), "image_width": 0, "image_height": 0}


def _review_downloads(rows: tuple[JsonObject, ...], labels: dict[str, JsonObject]) -> tuple[JsonObject, ...]:
    reviewed: list[JsonObject] = []
    for row in rows:
        status = str(row["download_status"])
        if status not in {"downloaded", "copied_prior_c074"}:
            continue
        review_source = str(row.get("review_source", ""))
        label_row = labels.get(str(row["candidate_id"]), {})
        manual_label = str(label_row.get("manual_label") or ("target_positive" if review_source == "prior_c074_manual" else "useful_proxy_non_human"))
        if manual_label not in LABEL_SCHEMA:
            raise ValueError(f"unknown c076 manual label: {manual_label}")
        note = str(label_row.get("manual_note") or _review_note(review_source))
        reviewed.append(dict(row) | {"manual_label": manual_label, "manual_note": note, "visual_confirmation": manual_label == "target_positive"})
    return tuple(reviewed)


def _review_note(review_source: str) -> str:
    if review_source == "prior_c074_manual":
        return "prior c074 manual label carried forward; not new c076 evidence"
    return "auto proxy from metadata/color availability; requires human visual confirmation before training as target-positive"


def _source_rows(probes: tuple[DatasetProbe, ...], c074_rows: tuple[JsonObject, ...], candidates: tuple[JsonObject, ...]) -> tuple[JsonObject, ...]:
    rows = [
        {
            "repo": "c074_confirmed_seed/CyberHarem/neeko_leagueoflegends",
            "official_url": "eval/c074_tag_backed_direct_green_source_acquisition_20260612",
            "access_status": "local_prior_review",
            "license_note": "mit/NFA/source-rights caution carried from c074",
            "metadata_probe_status": "prior_reviewed_rows",
            "inspected_row_count": len(c074_rows),
            "potential_candidate_count": sum(1 for row in c074_rows if row.get("manual_label") == "target_positive"),
            "probe_note": "baseline positives already failed c075 quality gate alone",
        }
    ]
    for probe in probes:
        rows.append(
            {
                "repo": probe.repo,
                "official_url": probe.official_url,
                "access_status": probe.access_status,
                "license_note": probe.license_note,
                "metadata_probe_status": probe.metadata_probe_status,
                "features": list(probe.features),
                "inspected_row_count": probe.inspected_row_count,
                "potential_candidate_count": sum(1 for row in candidates if probe.repo in row.get("source_buckets", [])),
                "probe_note": probe.probe_note,
            }
        )
    return tuple(rows)


def _summary(rows: tuple[JsonObject, ...], downloads: tuple[JsonObject, ...], reviewed: tuple[JsonObject, ...], metrics: JsonObject, sheet_path: Path, sheet_written: bool) -> JsonObject:
    target_count = len({str(row["image_id"]) for row in reviewed if row["manual_label"] == "target_positive"})
    new_target_count = sum(1 for row in reviewed if row["manual_label"] == "target_positive" and row.get("review_source") != "prior_c074_manual")
    candidate_count = sum(int(row["potential_candidate_count"]) for row in rows)
    decision = "source_blocked" if candidate_count == 0 else ("ready_for_c077_training" if target_count >= MINIMUM_TARGET_POSITIVES and new_target_count >= 12 else "more_data_required")
    source_probe_decision = "source_blocked" if candidate_count == 0 else "source_probe_ready_for_review"
    return {
        "source": "c076_paired_direct_green_source_expansion",
        "inspected_source_count": len(rows),
        "candidate_count": candidate_count,
        "downloaded_count": sum(1 for row in downloads if row["download_status"] in {"downloaded", "copied_prior_c074"}),
        "network_downloaded_count": sum(1 for row in downloads if row["download_status"] == "downloaded"),
        "reviewed_rows": len(reviewed),
        "target_positive_confirmed_count": target_count,
        "new_target_positive_confirmed_count": new_target_count,
        "unique_target_positive_count": target_count,
        "minimum_target_positive_required": MINIMUM_TARGET_POSITIVES,
        "minimum_new_target_positive_required": 12,
        "label_schema": list(LABEL_SCHEMA),
        "heldout_rows_used": sum(1 for row in downloads if bool(row.get("heldout_excluded", False))),
        "large_downloads_performed": False,
        "committed_external_image_count": 0,
        "contact_sheet_path": str(sheet_path),
        "contact_sheet_written": sheet_written,
        "feature_boundary_status": metrics["status"],
        "source_probe_decision": source_probe_decision,
        "candidate_review_decision": decision,
        "decision": decision,
        "next_training_or_data_action": next_action(decision),
    }


def _score(row: JsonObject) -> int:
    text = _text(row)
    return sum(3 for term in GREEN_TERMS if term in text) + sum(2 for term in NON_HUMAN_TERMS if term in text)


def _text(value: JsonValue) -> str:
    match value:
        case dict():
            return " ".join(_text(child) for key, child in value.items() if key not in {"image", "image_url", "jpg", "url"}).lower()
        case list():
            return " ".join(_text(child) for child in value).lower()
        case str():
            return value.lower()
        case None | bool() | int() | float():
            return str(value).lower()


def _image_url(row: JsonObject) -> str:
    for key in ("image_url", "image", "jpg"):
        value = row.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value
        if isinstance(value, dict) and isinstance(value.get("src"), str) and str(value["src"]).startswith("http"):
            return str(value["src"])
    return ""


def _excerpt(row: JsonObject) -> str:
    return _text(row).replace("\n", " ")[:240]


if __name__ == "__main__":
    build_c076_paired_source_expansion(C076SourceExpansionConfig())
