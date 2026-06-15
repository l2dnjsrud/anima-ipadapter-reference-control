from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

from tools.c071_seed_package import LABEL_SCHEMA
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
from tools.c077_acquisition_report import acquisition_report, summarize_acquisition, write_decision_report
from tools.c077_hf_sample_sources import C077_SAMPLE_SOURCES, C077TreeProbe, fetch_c077_tree_probe
from tools.siglip_auto_caption_types import JsonObject

OUT_DIR: Final = Path("eval/c077_direct_green_target_positive_acquisition_20260612")
SCRATCH: Final = Path(".tmp/c077_direct_green_target_positive_acquisition")
C074_LABELS: Final = Path("eval/c074_tag_backed_direct_green_source_acquisition_20260612/reviewed_external_labels.jsonl")
HELDOUT: Final = Path("training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl")
MAX_BYTES: Final = 4_194_304

type FetchTreeProbe = Callable[[float], tuple[C077TreeProbe, ...]]


@dataclass(frozen=True, slots=True)
class C077AcquisitionConfig:
    out_dir: Path = OUT_DIR
    scratch_dir: Path = SCRATCH
    c074_labels_path: Path = C074_LABELS
    heldout_manifest_path: Path = HELDOUT
    timeout_seconds: float = 12.0
    max_image_bytes: int = MAX_BYTES
    max_new_downloads: int = 36
    max_per_source: int = 8
    labels_path: Path | None = None


def build_c077_target_positive_acquisition(
    config: C077AcquisitionConfig,
    *,
    probes: tuple[C077TreeProbe, ...] | None = None,
    fetch_image: FetchImage | None = None,
) -> JsonObject:
    selected_fetch = fetch_image_to_path if fetch_image is None else fetch_image
    source_probes = probes if probes is not None else _fetch_probes(config.timeout_seconds)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = config.scratch_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    heldout_ids = read_ids(config.heldout_manifest_path)
    c074_rows = read_jsonl(config.c074_labels_path) if config.c074_labels_path.is_file() else ()
    seeds = _seed_candidates(c074_rows, heldout_ids=heldout_ids)
    candidates = _tree_candidates(source_probes, heldout_ids=heldout_ids, max_per_source=config.max_per_source)
    downloads = _materialize(seeds, candidates, images_dir, config, selected_fetch)
    labels_path = config.out_dir / "manual_visual_labels.csv" if config.labels_path is None else config.labels_path
    labels = read_label_map(labels_path) if labels_path.is_file() else {}
    reviewed = _review_downloads(downloads, labels)
    source_rows = _source_rows(source_probes, c074_rows, candidates)
    _write_outputs(config, source_rows, seeds, candidates, downloads, reviewed)
    sheet_path = config.scratch_dir / "contact_sheet.jpg"
    sheet_written = write_sheet(downloads, sheet_path)
    summary = summarize_acquisition(source_rows, downloads, reviewed, sheet_path=sheet_path, sheet_written=sheet_written)
    (config.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (config.out_dir / "report.md").write_text(acquisition_report(summary), encoding="utf-8")
    write_decision_report(config.out_dir, summary)
    return summary


def _fetch_probes(timeout_seconds: float) -> tuple[C077TreeProbe, ...]:
    return tuple(fetch_c077_tree_probe(source, timeout_seconds=timeout_seconds) for source in C077_SAMPLE_SOURCES)


def _seed_candidates(rows: tuple[JsonObject, ...], *, heldout_ids: set[str]) -> tuple[JsonObject, ...]:
    seeds: list[JsonObject] = []
    for row in rows:
        image_id = str(row.get("image_id", ""))
        if row.get("manual_label") == "target_positive" and image_id not in heldout_ids:
            seeds.append(dict(row) | {"candidate_id": f"c077_seed_{row['candidate_id']}", "review_source": "prior_c074_manual"})
    return tuple(seeds)


def _tree_candidates(probes: tuple[C077TreeProbe, ...], *, heldout_ids: set[str], max_per_source: int) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for probe in probes:
        repo_slug = _slug(probe.repo)
        for path in probe.sample_paths[:max_per_source]:
            image_id = f"external/{probe.repo}/{path.removeprefix('samples/').removesuffix('.png')}"
            if image_id in heldout_ids:
                continue
            rows.append(
                {
                    "candidate_id": f"c077_{repo_slug}_{path.removeprefix('samples/').removesuffix('.png').replace('/', '_').replace('-', '_')}",
                    "image_id": image_id,
                    "image_path": f"https://huggingface.co/datasets/{probe.repo}/resolve/main/{path}",
                    "source_bucket": "cyberharem_sample_asset_visual_probe",
                    "suggested_label": "useful_proxy_non_human",
                    "source_experiments": ["c077_direct_green_target_positive_acquisition"],
                    "source_labels": ["sample_asset_needs_visual_confirmation"],
                    "source_buckets": [probe.repo],
                    "review_notes": [probe.source_note],
                    "rank": len(rows) + 1,
                    "bucket_score": _source_score(probe.repo),
                    "heldout_excluded": False,
                    "external_source_url": probe.official_url,
                    "external_license_note": probe.license_note,
                    "review_source": "c077_visual_probe_needs_manual_confirmation",
                }
            )
    return tuple(rows)


def _materialize(
    seeds: tuple[JsonObject, ...],
    candidates: tuple[JsonObject, ...],
    images_dir: Path,
    config: C077AcquisitionConfig,
    fetch_image_fn: FetchImage,
) -> tuple[JsonObject, ...]:
    rows = [_copy_seed(row, images_dir) for row in seeds]
    for row in candidates[: config.max_new_downloads]:
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
        if row["download_status"] not in {"downloaded", "copied_prior_c074"}:
            continue
        label_row = labels.get(str(row["candidate_id"]), {})
        default_label = "target_positive" if row.get("review_source") == "prior_c074_manual" else "useful_proxy_non_human"
        manual_label = str(label_row.get("manual_label") or default_label)
        if manual_label not in LABEL_SCHEMA:
            raise ValueError(f"unknown c077 manual label: {manual_label}")
        note = str(label_row.get("manual_note") or _default_note(str(row.get("review_source", ""))))
        reviewed.append(dict(row) | {"manual_label": manual_label, "manual_note": note, "visual_confirmation": manual_label == "target_positive"})
    return tuple(reviewed)


def _write_outputs(
    config: C077AcquisitionConfig,
    source_rows: tuple[JsonObject, ...],
    seeds: tuple[JsonObject, ...],
    candidates: tuple[JsonObject, ...],
    downloads: tuple[JsonObject, ...],
    reviewed: tuple[JsonObject, ...],
) -> None:
    write_jsonl(config.out_dir / "source_manifest.jsonl", source_rows)
    write_jsonl(config.out_dir / "candidate_manifest.jsonl", seeds + candidates)
    write_jsonl(config.out_dir / "download_manifest.jsonl", downloads)
    write_jsonl(config.out_dir / "reviewed_external_labels.jsonl", reviewed)
    write_template(config.out_dir / "visual_label_template.csv", downloads)


def _source_rows(probes: tuple[C077TreeProbe, ...], c074_rows: tuple[JsonObject, ...], candidates: tuple[JsonObject, ...]) -> tuple[JsonObject, ...]:
    rows = [
        {
            "repo": "c074_confirmed_seed/CyberHarem/neeko_leagueoflegends",
            "official_url": "eval/c074_tag_backed_direct_green_source_acquisition_20260612",
            "access_status": "local_prior_review",
            "license_note": "mit/NFA/source-rights caution carried from c074",
            "path_status": "prior_reviewed_rows",
            "inspected_path_count": len(c074_rows),
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
                "path_status": probe.path_status,
                "inspected_path_count": probe.inspected_path_count,
                "potential_candidate_count": sum(1 for row in candidates if probe.repo in row.get("source_buckets", [])),
                "probe_note": probe.source_note,
            }
        )
    return tuple(rows)


def _slug(value: str) -> str:
    return value.lower().replace("/", "_").replace("-", "_")


def _source_score(repo: str) -> float:
    return 3.0 if "green" in repo.lower() else 1.0


def _default_note(review_source: str) -> str:
    if review_source == "prior_c074_manual":
        return "prior c074 target_positive carried forward; not new c077 evidence"
    return "new c077 sample asset is conservatively non-target until visual review confirms direct-green non-human traits"


if __name__ == "__main__":
    build_c077_target_positive_acquisition(C077AcquisitionConfig())
