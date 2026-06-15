from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tools.c071_seed_package import LABEL_SCHEMA
from tools.c072_source_probe import SOURCES, DatasetProbe, fetch_probe
from tools.siglip_auto_caption_types import JsonObject, JsonValue

MINIMUM_TARGET_POSITIVES: Final = 4
DEFAULT_OUT_DIR: Final = Path("eval/c072_external_direct_green_source_discovery_20260612")
DEFAULT_HELDOUT: Final = Path("training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl")


@dataclass(frozen=True, slots=True)
class C072DiscoveryConfig:
    heldout_manifest_path: Path = DEFAULT_HELDOUT
    out_dir: Path = DEFAULT_OUT_DIR
    row_limit: int = 100
    timeout_seconds: float = 12.0

DIRECT_GREEN_TERMS: Final = (
    "green skin",
    "glowing green",
    "green body",
    "green fur",
    "green face",
    "green creature",
    "colored skin",
)
NON_HUMAN_TERMS: Final = (
    "anthro",
    "anthropomorphic",
    "furry",
    "creature",
    "monster",
    "alien",
    "robot",
    "demon",
    "fang",
    "animal ears",
    "tail",
    "wolf",
    "fox",
    "dragon",
    "+anima",
)


def build_c072_external_source_discovery(
    config: C072DiscoveryConfig,
    *,
    probes: tuple[DatasetProbe, ...] | None = None,
) -> JsonObject:
    source_probes = probes if probes is not None else tuple(
        fetch_probe(source, row_limit=config.row_limit, timeout_seconds=config.timeout_seconds) for source in SOURCES
    )
    heldout_ids = _read_ids(config.heldout_manifest_path)
    source_rows = tuple(_source_manifest_row(probe) for probe in source_probes)
    candidates = _candidate_rows(source_probes, heldout_ids=heldout_ids)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(config.out_dir / "source_manifest.jsonl", source_rows)
    if candidates:
        _write_jsonl(config.out_dir / "external_candidates.jsonl", candidates)
        _write_candidate_csv(config.out_dir / "external_candidate_template.csv", candidates)
    summary = _summary(source_rows, candidates, heldout_ids=heldout_ids)
    (config.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (config.out_dir / "report.md").write_text(_report(summary, source_rows), encoding="utf-8")
    return summary

def _source_manifest_row(probe: DatasetProbe) -> JsonObject:
    candidate_count = len(_probe_candidates(probe))
    return {
        "repo": probe.repo,
        "official_url": probe.official_url,
        "access_status": probe.access_status,
        "license_note": probe.license_note,
        "metadata_probe_status": probe.metadata_probe_status,
        "features": list(probe.features),
        "inspected_row_count": probe.inspected_row_count,
        "potential_candidate_count": candidate_count,
        "potential_direct_green_positive_note": _potential_note(candidate_count),
        "probe_note": probe.probe_note,
    }


def _candidate_rows(probes: tuple[DatasetProbe, ...], *, heldout_ids: set[str]) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for probe in probes:
        for row_index, row in _probe_candidates(probe):
            image_id = f"external/{probe.repo}/{row_index}"
            if image_id in heldout_ids:
                continue
            rows.append(_candidate_row(probe, row, row_index, image_id))
    rows.sort(key=lambda row: float(row["bucket_score"]), reverse=True)
    return tuple(rows[:12])


def _probe_candidates(probe: DatasetProbe) -> tuple[tuple[int, JsonObject], ...]:
    candidates: list[tuple[int, JsonObject]] = []
    for row_index, row in enumerate(probe.rows):
        if _score(row) >= 4 and _image_url(row):
            candidates.append((row_index, row))
    return tuple(candidates)


def _candidate_row(probe: DatasetProbe, row: JsonObject, row_index: int, image_id: str) -> JsonObject:
    score = float(_score(row))
    return {
        "candidate_id": f"c072_{len(image_id):03d}_{row_index:04d}",
        "image_id": image_id,
        "image_path": _image_url(row),
        "source_bucket": "external_metadata_potential_direct_green_non_human",
        "suggested_label": "useful_proxy_non_human",
        "source_experiments": ["c072_external_source_discovery"],
        "source_labels": ["metadata_potential_direct_green_non_human"],
        "source_buckets": [probe.repo],
        "review_notes": [_text_excerpt(row)],
        "rank": row_index + 1,
        "bucket_score": score,
        "green_ratio": 0.0,
        "central_green_ratio": 0.0,
        "red_ratio": 0.0,
        "heldout_excluded": False,
        "path_exists": False,
        "external_source_url": probe.official_url,
        "external_license_note": probe.license_note,
    }


def _score(row: JsonObject) -> int:
    text = _text_blob(row)
    green_score = sum(3 for term in DIRECT_GREEN_TERMS if term in text)
    non_human_score = sum(2 for term in NON_HUMAN_TERMS if term in text)
    return green_score + non_human_score


def _text_blob(value: JsonValue) -> str:
    match value:
        case dict():
            return " ".join(_text_blob(child) for key, child in value.items() if key not in {"image", "image_url", "jpg", "url"}).lower()
        case list():
            return " ".join(_text_blob(child) for child in value).lower()
        case str():
            return value.lower()
        case None | bool() | int() | float():
            return str(value).lower()


def _image_url(row: JsonObject) -> str:
    for key in ("image_url", "image", "jpg"):
        value = row.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value
        if isinstance(value, dict):
            src = value.get("src")
            if isinstance(src, str) and src.startswith("http"):
                return src
    return ""


def _text_excerpt(row: JsonObject) -> str:
    return _text_blob(row).replace("\n", " ")[:280]


def _summary(source_rows: tuple[JsonObject, ...], candidates: tuple[JsonObject, ...], *, heldout_ids: set[str]) -> JsonObject:
    candidate_count = len({str(row["image_id"]) for row in candidates})
    package_status = "metadata_only_manual_review_required" if candidate_count >= MINIMUM_TARGET_POSITIVES else "no_safe_external_candidate_package"
    return {
        "source": "c072_external_direct_green_source_discovery",
        "inspected_source_count": len(source_rows),
        "large_downloads_performed": False,
        "heldout_ids_count": len(heldout_ids),
        "heldout_rows_used": sum(1 for row in candidates if bool(row["heldout_excluded"])),
        "source_candidate_counts": {str(row["repo"]): int(row["potential_candidate_count"]) for row in source_rows},
        "candidate_package_status": package_status,
        "unique_potential_target_positive_count": candidate_count,
        "target_positive_confirmed_count": 0,
        "minimum_target_positive_required": MINIMUM_TARGET_POSITIVES,
        "label_schema": list(LABEL_SCHEMA),
        "decision": "external_candidates_found_manual_confirmation_required" if candidate_count >= MINIMUM_TARGET_POSITIVES else "external_manual_data_required",
    }


def _report(summary: JsonObject, source_rows: tuple[JsonObject, ...]) -> str:
    lines = [
        "# c072 external direct-green source discovery",
        "",
        f"- decision: `{summary['decision']}`",
        f"- candidate_package_status: `{summary['candidate_package_status']}`",
        f"- unique_potential_target_positive_count: {summary['unique_potential_target_positive_count']}",
        f"- target_positive_confirmed_count: {summary['target_positive_confirmed_count']}",
        f"- large_downloads_performed: {str(summary['large_downloads_performed']).lower()}",
        "",
        "## Sources",
    ]
    for row in source_rows:
        lines.append(
            f"- `{row['repo']}`: license `{row['license_note']}`, metadata `{row['metadata_probe_status']}`, "
            f"potential candidates {row['potential_candidate_count']} - {row['official_url']}"
        )
    lines.extend(
        [
            "",
            "## Next decision",
            "Metadata-only candidates are not confirmed target positives. Use the template for manual review or run a small visual download/review pass before encoder training.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_candidate_csv(path: Path, candidates: tuple[JsonObject, ...]) -> None:
    fieldnames = ("candidate_id", "image_id", "image_path", "suggested_label", "manual_label", "allowed_labels", "external_license_note")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in candidates:
            writer.writerow({key: row.get(key, "") for key in fieldnames} | {"allowed_labels": "|".join(LABEL_SCHEMA), "manual_label": ""})


def _potential_note(candidate_count: int) -> str:
    return "candidate_rows_found_manual_review_required" if candidate_count else "no_candidate_rows_found_in_small_probe"


def _read_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(row["ref_id"]) for row in _read_jsonl(path) if isinstance(row.get("ref_id"), str)}


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

if __name__ == "__main__":
    build_c072_external_source_discovery(C072DiscoveryConfig())
