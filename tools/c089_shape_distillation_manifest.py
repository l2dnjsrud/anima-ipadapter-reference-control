from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST: Final = (
    ROOT / "training/manifests/c087_expanded_crop_pairs_20260613.jsonl"
)
DEFAULT_SOURCE_ROOT: Final = ROOT / ".tmp/c087_expanded_crop_pairs_root"
DEFAULT_OUTPUT_ROOT: Final = ROOT / ".tmp/c089_shape_silhouette_distillation_root"
DEFAULT_OUTPUT_MANIFEST: Final = (
    ROOT / "training/manifests/c089_shape_silhouette_distillation_20260613.jsonl"
)
DEFAULT_OUTPUT_SUMMARY: Final = (
    ROOT / "training/manifests/c089_shape_silhouette_distillation_20260613.summary.json"
)
DEFAULT_HELDOUT_SUMMARY: Final = (
    ROOT / "training/manifests/local_color_single_character_clean32_20260611.summary.json"
)
TEACHER_SOURCE_LABELS: Final = (
    "pe_teacher_prediction",
    "pe_token_retrieval",
    "edge_projection_silhouette_probe",
)


@dataclass(frozen=True, slots=True)
class C089ManifestSummary:
    output_manifest_path: str
    scratch_image_root: str
    source_manifest_path: str
    source_rows: int
    total_rows: int
    heldout_rows_used: int
    max_rows_per_group: int
    selected_group_counts: dict[str, int]
    teacher_source_labels: list[str]
    materialized_image_count: int
    decision: str


def build_c089_manifest(
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    output_manifest: Path = DEFAULT_OUTPUT_MANIFEST,
    output_summary: Path = DEFAULT_OUTPUT_SUMMARY,
    heldout_summary: Path = DEFAULT_HELDOUT_SUMMARY,
    max_rows_per_group: int = 16,
) -> C089ManifestSummary:
    if max_rows_per_group < 1:
        raise ValueError("max_rows_per_group must be >= 1")
    rows = _jsonl(source_manifest)
    heldout_ids = _heldout_ids(heldout_summary)
    selected = _balanced_rows(rows, max_rows_per_group=max_rows_per_group)
    leaked = [
        row
        for row in selected
        if str(row["ref_id"]) in heldout_ids or str(row["tgt_id"]) in heldout_ids
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    materialized: set[str] = set()
    normalized_rows: list[dict[str, str]] = []
    group_counts: dict[str, int] = defaultdict(int)
    for row in selected:
        ref_id = str(row["ref_id"])
        tgt_id = str(row["tgt_id"])
        prompt = str(row["prompt"])
        _materialize_image(ref_id, source_root, output_root, materialized)
        _materialize_image(tgt_id, source_root, output_root, materialized)
        _materialize_caption(tgt_id, source_root, output_root, prompt)
        normalized_rows.append({"ref_id": ref_id, "tgt_id": tgt_id, "prompt": prompt})
        group_counts[_shape_group(ref_id)] += 1
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in normalized_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = C089ManifestSummary(
        output_manifest_path=str(output_manifest),
        scratch_image_root=str(output_root),
        source_manifest_path=str(source_manifest),
        source_rows=len(rows),
        total_rows=len(normalized_rows),
        heldout_rows_used=len(leaked),
        max_rows_per_group=max_rows_per_group,
        selected_group_counts=dict(sorted(group_counts.items())),
        teacher_source_labels=list(TEACHER_SOURCE_LABELS),
        materialized_image_count=len(materialized),
        decision=(
            "ready_for_c089_siglip_pe_teacher_pilot"
            if not leaked
            else "blocked_heldout_leakage"
        ),
    )
    output_summary.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if leaked:
        raise SystemExit("heldout leakage detected in c089 manifest")
    return summary


def _jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _heldout_ids(summary_path: Path) -> set[str]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return {str(item) for item in data.get("heldout_ids", [])}


def _balanced_rows(
    rows: list[dict[str, object]],
    *,
    max_rows_per_group: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        group = _shape_group(str(row["ref_id"]))
        if counts[group] >= max_rows_per_group:
            continue
        selected.append(row)
        counts[group] += 1
    return selected


def _shape_group(image_id: str) -> str:
    marker = "c083_"
    if marker in image_id:
        tail = image_id.split(marker, 1)[1]
        for suffix in ("_action", "_front", "_profile", "_three_quarter", "_ref", "_tgt"):
            if suffix in tail:
                return tail.split(suffix, 1)[0]
    return image_id.split("/", 1)[0]


def _materialize_image(
    image_id: str,
    source_root: Path,
    output_root: Path,
    materialized: set[str],
) -> None:
    destination = output_root / f"{image_id}.jpg"
    if image_id in materialized and destination.exists():
        return
    source = source_root / f"{image_id}.jpg"
    if not source.is_file():
        raise FileNotFoundError(f"missing image for {image_id}: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _safe_symlink(source, destination)
    materialized.add(image_id)


def _materialize_caption(
    image_id: str,
    source_root: Path,
    output_root: Path,
    fallback_prompt: str,
) -> None:
    destination = output_root / f"{image_id}.txt"
    source = source_root / f"{image_id}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        _safe_symlink(source, destination)
        return
    if not destination.exists():
        destination.write_text(fallback_prompt + "\n", encoding="utf-8")


def _safe_symlink(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"destination points elsewhere: {destination}")
        return
    os.symlink(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--heldout-summary", type=Path, default=DEFAULT_HELDOUT_SUMMARY)
    parser.add_argument("--max-rows-per-group", type=int, default=16)
    args = parser.parse_args()
    summary = build_c089_manifest(
        source_manifest=args.source_manifest,
        source_root=args.source_root,
        output_root=args.output_root,
        output_manifest=args.output_manifest,
        output_summary=args.output_summary,
        heldout_summary=args.heldout_summary,
        max_rows_per_group=args.max_rows_per_group,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
