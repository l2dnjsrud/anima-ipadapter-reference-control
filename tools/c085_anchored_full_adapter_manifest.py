from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]
COLOR_ROOT: Final = Path("/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best")
DEFAULT_C084_ROOT: Final = ROOT / ".tmp/c084_balanced_crop_pairs_root"
DEFAULT_C085_ROOT: Final = ROOT / ".tmp/c085_anchored_full_adapter_root"
DEFAULT_C084_MANIFEST: Final = ROOT / "training/manifests/c084_balanced_crop_pairs_20260613.jsonl"
DEFAULT_C060_MANIFEST: Final = ROOT / "training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.jsonl"
DEFAULT_HELDOUT_SUMMARY: Final = ROOT / "training/manifests/local_color_single_character_clean32_20260611.summary.json"
DEFAULT_OUTPUT_MANIFEST: Final = ROOT / "training/manifests/c085_anchored_full_adapter_20260613.jsonl"
DEFAULT_OUTPUT_SUMMARY: Final = ROOT / "training/manifests/c085_anchored_full_adapter_20260613.summary.json"


@dataclass(frozen=True, slots=True)
class C085ManifestSummary:
    output_manifest_path: str
    scratch_image_root: str
    c084_crop_rows: int
    clean_anchor_rows: int
    c052_positive_anchor_rows: int
    failure_anchor_rows: int
    total_rows: int
    heldout_rows_used: int
    direct_self_pair_rows: int
    materialized_image_count: int
    source_manifests: tuple[str, ...]
    decision: str


def build_c085_manifest(
    *,
    c084_manifest: Path = DEFAULT_C084_MANIFEST,
    c060_manifest: Path = DEFAULT_C060_MANIFEST,
    c084_root: Path = DEFAULT_C084_ROOT,
    color_root: Path = COLOR_ROOT,
    output_root: Path = DEFAULT_C085_ROOT,
    output_manifest: Path = DEFAULT_OUTPUT_MANIFEST,
    output_summary: Path = DEFAULT_OUTPUT_SUMMARY,
    heldout_summary: Path = DEFAULT_HELDOUT_SUMMARY,
) -> C085ManifestSummary:
    heldout_ids = _heldout_ids(heldout_summary)
    c084_rows = _jsonl(c084_manifest)
    c060_rows = _jsonl(c060_manifest)
    selected = [
        *[(row, "c084_crop") for row in c084_rows[:80]],
        *[(row, "clean_anchor") for row in c060_rows[:32]],
        *[(row, "c052_positive_anchor") for row in c060_rows[32:48]],
        *[(row, "failure_anchor") for row in c060_rows[90:122]],
    ]
    leaked = [
        row
        for row, _source in selected
        if str(row["ref_id"]) in heldout_ids or str(row["tgt_id"]) in heldout_ids
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    materialized: set[str] = set()
    rows: list[dict[str, str]] = []
    counts = {
        "c084_crop": 0,
        "clean_anchor": 0,
        "c052_positive_anchor": 0,
        "failure_anchor": 0,
    }
    roots = (c084_root, color_root)
    for row, source in selected:
        normalized = {
            "ref_id": str(row["ref_id"]),
            "tgt_id": str(row["tgt_id"]),
            "prompt": str(row["prompt"]),
        }
        _materialize_image(normalized["ref_id"], roots, output_root, materialized)
        _materialize_image(normalized["tgt_id"], roots, output_root, materialized)
        _materialize_caption(normalized["tgt_id"], roots, output_root, normalized["prompt"])
        rows.append(normalized)
        counts[source] += 1
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = C085ManifestSummary(
        output_manifest_path=str(output_manifest),
        scratch_image_root=str(output_root),
        c084_crop_rows=counts["c084_crop"],
        clean_anchor_rows=counts["clean_anchor"],
        c052_positive_anchor_rows=counts["c052_positive_anchor"],
        failure_anchor_rows=counts["failure_anchor"],
        total_rows=len(rows),
        heldout_rows_used=len(leaked),
        direct_self_pair_rows=sum(1 for row in rows if row["ref_id"] == row["tgt_id"]),
        materialized_image_count=len(materialized),
        source_manifests=(str(c084_manifest), str(c060_manifest)),
        decision="ready_for_c085_full_adapter_training" if not leaked else "blocked_heldout_leakage",
    )
    output_summary.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if leaked:
        raise SystemExit("heldout leakage detected in c085 manifest")
    return summary


def _jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _heldout_ids(summary_path: Path) -> set[str]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return {str(item) for item in data.get("heldout_ids", [])}


def _materialize_image(
    image_id: str,
    roots: tuple[Path, ...],
    output_root: Path,
    materialized: set[str],
) -> None:
    destination = output_root / f"{image_id}.jpg"
    if image_id in materialized and destination.exists():
        return
    source = _resolve_image(image_id, roots)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"destination points elsewhere: {destination}")
    else:
        os.symlink(source, destination)
    materialized.add(image_id)


def _materialize_caption(
    image_id: str,
    roots: tuple[Path, ...],
    output_root: Path,
    fallback_prompt: str,
) -> None:
    destination = output_root / f"{image_id}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = _resolve_caption(image_id, roots)
    if source is None:
        if not destination.exists():
            destination.write_text(fallback_prompt + "\n", encoding="utf-8")
        return
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"caption destination points elsewhere: {destination}")
    else:
        os.symlink(source, destination)


def _resolve_image(image_id: str, roots: tuple[Path, ...]) -> Path:
    for root in roots:
        candidate = root / f"{image_id}.jpg"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing image for {image_id}")


def _resolve_caption(image_id: str, roots: tuple[Path, ...]) -> Path | None:
    for root in roots:
        candidate = root / f"{image_id}.txt"
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_C085_ROOT)
    args = parser.parse_args()
    summary = build_c085_manifest(
        output_manifest=args.output_manifest,
        output_summary=args.output_summary,
        output_root=args.output_root,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
