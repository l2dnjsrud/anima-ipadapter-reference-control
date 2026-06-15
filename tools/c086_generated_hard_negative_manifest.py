from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from PIL import Image

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.siglip_auto_caption_types import JsonObject, JsonValue  # noqa: E402


DEFAULT_C085_MANIFEST: Final = ROOT / "training/manifests/c085_anchored_full_adapter_20260613.jsonl"
DEFAULT_C085_ROOT: Final = ROOT / ".tmp/c085_anchored_full_adapter_root"
DEFAULT_C085_GATE: Final = ROOT / "eval/qwenvl_c085_anchored_full_adapter_gate_20260613"
DEFAULT_OUTPUT_ROOT: Final = ROOT / ".tmp/c086_generated_hard_negative_root"
DEFAULT_OUTPUT_MANIFEST: Final = ROOT / "training/manifests/c086_qwenvl_generated_hard_negative_20260613.jsonl"
DEFAULT_OUTPUT_SUMMARY: Final = ROOT / "training/manifests/c086_qwenvl_generated_hard_negative_20260613.summary.json"
NEGATIVE_VARIANT: Final = "c085_anchored_full_adapter_w14"
NEGATIVE_PREFIX: Final = "external/c086_generated_hard_negatives"


@dataclass(frozen=True, slots=True)
class C086ManifestError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class C086ManifestConfig:
    c085_manifest_path: Path = DEFAULT_C085_MANIFEST
    c085_image_root: Path = DEFAULT_C085_ROOT
    c085_gate_dir: Path = DEFAULT_C085_GATE
    c085_gate_summary_path: Path = DEFAULT_C085_GATE / "summary.json"
    c085_crop_summary_path: Path | None = None
    output_image_root: Path = DEFAULT_OUTPUT_ROOT
    output_manifest_path: Path = DEFAULT_OUTPUT_MANIFEST
    output_summary_path: Path = DEFAULT_OUTPUT_SUMMARY


@dataclass(frozen=True, slots=True)
class C086ManifestSummary:
    output_manifest_path: str
    scratch_image_root: str
    source_manifest_path: str
    c085_gate_dir: str
    train_negative_rows: int
    crop_negative_rows: int
    generated_negative_rows: int
    total_rows: int
    heldout_rows_used: int
    materialized_negative_images: int
    decision: str


def build_c086_generated_hard_negative_manifest(
    config: C086ManifestConfig = C086ManifestConfig(),
) -> C086ManifestSummary:
    c085_rows = _jsonl(config.c085_manifest_path)
    crop_rows = _crop_rows_by_ref(c085_rows)
    samples = _combined_samples(config)
    rows: list[dict[str, str]] = []
    train_count = 0
    crop_count = 0
    materialized_negative_ids: set[str] = set()

    for sample in samples:
        label = str(sample["label"])
        split = str(sample["split"])
        if split == "heldout" or label.startswith("heldout"):
            continue
        negative_id = f"{NEGATIVE_PREFIX}/{label}_{NEGATIVE_VARIANT}"
        _materialize_negative(label, negative_id, config, materialized_negative_ids)
        if split == "train":
            row = _train_row(sample, negative_id)
            train_count += 1
        elif split == "direct_green":
            row = _crop_row(sample, crop_rows, negative_id)
            crop_count += 1
        else:
            continue
        _materialize_positive(row["ref_id"], config)
        _materialize_positive(row["tgt_id"], config)
        _materialize_caption(row["tgt_id"], config, row["prompt"])
        rows.append(row)

    if not rows:
        raise C086ManifestError("c086 manifest has no usable rows")
    _write_jsonl(config.output_manifest_path, rows)
    summary = C086ManifestSummary(
        output_manifest_path=str(config.output_manifest_path),
        scratch_image_root=str(config.output_image_root),
        source_manifest_path=str(config.c085_manifest_path),
        c085_gate_dir=str(config.c085_gate_dir),
        train_negative_rows=train_count,
        crop_negative_rows=crop_count,
        generated_negative_rows=len(rows),
        total_rows=len(rows),
        heldout_rows_used=0,
        materialized_negative_images=len(materialized_negative_ids),
        decision="ready_for_c086_hard_negative_training",
    )
    config.output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _jsonl(path: Path) -> tuple[JsonObject, ...]:
    with path.open(encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle if line.strip())


def _samples(path: Path) -> tuple[JsonObject, ...]:
    data: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise C086ManifestError(f"invalid c085 gate summary: {path}")
    return tuple(item for item in data["samples"] if isinstance(item, dict))


def _combined_samples(config: C086ManifestConfig) -> tuple[JsonObject, ...]:
    samples = list(_samples(config.c085_gate_summary_path))
    crop_summary_path = config.c085_crop_summary_path
    if crop_summary_path is None:
        crop_summary_path = config.c085_gate_dir / "crop_pair_summary.json"
    if crop_summary_path.is_file():
        existing_labels = {str(sample["label"]) for sample in samples}
        for sample in _samples(crop_summary_path):
            if str(sample["label"]) not in existing_labels:
                samples.append(sample)
    return tuple(samples)


def _crop_rows_by_ref(rows: tuple[JsonObject, ...]) -> dict[str, JsonObject]:
    mapped: dict[str, JsonObject] = {}
    for row in rows:
        ref_id = str(row["ref_id"])
        if not ref_id.startswith("external/c084_sheet_crop_pairs/"):
            continue
        mapped.setdefault(ref_id, row)
    return mapped


def _train_row(sample: JsonObject, negative_id: str) -> dict[str, str]:
    ref_id = str(sample["ref_id"])
    return {
        "ref_id": ref_id,
        "tgt_id": ref_id,
        "neg_id": negative_id,
        "prompt": str(sample["prompt"]),
    }


def _crop_row(
    sample: JsonObject,
    crop_rows: dict[str, JsonObject],
    negative_id: str,
) -> dict[str, str]:
    ref_id = str(sample["ref_id"])
    if ref_id not in crop_rows:
        raise C086ManifestError(f"missing c085 crop row for {ref_id}")
    source = crop_rows[ref_id]
    return {
        "ref_id": ref_id,
        "tgt_id": str(source["tgt_id"]),
        "neg_id": negative_id,
        "prompt": str(source["prompt"]),
    }


def _materialize_negative(
    label: str,
    negative_id: str,
    config: C086ManifestConfig,
    materialized: set[str],
) -> None:
    source = config.c085_gate_dir / f"{label}_{NEGATIVE_VARIANT}.png"
    if not source.is_file():
        raise C086ManifestError(f"missing generated negative: {source}")
    destination = config.output_image_root / f"{negative_id}.jpg"
    if negative_id in materialized and destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, quality=95)
    materialized.add(negative_id)


def _materialize_positive(image_id: str, config: C086ManifestConfig) -> None:
    destination = config.output_image_root / f"{image_id}.jpg"
    if destination.exists() or destination.is_symlink():
        return
    source = config.c085_image_root / f"{image_id}.jpg"
    if not source.is_file():
        raise C086ManifestError(f"missing positive image: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source, destination)


def _materialize_caption(image_id: str, config: C086ManifestConfig, fallback_prompt: str) -> None:
    destination = config.output_image_root / f"{image_id}.txt"
    if destination.exists() or destination.is_symlink():
        return
    source = config.c085_image_root / f"{image_id}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        os.symlink(source, destination)
    else:
        destination.write_text(fallback_prompt + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    config = C086ManifestConfig(
        output_manifest_path=args.output_manifest,
        output_summary_path=args.output_summary,
        output_image_root=args.output_root,
    )
    summary = build_c086_generated_hard_negative_manifest(config)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
