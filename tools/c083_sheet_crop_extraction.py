from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image, ImageFilter

from tools.c071_seed_package import LABEL_SCHEMA
from tools.c076_source_expansion_io import read_label_map, write_jsonl, write_sheet
from tools.siglip_auto_caption_types import JsonObject, JsonValue

OUT_DIR: Final = Path("eval/c083_sheet_crop_identity_pair_extraction_20260613")
SCRATCH: Final = Path(".tmp/c083_sheet_crop_identity_pair_extraction")
C082_GENERATION_MANIFEST: Final = Path("eval/c082_single_image_pair_acquisition_20260613/generation_manifest.jsonl")
MIN_APPROVED_GROUPS: Final = 4
MIN_APPROVED_PAIRS: Final = 24


@dataclass(frozen=True, slots=True)
class C083Config:
    out_dir: Path = OUT_DIR
    scratch_dir: Path = SCRATCH
    source_manifest_path: Path = C082_GENERATION_MANIFEST
    labels_path: Path | None = None
    max_crops_per_source: int = 8


def extract_c083_crops(config: C083Config) -> JsonObject:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = config.scratch_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    generated = tuple(row for row in _read_jsonl(config.source_manifest_path) if row.get("status") == "generated")
    crops = tuple(candidate for row in generated for candidate in _crop_candidates(row, crop_dir, config.max_crops_per_source))
    write_jsonl(config.out_dir / "crop_candidate_manifest.jsonl", crops)
    _write_visual_template(config.out_dir / "visual_label_template.csv", crops)
    sheet_written = write_sheet(crops, config.out_dir / "contact_sheet.jpg")
    summary = {
        "source": "c083_sheet_crop_identity_pair_extraction",
        "source_generated_rows": len(generated),
        "crop_candidate_rows": len(crops),
        "heldout_rows_used": 0,
        "training_started": False,
        "raw_crop_images_committed": False,
        "contact_sheet_written": sheet_written,
        "decision": "crop_review_required" if crops else "no_crop_candidates",
    }
    _write_json(config.out_dir / "extraction_summary.json", summary)
    (config.out_dir / "report.md").write_text(_extraction_report(summary), encoding="utf-8")
    return summary


def review_c083_crops(config: C083Config) -> JsonObject:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    crops = _read_jsonl(config.out_dir / "crop_candidate_manifest.jsonl")
    labels_path = config.labels_path or config.out_dir / "manual_visual_labels.csv"
    labels = read_label_map(labels_path) if labels_path.is_file() else {}
    reviewed = tuple(_reviewed_crop(row, labels.get(str(row["candidate_id"]), {})) for row in crops)
    write_jsonl(config.out_dir / "reviewed_crop_labels.jsonl", reviewed)
    pairs = _approved_pairs(reviewed)
    write_jsonl(config.out_dir / "approved_pair_manifest.jsonl", pairs)
    summary = _review_summary(reviewed, pairs)
    _write_json(config.out_dir / "summary.json", summary)
    (config.out_dir / "report.md").write_text(_review_report(summary), encoding="utf-8")
    return summary


def _crop_candidates(row: JsonObject, crop_dir: Path, max_crops: int) -> tuple[JsonObject, ...]:
    image_path = Path(str(row["local_image_path"]))
    if not image_path.is_file():
        return ()
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
        boxes = _component_boxes(image, max_crops=max_crops)
        candidates: list[JsonObject] = []
        for index, box in enumerate(boxes, start=1):
            crop_path = crop_dir / f"{row['candidate_id']}_crop{index:02d}.png"
            image.crop(box).save(crop_path)
            width, height = box[2] - box[0], box[3] - box[1]
            candidates.append(
                {
                    "candidate_id": f"c083_{row['candidate_id']}_crop{index:02d}",
                    "source_candidate_id": row["candidate_id"],
                    "group_id": row["group_id"],
                    "source_view_id": row["view_id"],
                    "crop_index": index,
                    "bbox_xyxy": list(box),
                    "crop_width": width,
                    "crop_height": height,
                    "crop_area_ratio": round((width * height) / (image.width * image.height), 6),
                    "local_image_path": str(crop_path),
                    "source_image_path": str(image_path),
                    "download_status": "generated_crop",
                    "review_source": "c082_sheet_crop",
                    "external_license_note": "local_synthetic_generation_crop",
                    "heldout_excluded": True,
                }
            )
    return tuple(candidates)


def _component_boxes(image: Image.Image, *, max_crops: int) -> tuple[tuple[int, int, int, int], ...]:
    mask = _foreground_mask(image)
    components = _connected_components(mask)
    min_area = max(900, (image.width * image.height) // 900)
    boxes: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2, area in components:
        width, height = x2 - x1, y2 - y1
        if area < min_area or width < 32 or height < 48:
            continue
        expanded = _expand_box((x1, y1, x2, y2), image.size, pad=18)
        if _box_area(expanded) / (image.width * image.height) > 0.88:
            continue
        boxes.append(expanded)
    return tuple(sorted(boxes, key=lambda box: (-_box_area(box), box[1], box[0]))[:max_crops])


def _foreground_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    distance_from_white = np.max(255 - rgb, axis=2)
    color_span = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    mask = (distance_from_white > 22) & ((color_span > 8) | (np.mean(rgb, axis=2) < 235))
    dilated = Image.fromarray(mask.astype("uint8") * 255).filter(ImageFilter.MaxFilter(9))
    return np.asarray(dilated) > 0


def _connected_components(mask: np.ndarray) -> tuple[tuple[int, int, int, int, int], ...]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    for start_y in range(height):
        for start_x in range(width):
            if seen[start_y, start_x] or not mask[start_y, start_x]:
                continue
            stack = [(start_x, start_y)]
            seen[start_y, start_x] = True
            min_x = max_x = start_x
            min_y = max_y = start_y
            area = 0
            while stack:
                x, y = stack.pop()
                area += 1
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= next_x < width and 0 <= next_y < height and mask[next_y, next_x] and not seen[next_y, next_x]:
                        seen[next_y, next_x] = True
                        stack.append((next_x, next_y))
            components.append((min_x, min_y, max_x + 1, max_y + 1, area))
    return tuple(components)


def _reviewed_crop(row: JsonObject, label_row: JsonObject) -> JsonObject:
    label = str(label_row.get("manual_label") or "reject_unclear")
    if label not in LABEL_SCHEMA:
        raise ValueError(f"unknown c083 manual label: {label}")
    return dict(row) | {
        "manual_label": label,
        "manual_note": str(label_row.get("manual_note") or "crop requires single-figure visual review"),
        "visual_confirmation": label == "target_positive",
    }


def _approved_pairs(reviewed: tuple[JsonObject, ...]) -> tuple[JsonObject, ...]:
    pairs: list[JsonObject] = []
    for group_id, rows in _eligible_rows_by_group(reviewed).items():
        for ref in rows:
            for tgt in rows:
                if ref["candidate_id"] == tgt["candidate_id"] or ref["source_candidate_id"] == tgt["source_candidate_id"]:
                    continue
                pairs.append(
                    {
                        "group_id": group_id,
                        "ref_id": f"external/c083_sheet_crop_pairs/{ref['candidate_id']}",
                        "tgt_id": f"external/c083_sheet_crop_pairs/{tgt['candidate_id']}",
                        "prompt": "mrcolor_panel_style, same direct-green non-human character identity, clean color manhwa panel",
                    }
                )
    return tuple(pairs)


def _eligible_rows_by_group(reviewed: tuple[JsonObject, ...]) -> dict[str, tuple[JsonObject, ...]]:
    by_group: dict[str, list[JsonObject]] = {}
    for row in reviewed:
        if row.get("manual_label") == "target_positive":
            by_group.setdefault(str(row["group_id"]), []).append(row)
    return {group_id: tuple(rows) for group_id, rows in sorted(by_group.items()) if len({str(row["source_candidate_id"]) for row in rows}) >= 2}


def _review_summary(reviewed: tuple[JsonObject, ...], pairs: tuple[JsonObject, ...]) -> JsonObject:
    groups = _eligible_rows_by_group(reviewed)
    decision = "ready_for_c084_paired_training_manifest" if len(groups) >= MIN_APPROVED_GROUPS and len(pairs) >= MIN_APPROVED_PAIRS else "more_pairs_required"
    return {
        "source": "c083_sheet_crop_identity_pair_extraction",
        "reviewed_rows": len(reviewed),
        "target_positive_rows": sum(1 for row in reviewed if row.get("manual_label") == "target_positive"),
        "approved_group_count": len(groups),
        "approved_pair_rows": len(pairs),
        "direct_self_pair_rows": sum(1 for row in pairs if row["ref_id"] == row["tgt_id"]),
        "heldout_rows_used": 0,
        "training_started": False,
        "raw_crop_images_committed": False,
        "minimum_approved_groups": MIN_APPROVED_GROUPS,
        "minimum_approved_pairs": MIN_APPROVED_PAIRS,
        "decision": decision,
    }


def _write_visual_template(path: Path, rows: tuple[JsonObject, ...]) -> None:
    fields = ("candidate_id", "group_id", "source_candidate_id", "source_view_id", "manual_label", "manual_note", "allowed_labels")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields[:-1]} | {"allowed_labels": "|".join(LABEL_SCHEMA)})


def _extraction_report(summary: JsonObject) -> str:
    return "\n".join(["# c083 sheet crop identity pair extraction", "", f"- decision: `{summary['decision']}`", f"- source_generated_rows: `{summary['source_generated_rows']}`", f"- crop_candidate_rows: `{summary['crop_candidate_rows']}`", "- heldout_rows_used: `0`", "- training_started: `false`", ""])


def _review_report(summary: JsonObject) -> str:
    return "\n".join(["# c083 sheet crop identity pair extraction", "", f"- decision: `{summary['decision']}`", f"- reviewed_rows: `{summary['reviewed_rows']}`", f"- target_positive_rows: `{summary['target_positive_rows']}`", f"- approved_group_count: `{summary['approved_group_count']}`", f"- approved_pair_rows: `{summary['approved_pair_rows']}`", f"- direct_self_pair_rows: `{summary['direct_self_pair_rows']}`", ""])


def _expand_box(box: tuple[int, int, int, int], size: tuple[int, int], *, pad: int) -> tuple[int, int, int, int]:
    width, height = size
    return (max(0, box[0] - pad), max(0, box[1] - pad), min(width, box[2] + pad), min(height, box[3] + pad))


def _box_area(box: tuple[int, int, int, int]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)


def _write_json(path: Path, payload: JsonObject) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    extract_c083_crops(C083Config())
    review_c083_crops(C083Config())
