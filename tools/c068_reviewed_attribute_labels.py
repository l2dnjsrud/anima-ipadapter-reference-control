from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from PIL import Image, ImageDraw, ImageOps

from tools.siglip_auto_caption_types import JsonObject, JsonValue

ReviewLabel = Literal[
    "target_positive",
    "useful_proxy_positive",
    "negative_anchor",
    "false_positive_background_object",
    "false_positive_human_face",
    "false_positive_human_old_face",
    "false_positive_red_eye_human",
]

QUERY_ORDER: Final = (
    "direct_green_non_human_face",
    "red_glowing_eye",
    "side_profile_silhouette",
    "beard_headwear_crop",
    "human_negative",
    "background_object_green",
)


@dataclass(frozen=True, slots=True)
class C068Config:
    c067_topk_path: Path
    heldout_manifest_path: Path
    out_dir: Path
    source_c067_commit: str


@dataclass(frozen=True, slots=True)
class LabelDecision:
    review_label: ReviewLabel
    note: str


def build_reviewed_attribute_labels(config: C068Config) -> JsonObject:
    heldout_ids = _read_ids(config.heldout_manifest_path)
    topk = _read_topk(config.c067_topk_path)
    rows = _review_rows(topk, heldout_ids=heldout_ids)
    _write_jsonl(config.out_dir / "reviewed_attribute_labels.jsonl", rows)
    _write_annotated_sheet(rows, config.out_dir / "annotated_review_sheet.jpg")
    summary = _summary(rows, source_c067_commit=config.source_c067_commit)
    _write_summary_and_report(config.out_dir, summary)
    return summary


def _review_rows(topk: dict[str, list[JsonObject]], *, heldout_ids: set[str]) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for query_key in QUERY_ORDER:
        for rank, raw in enumerate(topk.get(query_key, []), start=1):
            image_id = str(raw["image_id"])
            if image_id in heldout_ids:
                continue
            decision = _label_for(query_key, rank)
            rows.append(
                {
                    "query_key": query_key,
                    "rank": rank,
                    "image_id": image_id,
                    "image_path": str(raw["image_path"]),
                    "score": float(raw["score"]),
                    "source_bucket": str(raw.get("source_bucket", "")),
                    "source_manifest": str(raw.get("source_manifest", "")),
                    "review_label": decision.review_label,
                    "review_note": decision.note,
                    "heldout_excluded": False,
                }
            )
    return tuple(rows)


def _label_for(query_key: str, rank: int) -> LabelDecision:
    match query_key:
        case "direct_green_non_human_face":
            return _direct_green_label(rank)
        case "red_glowing_eye":
            if rank == 1:
                return LabelDecision("target_positive", "visible red glowing eye proxy")
            return LabelDecision("false_positive_human_face", "not a clean red-eye target")
        case "side_profile_silhouette":
            if rank in {1, 2, 3, 4, 5, 7}:
                return LabelDecision("useful_proxy_positive", "useful side-profile face")
            return LabelDecision("false_positive_human_face", "weak or cropped profile cue")
        case "beard_headwear_crop":
            return LabelDecision("useful_proxy_positive", "useful beard/headwear/crop proxy")
        case "human_negative":
            return LabelDecision("negative_anchor", "ordinary human face negative anchor")
        case "background_object_green":
            return LabelDecision(
                "false_positive_background_object",
                "green object/background guard",
            )
        case _:
            return LabelDecision("false_positive_human_face", "unrecognized query")


def _direct_green_label(rank: int) -> LabelDecision:
    if rank == 3:
        return LabelDecision(
            "false_positive_red_eye_human",
            "red-eye human/monk proxy, not green non-human face",
        )
    if rank in {4, 5, 8}:
        return LabelDecision(
            "false_positive_background_object",
            "green object/background, not character skin/species",
        )
    return LabelDecision(
        "false_positive_human_old_face",
        "human old face/headwear/shadow, not green non-human",
    )


def _summary(rows: tuple[JsonObject, ...], *, source_c067_commit: str) -> JsonObject:
    label_counts = _count(rows, "review_label")
    query_counts = _count(rows, "query_key")
    direct_green_target = sum(
        1
        for row in rows
        if row["query_key"] == "direct_green_non_human_face"
        and row["review_label"] == "target_positive"
    )
    return {
        "source_c067_commit": source_c067_commit,
        "reviewed_rows": len(rows),
        "query_count": len(query_counts),
        "heldout_rows_used": sum(1 for row in rows if bool(row["heldout_excluded"])),
        "label_counts": label_counts,
        "query_counts": query_counts,
        "direct_green_target_positive_count": direct_green_target,
        "decision": _decision(direct_green_target),
    }


def _decision(direct_green_target: int) -> str:
    if direct_green_target >= 4:
        return "direct_green_reviewed_seed_ready_for_encoder_training"
    return "direct_green_reviewed_seed_insufficient_new_annotation_required"


def _write_annotated_sheet(rows: tuple[JsonObject, ...], output_path: Path) -> None:
    grouped = {query: [row for row in rows if row["query_key"] == query] for query in QUERY_ORDER}
    cell_w = 224
    cell_h = 266
    row_count = max((len(items) for items in grouped.values()), default=1)
    sheet = Image.new("RGB", (cell_w * len(QUERY_ORDER), cell_h * (row_count + 1)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, query in enumerate(QUERY_ORDER):
        draw.text((col * cell_w + 6, 8), query[:30], fill="black")
        for row_index, row in enumerate(grouped[query]):
            _paste_cell(sheet, draw, row, col * cell_w, (row_index + 1) * cell_h, cell_w, cell_h)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def _paste_cell(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    row: JsonObject,
    x: int,
    y: int,
    cell_w: int,
    cell_h: int,
) -> None:
    with Image.open(Path(str(row["image_path"]))) as image:
        thumb = ImageOps.fit(image.convert("RGB"), (cell_w, cell_h - 66))
    sheet.paste(thumb, (x, y))
    draw.text((x + 4, y + cell_h - 62), f"r{row['rank']} {float(row['score']):.3f}", fill="black")
    draw.text((x + 4, y + cell_h - 44), str(row["review_label"])[:30], fill="black")
    draw.text((x + 4, y + cell_h - 24), Path(str(row["image_id"])).name[:28], fill="black")


def _read_topk(path: Path) -> dict[str, list[JsonObject]]:
    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    topk: dict[str, list[JsonObject]] = {}
    for query_key, values in raw.items():
        if isinstance(query_key, str) and isinstance(values, list):
            topk[query_key] = [value for value in values if isinstance(value, dict)]
    return topk


def _read_ids(path: Path) -> set[str]:
    return {str(row["ref_id"]) for row in _read_jsonl(path)}


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw: JsonValue = json.loads(line)
            if isinstance(raw, dict):
                rows.append(raw)
    return tuple(rows)


def _count(rows: tuple[JsonObject, ...], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_summary_and_report(out_dir: Path, summary: JsonObject) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")


def _render_report(summary: JsonObject) -> str:
    lines = [
        "# c068 Reviewed Attribute Label Seed",
        "",
        f"- Source c067 commit: `{summary['source_c067_commit']}`",
        f"- Reviewed rows: `{summary['reviewed_rows']}`",
        f"- Heldout rows used: `{summary['heldout_rows_used']}`",
        f"- Direct-green target positives: `{summary['direct_green_target_positive_count']}`",
        f"- Decision: `{summary['decision']}`",
        "",
        "## Label Counts",
        "",
    ]
    label_counts = summary["label_counts"]
    if isinstance(label_counts, dict):
        for label, count in sorted(label_counts.items()):
            lines.append(f"- `{label}`: `{count}`")
    lines.extend(
        [
            "",
            "The reviewed direct-green/non-human seed is not sufficient for encoder-side "
            "training unless direct-green target positives are at least 4.",
            "Next decision: do not train encoder-side positives from this seed; collect "
            "new captioned or manually reviewed direct-green/non-human data first.",
            "",
        ]
    )
    return "\n".join(lines)
