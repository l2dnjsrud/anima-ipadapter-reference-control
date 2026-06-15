from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from tools.c088_probe_io import (
    VARIANTS,
    json_object,
    read_manifest_rows,
    write_json,
)
from tools.siglip_auto_caption_types import JsonObject


SUPPORT_UPLIFT: Final = 0.03
SUPPORT_MARGIN: Final = 0.01
EDGE_END: Final = 9216
PROJECTION_END: Final = 9408


@dataclass(frozen=True, slots=True)
class ShapeScoreRow:
    sample: str
    shape_group: str
    variant: str
    edge_cosine: float
    projection_cosine: float
    silhouette_iou: float
    shape_score: float
    no_ip_shape_score: float
    uplift: float
    rank: int


@dataclass(frozen=True, slots=True)
class ShapeCaseDecision:
    sample: str
    shape_group: str
    best_variant: str
    best_uplift: float
    top_margin: float
    decision: str


def score_shape_silhouette_manifest(manifest_path: Path, output_dir: Path) -> JsonObject:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = read_manifest_rows(manifest_path)
    rows: list[ShapeScoreRow] = []
    decisions: list[ShapeCaseDecision] = []
    for case in cases:
        scored = tuple(_score_variant(_shape_feature(Path(str(case["reference_path"]))), case, variant) for variant in VARIANTS)
        ranked = sorted(scored, key=lambda row: row.shape_score, reverse=True)
        no_ip_score = next(row.shape_score for row in ranked if row.variant == "no_ip")
        ranks = {row.variant: index for index, row in enumerate(ranked, start=1)}
        rows.extend(_ranked_rows(scored, ranks, no_ip_score))
        decisions.append(_shape_decision(ranked, no_ip_score))
    result = {
        "manifest_path": str(manifest_path),
        "feature": "edge_projection_silhouette",
        "rows": [dataclasses.asdict(row) for row in sorted(rows, key=lambda row: (row.sample, row.rank))],
        "case_decisions": [dataclasses.asdict(decision) for decision in decisions],
        "summary": _shape_summary(decisions),
    }
    write_json(output_dir / "shape_silhouette_metrics.json", result)
    _write_contact_sheet(cases, output_dir / "contact_sheet.jpg")
    return result


def _ranked_rows(
    rows: tuple[ShapeScoreRow, ...],
    ranks: dict[str, int],
    no_ip_score: float,
) -> tuple[ShapeScoreRow, ...]:
    return tuple(
        ShapeScoreRow(
            sample=row.sample,
            shape_group=row.shape_group,
            variant=row.variant,
            edge_cosine=row.edge_cosine,
            projection_cosine=row.projection_cosine,
            silhouette_iou=row.silhouette_iou,
            shape_score=row.shape_score,
            no_ip_shape_score=no_ip_score,
            uplift=row.shape_score - no_ip_score,
            rank=ranks[row.variant],
        )
        for row in rows
    )


def _score_variant(ref_feature: np.ndarray, case: JsonObject, variant: str) -> ShapeScoreRow:
    candidate_path = Path(str(json_object(case, "candidates")[variant]))
    feature = _shape_feature(candidate_path)
    edge_cosine = _cosine(ref_feature[:EDGE_END], feature[:EDGE_END])
    projection_cosine = _cosine(ref_feature[EDGE_END:PROJECTION_END], feature[EDGE_END:PROJECTION_END])
    silhouette_iou = _iou(ref_feature[PROJECTION_END:], feature[PROJECTION_END:])
    shape_score = 0.45 * edge_cosine + 0.30 * projection_cosine + 0.25 * silhouette_iou
    return ShapeScoreRow(
        sample=str(case["sample"]),
        shape_group=str(case["shape_group"]),
        variant=variant,
        edge_cosine=edge_cosine,
        projection_cosine=projection_cosine,
        silhouette_iou=silhouette_iou,
        shape_score=shape_score,
        no_ip_shape_score=0.0,
        uplift=0.0,
        rank=0,
    )


def _shape_feature(path: Path) -> np.ndarray:
    with Image.open(path) as raw:
        image = ImageOps.fit(raw.convert("RGB"), (96, 96), method=Image.Resampling.LANCZOS)
    gray = ImageOps.grayscale(ImageOps.autocontrast(image))
    edge_array = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    edge_vector = _normalize(edge_array.reshape(-1))
    projection = _normalize(np.concatenate([edge_array.mean(axis=0), edge_array.mean(axis=1)]))
    silhouette = _silhouette(image, edge_array).reshape(-1)
    return np.concatenate([edge_vector, projection, silhouette]).astype(np.float32)


def _silhouette(image: Image.Image, edge_array: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.int16)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    background = np.median(border, axis=0)
    color_distance = np.linalg.norm(rgb - background, axis=2)
    return ((color_distance > 35.0) | (edge_array > 0.12)).astype(np.float32)


def _shape_decision(ranked: list[ShapeScoreRow], no_ip_score: float) -> ShapeCaseDecision:
    best = ranked[0]
    second = ranked[1]
    best_uplift = best.shape_score - no_ip_score
    top_margin = best.shape_score - second.shape_score
    supports = best.variant != "no_ip" and best_uplift >= SUPPORT_UPLIFT and top_margin >= SUPPORT_MARGIN
    return ShapeCaseDecision(
        sample=best.sample,
        shape_group=best.shape_group,
        best_variant=best.variant,
        best_uplift=best_uplift,
        top_margin=top_margin,
        decision="shape_signal_supports_supervised_objective" if supports else "shape_signal_not_enough",
    )


def _shape_summary(decisions: list[ShapeCaseDecision]) -> JsonObject:
    supported = [item for item in decisions if item.decision == "shape_signal_supports_supervised_objective"]
    support_rate = len(supported) / len(decisions)
    return {
        "cases": len(decisions),
        "supported_cases": len(supported),
        "support_rate": support_rate,
        "decision": "shape_silhouette_signal_viable" if support_rate >= 0.5 else "shape_silhouette_signal_not_viable",
    }


def _write_contact_sheet(cases: tuple[JsonObject, ...], output_path: Path) -> None:
    thumb_size = (160, 160)
    label_height = 28
    sheet = Image.new("RGB", ((1 + len(VARIANTS)) * thumb_size[0], len(cases) * (thumb_size[1] + label_height)), "white")
    for row_index, case in enumerate(cases):
        paths = [Path(str(case["reference_path"]))] + [Path(str(json_object(case, "candidates")[variant])) for variant in VARIANTS]
        for column, path in enumerate(paths):
            with Image.open(path) as raw:
                thumb = ImageOps.fit(raw.convert("RGB"), thumb_size, method=Image.Resampling.LANCZOS)
            sheet.paste(thumb, (column * thumb_size[0], row_index * (thumb_size[1] + label_height)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(_normalize(left), _normalize(right)))


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    left_mask = left > 0.0
    right_mask = right > 0.0
    union = np.logical_or(left_mask, right_mask).sum()
    return 1.0 if union == 0 else float(np.logical_and(left_mask, right_mask).sum() / union)
