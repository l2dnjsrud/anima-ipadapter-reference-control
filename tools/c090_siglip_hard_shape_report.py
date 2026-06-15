from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.c088_shape_metrics import EDGE_END, PROJECTION_END, _cosine, _iou, _shape_feature
from tools.c090_siglip_hard_shape_data import BASELINE_VARIANTS
from tools.siglip_auto_caption_types import JsonObject, JsonValue, Sample, Variant


def summarize_shape_rows(rows: Iterable[JsonObject]) -> JsonObject:
    by_variant: dict[str, list[float]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(float(row["uplift"]))
    summaries = {
        variant: {
            "cases": len(values),
            "mean_uplift": sum(values) / len(values),
            "improved_rate": sum(1 for value in values if value > 0.0) / len(values),
        }
        for variant, values in sorted(by_variant.items())
        if values
    }
    c089 = _best_prefixed(summaries, "c089_")
    pilot = summaries.get("siglip_pilot_w14", {"mean_uplift": 0.0})
    qwen = _best_named(summaries, BASELINE_VARIANTS)
    decision = _decision(c089, pilot, qwen)
    return {
        "variant_summaries": summaries,
        "best_c089_variant": c089[0],
        "best_qwen_baseline_variant": qwen[0],
        "decision": decision,
    }


def write_c090_shape_metrics(summary_path: Path, output_dir: Path) -> JsonObject:
    summary = _read_json(summary_path)
    rows: list[JsonObject] = []
    for sample in summary["samples"]:
        if not isinstance(sample, dict):
            continue
        sample_rows = _score_sample(summary, sample)
        rows.extend(sample_rows)
    rollup = summarize_shape_rows(rows)
    metrics = {"summary_path": str(summary_path), "rows": rows, "rollup": rollup}
    _write_json(output_dir / "shape_metrics.json", metrics)
    _write_json(output_dir / "metric_rollup.json", rollup)
    return rollup


def write_extended_contact_sheet(
    samples: tuple[Sample, ...],
    generated_variants: tuple[Variant, ...],
    baseline_candidates: dict[str, dict[str, str]],
    *,
    data_root: Path,
    out_dir: Path,
    output_path: Path,
) -> None:
    columns = ("reference", *(variant.label for variant in generated_variants), *BASELINE_VARIANTS)
    cell = (210, 260)
    label_h = 30
    margin = 12
    sheet = Image.new(
        "RGB",
        (margin * 2 + len(columns) * cell[0], margin * 2 + (len(samples) + 1) * (cell[1] + label_h)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, title in enumerate(columns):
        draw.text((margin + column * cell[0] + 6, margin + 8), title, fill="black", font=font)
    for row_index, sample in enumerate(samples, start=1):
        y = margin + row_index * (cell[1] + label_h)
        draw.text((margin + 6, y - label_h + 8), sample.label, fill="black", font=font)
        paths = _contact_paths(sample, generated_variants, baseline_candidates, data_root, out_dir)
        for column, path in enumerate(paths):
            sheet.paste(_fit(path, cell), (margin + column * cell[0], y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_report(output_dir: Path, rollup: JsonObject, *, blank_count: int) -> None:
    lines = [
        "# c090 SigLIP Hard-Shape Generation Gate",
        "",
        f"- Decision: `{rollup['decision']}`",
        f"- Contact sheet: `{output_dir / 'contact_sheet.jpg'}`",
        f"- Blank count: `{blank_count}`",
        "",
        "| variant | mean uplift | improved rate | cases |",
        "| --- | ---: | ---: | ---: |",
    ]
    summaries = rollup["variant_summaries"]
    if isinstance(summaries, dict):
        for variant, raw in summaries.items():
            if isinstance(raw, dict):
                lines.append(
                    f"| `{variant}` | `{raw['mean_uplift']}` | `{raw['improved_rate']}` | `{raw['cases']}` |"
                )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _score_sample(summary: JsonObject, sample: JsonObject) -> list[JsonObject]:
    label = str(sample["label"])
    ref = _shape_feature(Path(str(sample["reference_path"])))
    candidates = _candidate_paths(summary, label)
    no_ip = _shape_score(ref, candidates["no_ip"])
    rows: list[JsonObject] = []
    for variant, path in candidates.items():
        score = _shape_score(ref, path)
        rows.append({"sample": label, "variant": variant, "shape_score": score, "uplift": score - no_ip})
    return rows


def _candidate_paths(summary: JsonObject, label: str) -> dict[str, Path]:
    results = summary["results"]
    baselines = summary.get("baseline_candidates", {})
    paths: dict[str, Path] = {}
    if isinstance(results, dict):
        for key, raw in results.items():
            if key.startswith(f"{label}_") and isinstance(raw, dict):
                paths[key.removeprefix(f"{label}_")] = Path(str(raw["image"]))
    if isinstance(baselines, dict) and isinstance(baselines.get(label), dict):
        for variant, path in baselines[label].items():
            paths[str(variant)] = Path(str(path))
    return paths


def _shape_score(ref: np.ndarray, path: Path) -> float:
    feature = _shape_feature(path)
    edge = _cosine(ref[:EDGE_END], feature[:EDGE_END])
    projection = _cosine(ref[EDGE_END:PROJECTION_END], feature[EDGE_END:PROJECTION_END])
    silhouette = _iou(ref[PROJECTION_END:], feature[PROJECTION_END:])
    return 0.45 * edge + 0.30 * projection + 0.25 * silhouette


def _best_prefixed(summaries: JsonObject, prefix: str) -> tuple[str, float]:
    return max(
        (
            (variant, float(raw["mean_uplift"]))
            for variant, raw in summaries.items()
            if isinstance(raw, dict) and variant.startswith(prefix)
        ),
        key=lambda item: item[1],
        default=("", float("-inf")),
    )


def _best_named(summaries: JsonObject, names: tuple[str, ...]) -> tuple[str, float]:
    return max(
        (
            (name, float(summaries[name]["mean_uplift"]))
            for name in names
            if isinstance(summaries.get(name), dict)
        ),
        key=lambda item: item[1],
        default=("", float("-inf")),
    )


def _decision(c089: tuple[str, float], pilot: JsonObject, qwen: tuple[str, float]) -> str:
    pilot_uplift = float(pilot["mean_uplift"])
    if c089[1] >= pilot_uplift + 0.01 and c089[1] >= qwen[1] - 0.01:
        return "c089_shape_siglip_candidate_for_larger_gate"
    if c089[1] >= pilot_uplift + 0.01:
        return "c089_improves_prior_siglip_but_not_qwen_baseline"
    return "c089_not_promoted_escalate_encoder_side"


def _contact_paths(
    sample: Sample,
    generated_variants: tuple[Variant, ...],
    baseline_candidates: dict[str, dict[str, str]],
    data_root: Path,
    out_dir: Path,
) -> list[Path]:
    return [
        data_root / f"{sample.ref_id}.jpg",
        *[out_dir / f"{sample.label}_{variant.label}.png" for variant in generated_variants],
        *[Path(baseline_candidates[sample.label][variant]) for variant in BASELINE_VARIANTS],
    ]


def _fit(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as raw:
        image = raw.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _read_json(path: Path) -> JsonObject:
    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"json root must be object: {path}")
    return raw


def _write_json(path: Path, payload: JsonObject) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
