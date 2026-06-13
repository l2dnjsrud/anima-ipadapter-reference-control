from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Final

from tools.siglip_auto_caption_types import AutoPromptRow, JsonObject, JsonValue, Sample, Variant


C089_CHECKPOINT: Final = "anima_siglip_ip_adapter_c089_shape_pe_teacher_0032_20260613.safetensors"
SIGLIP_PILOT_CHECKPOINT: Final = "anima_siglip_ip_adapter_pilot_20260610.safetensors"
GENERIC_PROMPT: Final = (
    "masterpiece, best quality, score_7, safe, solo manhwa character, "
    "full color comic panel, clean line art, strong silhouette, faithful reference identity"
)
BASELINE_VARIANTS: Final = (
    "blend_species_face",
    "c086_hard_negative_w14",
    "c087_expanded_crop_positive_w14",
)


def siglip_variants() -> tuple[Variant, ...]:
    return (
        Variant("no_ip", None, 0.0),
        Variant("siglip_pilot_w14", SIGLIP_PILOT_CHECKPOINT, 1.4),
        Variant("c089_shape_w10", C089_CHECKPOINT, 1.0),
        Variant("c089_shape_w14", C089_CHECKPOINT, 1.4),
    )


def materialize_c090_prompt_manifest(
    probe_manifest_path: Path,
    *,
    out_dir: Path,
    reference_root: Path,
) -> tuple[tuple[Sample, ...], dict[str, dict[str, str]]]:
    rows = _read_jsonl(probe_manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_root.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    baselines: dict[str, dict[str, str]] = {}
    prompt_rows: list[JsonObject] = []
    for index, row in enumerate(rows):
        sample = _sample_from_probe_row(row, index, reference_root)
        samples.append(sample)
        prompt_rows.append(_prompt_manifest_row(sample))
        baselines[sample.label] = _baseline_candidates(row)
    (out_dir / "auto_reference_prompts.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in prompt_rows),
        encoding="utf-8",
    )
    return tuple(samples), baselines


def _sample_from_probe_row(row: JsonObject, index: int, reference_root: Path) -> Sample:
    label = str(row["sample"])
    ref_id = f"c090_refs/{label}"
    _copy_reference(Path(str(row["reference_path"])), reference_root / f"{ref_id}.jpg")
    prompt, attrs = _prompt_and_attributes(row, label)
    return Sample(
        label=label,
        ref_id=ref_id,
        seed=20260900 + index,
        prompt_row=AutoPromptRow(
            ref_id=ref_id,
            tgt_id=label,
            source_prompt=prompt,
            prompt=prompt,
            selected_attributes=attrs,
        ),
    )


def _copy_reference(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _prompt_and_attributes(row: JsonObject, label: str) -> tuple[str, tuple[str, ...]]:
    source_summary = row.get("source_summary_path")
    if isinstance(source_summary, str) and source_summary:
        sample = _source_sample(Path(source_summary), label)
        prompt = str(sample.get("prompt") or GENERIC_PROMPT)
        attrs = sample.get("selected_attributes")
        if isinstance(attrs, list):
            return prompt, tuple(str(item) for item in attrs)
        return prompt, (str(row.get("shape_group", "hard_shape")),)
    return GENERIC_PROMPT, (str(row.get("shape_group", "hard_shape")),)


def _source_sample(summary_path: Path, label: str) -> JsonObject:
    summary = _read_json(summary_path)
    raw_samples = summary.get("samples")
    if isinstance(raw_samples, list):
        for item in raw_samples:
            if isinstance(item, dict) and item.get("label") == label:
                return item
    return {}


def _baseline_candidates(row: JsonObject) -> dict[str, str]:
    raw = row.get("candidates")
    if not isinstance(raw, dict):
        return {}
    return {
        label: str(raw[label])
        for label in BASELINE_VARIANTS
        if isinstance(raw.get(label), str)
    }


def _prompt_manifest_row(sample: Sample) -> JsonObject:
    return {
        "ref_id": sample.ref_id,
        "tgt_id": sample.prompt_row.tgt_id,
        "source_prompt": sample.prompt_row.source_prompt,
        "prompt": sample.prompt_row.prompt,
        "selected_attributes": list(sample.prompt_row.selected_attributes),
    }


def _read_json(path: Path) -> JsonObject:
    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"json root must be object: {path}")
    return raw


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)
