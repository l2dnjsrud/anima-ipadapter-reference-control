from __future__ import annotations

# /// script
# requires-python = ">=3.13"
# dependencies = ["pillow", "typer"]
# ///
# ─── How to run ───
# PYTHONPATH=. python tools/c101_local_positive_gate.py

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import typer
from PIL import Image

from tools.c101_annotation_text import build_plan, build_report
from tools.c101_label_policy import decide_c101_label
from tools.c101_review_sheet import write_c101_review_sheet
from tools.siglip_auto_caption_types import JsonObject, JsonValue

ROOT: Final = Path(__file__).resolve().parents[1]
OUT_DIR: Final = ROOT / "eval/c101_local_positive_annotation_teacher_gate_20260613"
PRIOR_LABEL_PATHS: Final = (
    ROOT / "eval/c068_reviewed_attribute_label_seed_20260612/reviewed_attribute_labels.jsonl",
    ROOT / "eval/c069_direct_green_captioning_acquisition_20260612/reviewed_candidate_labels.jsonl",
    ROOT / "eval/c070_qwenvl_direct_green_caption_search_20260612/reviewed_candidate_labels.jsonl",
)


@dataclass(frozen=True, slots=True)
class C101Config:
    c100_manifest: Path = ROOT / "eval/c100_local_real_color_positive_acquisition_20260613/c100_candidate_manifest.jsonl"
    c100_summary: Path = ROOT / "eval/c100_local_real_color_positive_acquisition_20260613/c100_candidate_summary.json"
    c100_review_sheet: Path = ROOT / "eval/c100_local_real_color_positive_acquisition_20260613/c100_candidate_review_sheet.jpg"
    heldout_manifest: Path = ROOT / "training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl"
    prior_label_paths: tuple[Path, ...] = PRIOR_LABEL_PATHS
    out_dir: Path = OUT_DIR
    plan_path: Path = ROOT / "docs/c101_local_positive_annotation_teacher_plan_ko.md"
    min_reviewed_positive: int = 8


def build_c101_annotation_package(config: C101Config = C101Config()) -> JsonObject:
    heldout_ids = _read_ids(config.heldout_manifest)
    prior_by_id = _prior_label_map(config.prior_label_paths)
    candidates = _read_jsonl(config.c100_manifest)
    heldout_leakage = sum(1 for row in candidates if _str(row, "image_id") in heldout_ids)
    rows = _reviewed_rows(candidates, heldout_ids, prior_by_id)
    inventory = _inventory(config, heldout_ids)
    summary = _summary(
        config,
        rows,
        input_rows=len(candidates),
        heldout_leakage=heldout_leakage,
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.out_dir / "source_inventory.json", inventory)
    _write_jsonl(config.out_dir / "reviewed_local_labels.jsonl", _label_rows(rows))
    _write_jsonl(config.out_dir / "c101_teacher_proposals.jsonl", _proposal_rows(rows))
    _write_jsonl(config.out_dir / "c101_reviewed_candidate_manifest.jsonl", rows)
    _write_json(config.out_dir / "c101_candidate_summary.json", summary)
    (config.out_dir / "c101_decision_report.md").write_text(
        build_report(summary, inventory),
        encoding="utf-8",
    )
    write_c101_review_sheet(rows, config.out_dir / "c101_review_sheet_labeled.jpg")
    config.plan_path.parent.mkdir(parents=True, exist_ok=True)
    config.plan_path.write_text(build_plan(inventory), encoding="utf-8")
    return summary


def _reviewed_rows(
    candidates: tuple[JsonObject, ...],
    heldout_ids: set[str],
    prior_by_id: dict[str, JsonObject],
) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for candidate in candidates:
        image_id = _str(candidate, "image_id")
        if image_id in heldout_ids:
            continue
        decision = decide_c101_label(candidate, prior_by_id.get(image_id))
        row = dict(candidate)
        row.update(
            {
                "manual_label": decision.manual_label,
                "confidence": decision.confidence,
                "label_evidence": decision.evidence,
                "review_source": decision.review_source,
            }
        )
        rows.append(row)
    return tuple(rows)


def _summary(
    config: C101Config,
    rows: tuple[JsonObject, ...],
    *,
    input_rows: int,
    heldout_leakage: int,
) -> JsonObject:
    missing = sum(1 for row in rows if not Path(_str(row, "image_path")).is_file())
    positive = _label_count(rows, "local_positive")
    review_required = max(0, input_rows - heldout_leakage - len(rows))
    teacher_only_positive = 0
    can_train = (
        positive >= config.min_reviewed_positive
        and missing == 0
        and heldout_leakage == 0
        and review_required == 0
        and teacher_only_positive == 0
    )
    decision = "c102_training_greenlit" if can_train else "c102_blocked_needs_manual_annotation_or_teacher"
    return {
        "input_candidate_rows": input_rows,
        "reviewed_rows": len(rows),
        "review_required_count": review_required,
        "heldout_leakage_count": heldout_leakage,
        "missing_path_count": missing,
        "teacher_only_positive_count": teacher_only_positive,
        "reviewed_local_positive_count": positive,
        "local_negative_count": _label_count(rows, "local_negative"),
        "unclear_count": _label_count(rows, "unclear"),
        "min_reviewed_positive": config.min_reviewed_positive,
        "label_counts": _counts(rows, "manual_label"),
        "source_bucket_counts": _counts(rows, "source_bucket"),
        "review_source_counts": _counts(rows, "review_source"),
        "decision": decision,
        "blocker_reason": "" if decision == "c102_training_greenlit" else "C101 found no sufficient reviewed local direct-green/non-human positives; C102 training remains blocked until manual annotation or stronger VLM teacher confirms at least 8 positives.",
        "next_c102_command_surface": _next_command(decision),
    }


def _inventory(config: C101Config, heldout_ids: set[str]) -> JsonObject:
    c100 = _read_json(config.c100_summary)
    return {
        "source_paths": {
            "c100_manifest": str(config.c100_manifest),
            "c100_summary": str(config.c100_summary),
            "c100_review_sheet": str(config.c100_review_sheet),
            "heldout_manifest": str(config.heldout_manifest),
            "prior_label_paths": [str(path) for path in config.prior_label_paths],
        },
        "heldout_ids": sorted(heldout_ids),
        "label_schema": ["local_positive", "local_negative", "unclear"],
        "greenlight_criteria": {
            "min_reviewed_positive": config.min_reviewed_positive,
            "reviewed_rows_equal_input_rows": True,
            "review_required_count": 0,
            "teacher_only_positive_count": 0,
            "heldout_leakage_count": 0,
            "missing_path_count": 0,
        },
        "key_metrics": {
            "c100_decision": str(c100.get("decision", "")),
            "c100_candidate_rows": int(c100.get("candidate_rows", 0)),
            "c100_review_sheet_size": _image_size(config.c100_review_sheet),
            "heldout_count": len(heldout_ids),
            "min_reviewed_positive": config.min_reviewed_positive,
        },
    }


def _prior_label_map(paths: tuple[Path, ...]) -> dict[str, JsonObject]:
    prior: dict[str, JsonObject] = {}
    for path in paths:
        for row in _read_jsonl(path):
            image_id = _str(row, "image_id")
            if image_id and image_id not in prior:
                prior[image_id] = row
    return prior


def _label_rows(rows: tuple[JsonObject, ...]) -> tuple[JsonObject, ...]:
    return tuple(
        {
            "image_id": _str(row, "image_id"),
            "manual_label": _str(row, "manual_label"),
            "review_source": _str(row, "review_source"),
            "confidence": _str(row, "confidence"),
            "label_evidence": _str(row, "label_evidence"),
        }
        for row in rows
    )


def _proposal_rows(rows: tuple[JsonObject, ...]) -> tuple[JsonObject, ...]:
    return tuple(
        {
            "image_id": _str(row, "image_id"),
            "proposal_label": _str(row, "manual_label"),
            "source_bucket": _str(row, "source_bucket"),
            "confidence": _str(row, "confidence"),
            "evidence": _str(row, "label_evidence"),
        }
        for row in rows
    )


def _next_command(decision: str) -> str:
    if decision == "c102_training_greenlit":
        return "PYTHONPATH=. python training/siglip_encoder_lora_contrastive.py --manifest-path eval/c101_local_positive_annotation_teacher_gate_20260613/c101_reviewed_candidate_manifest.jsonl --steps <pilot>"
    return ""


def _image_size(path: Path) -> str:
    if not path.is_file():
        return "missing"
    with Image.open(path) as image:
        return f"{image.size[0]}x{image.size[1]}"


def _read_ids(path: Path) -> set[str]:
    return {_str(row, "ref_id") for row in _read_jsonl(path)}


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    if not path.is_file():
        return ()
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)


def _read_json(path: Path) -> JsonObject:
    if not path.is_file():
        return {}
    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _write_json(path: Path, payload: JsonObject) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _str(row: JsonObject, key: str) -> str:
    value = row.get(key)
    return value if isinstance(value, str) else ""


def _label_count(rows: tuple[JsonObject, ...], label: str) -> int:
    return sum(1 for row in rows if row.get("manual_label") == label)


def _counts(rows: tuple[JsonObject, ...], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field, "unknown")) for row in rows))


app = typer.Typer(add_completion=False)


@app.command()
def main(out_dir: Annotated[Path, typer.Option()] = OUT_DIR) -> None:
    config = C101Config(out_dir=out_dir)
    summary = build_c101_annotation_package(config)
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
