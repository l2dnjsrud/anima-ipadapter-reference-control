from __future__ import annotations

# /// script
# requires-python = ">=3.13"
# dependencies = ["pillow", "torch", "transformers", "qwen-vl-utils", "typer"]
# ///
# ─── How to run ───
# PYTHONPATH=. python tools/c102_vlm_teacher_gate.py

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Protocol

import typer

from tools.c102_review_sheet import write_c102_review_sheet
from tools.c102_vlm_teacher_core import build_prompt, build_review_rows, build_summary
from tools.c102_vlm_teacher_text import build_plan, build_report
from tools.siglip_auto_caption_types import JsonObject, JsonValue

ROOT: Final = Path(__file__).resolve().parents[1]
OUT_DIR: Final = ROOT / "eval/c102_stronger_vlm_qa_teacher_gate_20260613"
DEFAULT_MODEL_PATH: Final = Path("/data/ai/models/LLM/Qwen-VL/Qwen3-VL-8B-Instruct")


class Teacher(Protocol):
    def ask(self, image_path: str, prompt: str) -> str: ...


@dataclass(frozen=True, slots=True)
class C102Config:
    candidate_manifest: Path = ROOT / "eval/c101_local_positive_annotation_teacher_gate_20260613/c101_reviewed_candidate_manifest.jsonl"
    c101_summary: Path = ROOT / "eval/c101_local_positive_annotation_teacher_gate_20260613/c101_candidate_summary.json"
    c100_summary: Path = ROOT / "eval/c100_local_real_color_positive_acquisition_20260613/c100_candidate_summary.json"
    heldout_manifest: Path = ROOT / "training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl"
    response_source: Path | None = None
    out_dir: Path = OUT_DIR
    plan_path: Path = ROOT / "docs/c102_stronger_vlm_qa_teacher_gate_plan_ko.md"
    model_path: Path = DEFAULT_MODEL_PATH
    min_confirmed_positive: int = 8


def build_c102_teacher_package(config: C102Config = C102Config(), teacher: Teacher | None = None) -> JsonObject:
    candidates = _read_jsonl(config.candidate_manifest)
    heldout_ids = _read_ids(config.heldout_manifest)
    heldout_leakage = sum(1 for row in candidates if _str(row, "image_id") in heldout_ids)
    prompts = tuple(_prompt_row(row) for row in candidates)
    responses = _responses(config, candidates, prompts, teacher)
    rows = build_review_rows(candidates, _response_map(responses))
    selected_status = _selected_teacher_status(config, teacher)
    summary = build_summary(
        rows,
        input_rows=len(candidates),
        heldout_leakage=heldout_leakage,
        min_confirmed_positive=config.min_confirmed_positive,
        selected_teacher_status=selected_status,
    )
    inventory = _inventory(config, heldout_ids, selected_status)
    _write_package(config, prompts, responses, rows, summary, inventory)
    return summary


def _responses(
    config: C102Config,
    candidates: tuple[JsonObject, ...],
    prompts: tuple[JsonObject, ...],
    teacher: Teacher | None,
) -> tuple[JsonObject, ...]:
    if config.response_source is not None:
        return _read_jsonl(config.response_source)
    active_teacher = teacher or _load_qwen_teacher(config)
    response_path = config.out_dir / "c102_vlm_qa_responses.jsonl"
    if response_path.is_file():
        response_path.unlink()
    rows: list[JsonObject] = []
    for candidate, prompt in zip(candidates, prompts):
        image_id = _str(candidate, "image_id")
        raw_response = active_teacher.ask(_str(candidate, "image_path"), _str(prompt, "prompt"))
        rows.append({"image_id": image_id, "raw_response": raw_response})
        _append_jsonl(response_path, rows[-1])
    return tuple(rows)


def _load_qwen_teacher(config: C102Config) -> Teacher:
    from tools.c102_qwen3vl_teacher import Qwen3VLTeacher, Qwen3VLTeacherConfig

    return Qwen3VLTeacher(Qwen3VLTeacherConfig(model_path=config.model_path))


def _write_package(
    config: C102Config,
    prompts: tuple[JsonObject, ...],
    responses: tuple[JsonObject, ...],
    rows: tuple[JsonObject, ...],
    summary: JsonObject,
    inventory: JsonObject,
) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.out_dir / "source_inventory.json", inventory)
    _write_jsonl(config.out_dir / "c102_vlm_qa_prompts.jsonl", prompts)
    _write_jsonl(config.out_dir / "c102_vlm_qa_responses.jsonl", responses)
    _write_jsonl(config.out_dir / "c102_teacher_reviewed_manifest.jsonl", rows)
    _write_json(config.out_dir / "c102_teacher_summary.json", summary)
    (config.out_dir / "c102_decision_report.md").write_text(
        build_report(summary, inventory),
        encoding="utf-8",
    )
    write_c102_review_sheet(rows, config.out_dir / "c102_teacher_review_sheet.jpg")
    config.plan_path.parent.mkdir(parents=True, exist_ok=True)
    config.plan_path.write_text(build_plan(inventory), encoding="utf-8")


def _inventory(config: C102Config, heldout_ids: set[str], selected_status: str) -> JsonObject:
    c101 = _read_json(config.c101_summary)
    c100 = _read_json(config.c100_summary)
    return {
        "source_paths": {
            "candidate_manifest": str(config.candidate_manifest),
            "c101_summary": str(config.c101_summary),
            "c100_summary": str(config.c100_summary),
            "heldout_manifest": str(config.heldout_manifest),
            "response_source": str(config.response_source or ""),
        },
        "heldout_ids": sorted(heldout_ids),
        "vlm_surfaces_checked": _vlm_surfaces(config),
        "selected_teacher": {
            "status": selected_status,
            "model_path": str(config.model_path),
            "reason": "Qwen3-VL-8B-Instruct is a local HF-format generative VLM and follows compact labels better than Qwen3-VL-2B-Thinking.",
        },
        "greenlight_criteria": {
            "candidate_rows": 64,
            "min_confirmed_positive": config.min_confirmed_positive,
            "teacher_only_positive_count": 0,
            "heldout_leakage_count": 0,
            "missing_path_count": 0,
        },
        "key_metrics": {
            "c101_decision": str(c101.get("decision", "")),
            "c101_reviewed_local_positive_count": int(c101.get("reviewed_local_positive_count", 0)),
            "c100_candidate_rows": int(c100.get("candidate_rows", 0)),
            "heldout_count": len(heldout_ids),
            "min_confirmed_positive": config.min_confirmed_positive,
        },
    }


def _vlm_surfaces(config: C102Config) -> tuple[JsonObject, ...]:
    return (
        {"surface": "repo_qwen3vl_embedding", "status": "embedding_only"},
        {"surface": "repo_c070_caption_search", "status": "sidecar_caption_heuristic_only"},
        {"surface": "local_qwen3vl_2b_thinking", "status": "runnable_but_verbose_thinking_output"},
        {"surface": "local_qwen3vl_8b_instruct", "status": "selected" if config.model_path.is_dir() else "missing"},
    )


def _selected_teacher_status(config: C102Config, teacher: Teacher | None) -> str:
    if config.response_source is not None:
        return "response_source_replay"
    if teacher is not None:
        return "injected_teacher"
    return "local_qwen3vl_8b_instruct_runnable" if config.model_path.is_dir() else "blocked_no_local_generative_vlm"


def _prompt_row(row: JsonObject) -> JsonObject:
    return {
        "image_id": _str(row, "image_id"),
        "image_path": _str(row, "image_path"),
        "prompt": build_prompt(row),
    }


def _response_map(rows: tuple[JsonObject, ...]) -> dict[str, str]:
    return {_str(row, "image_id"): _str(row, "raw_response") for row in rows}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _append_jsonl(path: Path, row: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _str(row: JsonObject, key: str) -> str:
    value = row.get(key)
    return value if isinstance(value, str) else ""


app = typer.Typer(add_completion=False)


@app.command()
def main(
    out_dir: Annotated[Path, typer.Option()] = OUT_DIR,
    response_source: Annotated[Path | None, typer.Option()] = None,
    model_path: Annotated[Path, typer.Option()] = DEFAULT_MODEL_PATH,
) -> None:
    summary = build_c102_teacher_package(
        C102Config(out_dir=out_dir, response_source=response_source, model_path=model_path)
    )
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
