from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c102_vlm_teacher_core import parse_teacher_label
from tools.c102_vlm_teacher_gate import C102Config, build_c102_teacher_package
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c102_greenlights_when_eight_clean_qa_confirmed(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    rows = tuple(_candidate(f"green_{index}", data_root / f"green_{index}.jpg", "unclear") for index in range(8))
    for index in range(8):
        _image(data_root / f"green_{index}.jpg", (24, 160, 60))
    responses = tuple(_response(f"green_{index}", "LABEL: direct_green_non_human\nEVIDENCE: green creature face") for index in range(8))

    summary = build_c102_teacher_package(_config(tmp_path, rows, responses, min_confirmed_positive=8))

    reviewed = _rows_by_id(tmp_path / "out/c102_teacher_reviewed_manifest.jsonl")
    assert summary["covered_rows"] == 8
    assert summary["confirmed_local_positive_count"] == 8
    assert summary["teacher_only_positive_count"] == 0
    assert summary["decision"] == "c103_training_greenlit"
    assert reviewed["green_0"]["final_label"] == "local_positive"


def test_c102_blocks_prior_negative_teacher_disagreement(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _image(data_root / "negative.jpg", (24, 160, 60))
    rows = (_candidate("negative", data_root / "negative.jpg", "local_negative"),)
    responses = (_response("negative", "LABEL: direct_green_non_human\nEVIDENCE: green creature face"),)

    summary = build_c102_teacher_package(_config(tmp_path, rows, responses, min_confirmed_positive=1))

    reviewed = _rows_by_id(tmp_path / "out/c102_teacher_reviewed_manifest.jsonl")
    assert summary["qa_positive_candidate_count"] == 1
    assert summary["confirmed_local_positive_count"] == 0
    assert summary["teacher_only_positive_count"] == 1
    assert summary["decision"] == "c103_blocked_needs_manual_annotation_or_external_teacher"
    assert reviewed["negative"]["final_label"] == "local_negative"


def test_c102_parser_handles_qwen_label_variants() -> None:
    assert parse_teacher_label("LABEL: human_character\nEVIDENCE: ordinary person") == "human_character"
    assert parse_teacher_label("green_background_or_object\nshort phrase") == "green_background_or_object"
    assert parse_teacher_label("The image is unclear and not enough evidence.") == "unclear"


def _config(
    tmp_path: Path,
    rows: tuple[JsonObject, ...],
    responses: tuple[JsonObject, ...],
    *,
    min_confirmed_positive: int,
) -> C102Config:
    candidate_manifest = tmp_path / "c101.jsonl"
    c101_summary = tmp_path / "c101.summary.json"
    c100_summary = tmp_path / "c100.summary.json"
    heldout_manifest = tmp_path / "heldout.jsonl"
    response_source = tmp_path / "responses.jsonl"
    _write_jsonl(candidate_manifest, rows)
    _write_jsonl(heldout_manifest, ({"ref_id": "heldout"},))
    _write_jsonl(response_source, responses)
    c101_summary.write_text(
        json.dumps({"decision": "c102_blocked_needs_manual_annotation_or_teacher", "reviewed_local_positive_count": 0}),
        encoding="utf-8",
    )
    c100_summary.write_text(json.dumps({"candidate_rows": len(rows)}), encoding="utf-8")
    return C102Config(
        candidate_manifest=candidate_manifest,
        c101_summary=c101_summary,
        c100_summary=c100_summary,
        heldout_manifest=heldout_manifest,
        response_source=response_source,
        out_dir=tmp_path / "out",
        plan_path=tmp_path / "plan.md",
        min_confirmed_positive=min_confirmed_positive,
    )


def _candidate(image_id: str, image_path: Path, manual_label: str) -> JsonObject:
    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "source_type": "real_local_color",
        "source_bucket": "direct_green_pixel_candidate",
        "manual_label": manual_label,
        "label_evidence": "fixture",
        "green_ratio": 0.25,
        "strong_green_ratio": 0.18,
    }


def _response(image_id: str, raw_response: str) -> JsonObject:
    return {"image_id": image_id, "raw_response": raw_response}


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rows_by_id(path: Path) -> dict[str, JsonObject]:
    rows: dict[str, JsonObject] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows[str(raw["image_id"])] = raw
    return rows
