from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c068_reviewed_attribute_labels import (
    C068Config,
    build_reviewed_attribute_labels,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_build_reviewed_attribute_labels_marks_direct_green_as_not_sufficient(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "dataset" / "green_object.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (96, 128), (80, 160, 80)).save(image_path)
    topk_path = tmp_path / "attribute_topk.json"
    topk_path.write_text(
        json.dumps(
            {
                "direct_green_non_human_face": [
                    _topk_row("train/green_object", image_path, 0.7)
                ],
                "background_object_green": [
                    _topk_row("train/green_object", image_path, 0.8)
                ],
            }
        ),
        encoding="utf-8",
    )
    heldout_path = tmp_path / "heldout.jsonl"
    heldout_path.write_text(
        json.dumps({"ref_id": "heldout/not_used", "tgt_id": "heldout/not_used"})
        + "\n",
        encoding="utf-8",
    )

    summary = build_reviewed_attribute_labels(
        C068Config(
            c067_topk_path=topk_path,
            heldout_manifest_path=heldout_path,
            out_dir=tmp_path / "out",
            source_c067_commit="testcommit",
        )
    )

    rows = _read_jsonl(tmp_path / "out" / "reviewed_attribute_labels.jsonl")
    assert summary["heldout_rows_used"] == 0
    assert summary["direct_green_target_positive_count"] == 0
    assert summary["decision"] == "direct_green_reviewed_seed_insufficient_new_annotation_required"
    assert rows[0]["review_label"] == "false_positive_human_old_face"
    assert rows[1]["review_label"] == "false_positive_background_object"
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "do not train encoder-side positives" in report
    assert "new captioned or manually reviewed" in report
    assert (tmp_path / "out" / "annotated_review_sheet.jpg").is_file()


def _topk_row(image_id: str, image_path: Path, score: float) -> JsonObject:
    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "source_bucket": "direct_green_pixel_candidate",
        "source_manifest": "c067",
        "source_label": "positive",
        "query_key": "direct_green_non_human_face",
        "query_text": "green non-human",
        "query_kind": "target_positive",
        "score": score,
    }


def _read_jsonl(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return rows
