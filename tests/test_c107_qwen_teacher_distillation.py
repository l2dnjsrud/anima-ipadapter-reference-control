from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.c107_qwen_teacher_distillation import (
    C107DistillationConfig,
    C107DistillationError,
    build_c107_training_manifest,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_build_c107_training_manifest_preserves_teacher_margins(tmp_path: Path) -> None:
    source_manifest = tmp_path / "c097.jsonl"
    image_root = tmp_path / "root"
    _write_source_row(source_manifest, image_root, index=0, group="frog", negative_group="mage")
    _write_source_row(source_manifest, image_root, index=1, group="oni", negative_group="lizard")
    scores_path = tmp_path / "scores.json"
    _write_scores(scores_path, ("pair_000", 0.91, 0.64), ("pair_001", 0.88, 0.70))

    config = C107DistillationConfig(
        source_manifest=source_manifest,
        image_root=image_root,
        score_path=scores_path,
        output_manifest=tmp_path / "c107.jsonl",
        summary_path=tmp_path / "c107.summary.json",
        plan_path=tmp_path / "plan.md",
    )
    summary = build_c107_training_manifest(config)

    rows = _read_jsonl(config.output_manifest)
    assert summary.rows_written == 2
    assert summary.explicit_negative_rows == 2
    assert summary.heldout_rows_used == 0
    assert summary.teacher_source == "C106"
    assert summary.teacher_margin_mean == pytest.approx(0.225)
    assert rows[0]["teacher_source"] == "C106"
    assert rows[0]["teacher_positive_cosine"] == pytest.approx(0.91)
    assert rows[0]["teacher_negative_cosine"] == pytest.approx(0.64)
    assert rows[0]["teacher_margin"] == pytest.approx(0.27)
    assert rows[0]["neg_id"] == "c097_hard_shape/pair_000_negative"
    assert json.loads(config.summary_path.read_text(encoding="utf-8"))["heldout_rows_used"] == 0
    assert "C107" in config.plan_path.read_text(encoding="utf-8")


def test_build_c107_training_manifest_skips_heldout_without_leakage(tmp_path: Path) -> None:
    source_manifest = tmp_path / "c097.jsonl"
    image_root = tmp_path / "root"
    _write_source_row(source_manifest, image_root, index=0, group="frog", negative_group="mage")
    _write_heldout_row(source_manifest, image_root)
    scores_path = tmp_path / "scores.json"
    _write_scores(scores_path, ("pair_000", 0.91, 0.64))

    summary = build_c107_training_manifest(
        C107DistillationConfig(
            source_manifest=source_manifest,
            image_root=image_root,
            score_path=scores_path,
            output_manifest=tmp_path / "c107.jsonl",
            summary_path=tmp_path / "c107.summary.json",
            plan_path=tmp_path / "plan.md",
        )
    )

    assert summary.rows_written == 1
    assert summary.heldout_rows_used == 0
    assert summary.heldout_rows_rejected == 1


def test_build_c107_training_manifest_rejects_incomplete_teacher_scores(tmp_path: Path) -> None:
    source_manifest = tmp_path / "c097.jsonl"
    image_root = tmp_path / "root"
    _write_source_row(source_manifest, image_root, index=0, group="frog", negative_group="mage")
    scores_path = tmp_path / "scores.json"
    payload: JsonObject = {
        "summary": {"separation_margin": 0.2},
        "rows": [_score_row("pair_000", "positive", "ref", "target", 0.9)],
    }
    scores_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(C107DistillationError, match="missing score labels"):
        build_c107_training_manifest(
            C107DistillationConfig(
                source_manifest=source_manifest,
                image_root=image_root,
                score_path=scores_path,
                output_manifest=tmp_path / "c107.jsonl",
                summary_path=tmp_path / "c107.summary.json",
                plan_path=tmp_path / "plan.md",
            )
        )


def test_build_c107_training_manifest_rejects_teacher_id_mismatch(tmp_path: Path) -> None:
    source_manifest = tmp_path / "c097.jsonl"
    image_root = tmp_path / "root"
    _write_source_row(source_manifest, image_root, index=0, group="frog", negative_group="mage")
    scores_path = tmp_path / "scores.json"
    payload: JsonObject = {
        "summary": {"separation_margin": 0.2},
        "rows": [
            _score_row("pair_000", "positive", "wrong_ref", "wrong_target", 0.9),
            _score_row("pair_000", "negative", "wrong_ref", "wrong_negative", 0.6),
        ],
    }
    scores_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(C107DistillationError, match="teacher score id mismatch"):
        build_c107_training_manifest(
            C107DistillationConfig(
                source_manifest=source_manifest,
                image_root=image_root,
                score_path=scores_path,
                output_manifest=tmp_path / "c107.jsonl",
                summary_path=tmp_path / "c107.summary.json",
                plan_path=tmp_path / "plan.md",
            )
        )


def _write_scores(path: Path, *pairs: tuple[str, float, float]) -> None:
    rows: list[JsonObject] = []
    for pair_id, positive, negative in pairs:
        prefix = f"c097_hard_shape/{pair_id.replace('pair_', 'pair_')}"
        rows.append(_score_row(pair_id, "positive", f"{prefix}_ref", f"{prefix}_target", positive))
        rows.append(_score_row(pair_id, "negative", f"{prefix}_ref", f"{prefix}_negative", negative))
    positive_mean = sum(pair[1] for pair in pairs) / len(pairs)
    negative_mean = sum(pair[2] for pair in pairs) / len(pairs)
    payload: JsonObject = {
        "summary": {
            "positive_pairs": len(pairs),
            "negative_pairs": len(pairs),
            "separation_margin": positive_mean - negative_mean,
            "pairwise_auc": 1.0,
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _score_row(
    pair_id: str,
    label: str,
    anchor_id: str,
    candidate_id: str,
    cosine: float,
) -> JsonObject:
    return {
        "pair_id": pair_id,
        "label": label,
        "anchor_id": anchor_id,
        "candidate_id": candidate_id,
        "cosine": cosine,
    }


def _write_source_row(
    source_manifest: Path,
    image_root: Path,
    *,
    index: int,
    group: str,
    negative_group: str,
) -> None:
    prefix = f"c097_hard_shape/pair_{index:03d}"
    row: JsonObject = {
        "ref_id": f"{prefix}_ref",
        "tgt_id": f"{prefix}_target",
        "neg_id": f"{prefix}_negative",
        "prompt": "mrcolor_panel_style, full color manhwa comic panel",
        "shape_group": group,
        "negative_shape_group": negative_group,
        "source_pose_pair": "front->profile",
    }
    _append_jsonl(source_manifest, row)
    _write_training_files(image_root, prefix)


def _write_heldout_row(source_manifest: Path, image_root: Path) -> None:
    prefix = "heldout/c097_hard_shape/pair_999"
    row: JsonObject = {
        "ref_id": f"{prefix}_ref",
        "tgt_id": f"{prefix}_target",
        "neg_id": f"{prefix}_negative",
        "prompt": "heldout row",
        "shape_group": "heldout",
        "negative_shape_group": "heldout_negative",
    }
    _append_jsonl(source_manifest, row)
    _write_training_files(image_root, prefix)


def _write_training_files(image_root: Path, prefix: str) -> None:
    for suffix in ("ref", "target", "negative"):
        path = image_root / f"{prefix}_{suffix}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 40), "green").save(path)
    (image_root / f"{prefix}_target.txt").write_text("caption\n", encoding="utf-8")


def _append_jsonl(path: Path, row: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    parsed: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            parsed.append(raw)
    return tuple(parsed)
