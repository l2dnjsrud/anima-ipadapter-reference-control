from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.c106_qwen_teacher_feature_distillation import (
    C106ProbeConfig,
    C106ProbeError,
    build_c106_probe_manifest,
    write_c106_probe_summary,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_build_c106_probe_manifest_expands_positive_and_negative_rows(tmp_path: Path) -> None:
    source_manifest = tmp_path / "c097.jsonl"
    image_root = tmp_path / "root"
    _write_c097_row(source_manifest, image_root, index=0, group="green_guard", negative_group="red_mage")
    _write_c097_row(source_manifest, image_root, index=1, group="green_guard", negative_group="blue_robe")

    config = C106ProbeConfig(
        source_manifest=source_manifest,
        image_root=image_root,
        out_dir=tmp_path / "eval",
    )
    summary = build_c106_probe_manifest(config)

    rows = _read_jsonl(config.probe_manifest)
    assert summary.selected_rows == 2
    assert summary.positive_rows == 2
    assert summary.explicit_negative_rows == 2
    assert summary.pair_probe_rows == 4
    assert summary.heldout_rows_used == 0
    assert summary.missing_path_count == 0
    assert summary.c105_selected_route == "qwen_teacher_distillation"
    assert summary.decision == "ready_for_c106_qwen_teacher_probe"
    assert [row["label"] for row in rows] == ["positive", "negative", "positive", "negative"]
    assert rows[0]["anchor_id"] == "c097_hard_shape/pair_000_ref"
    assert rows[0]["candidate_id"] == "c097_hard_shape/pair_000_target"
    assert rows[0]["anchor_group"] == "green_guard"
    assert rows[1]["candidate_group"] == "red_mage"


def test_build_c106_probe_manifest_blocks_missing_images(tmp_path: Path) -> None:
    source_manifest = tmp_path / "c097.jsonl"
    _write_jsonl(
        source_manifest,
        (
            {
                "ref_id": "c097_hard_shape/pair_000_ref",
                "tgt_id": "c097_hard_shape/pair_000_target",
                "neg_id": "c097_hard_shape/pair_000_negative",
                "shape_group": "green_guard",
                "negative_shape_group": "red_mage",
            },
        ),
    )

    with pytest.raises(C106ProbeError, match="missing c106 image paths"):
        build_c106_probe_manifest(
            C106ProbeConfig(
                source_manifest=source_manifest,
                image_root=tmp_path / "missing_root",
                out_dir=tmp_path / "eval",
            )
        )


def test_write_c106_probe_summary_passes_strong_qwen_teacher_signal(tmp_path: Path) -> None:
    score_path = tmp_path / "qwenvl_pair_scores.json"
    _write_pair_scores(score_path, margin=0.07, auc=0.90)

    summary = write_c106_probe_summary(
        score_path=score_path,
        out_dir=tmp_path / "eval",
        manifest_summary_path=tmp_path / "manifest_summary.json",
    )

    assert summary["decision"] == "c106_probe_pass_prepare_training"
    assert summary["rows_evaluated"] == 2
    assert summary["explicit_negative_rows"] == 2
    assert summary["finite_metrics"] is True
    assert summary["next_branch"] == "c107_bounded_qwen_teacher_distillation_training"
    assert (tmp_path / "eval" / "probe_summary.json").is_file()
    assert (tmp_path / "eval" / "report.md").is_file()


def test_write_c106_probe_summary_rejects_weak_qwen_teacher_signal(tmp_path: Path) -> None:
    score_path = tmp_path / "qwenvl_pair_scores.json"
    _write_pair_scores(score_path, margin=0.02, auc=0.72)

    summary = write_c106_probe_summary(
        score_path=score_path,
        out_dir=tmp_path / "eval",
        manifest_summary_path=tmp_path / "manifest_summary.json",
    )

    assert summary["decision"] == "c106_probe_not_enough_signal_or_blocked"
    assert summary["next_branch"] == "manual_external_annotation_or_stronger_teacher_checkpoint"


def _write_pair_scores(path: Path, *, margin: float, auc: float) -> None:
    negative_mean = 0.50
    positive_mean = negative_mean + margin
    payload: JsonObject = {
        "summary": {
            "pairs": 4,
            "positive_pairs": 2,
            "negative_pairs": 2,
            "positive_mean": positive_mean,
            "negative_mean": negative_mean,
            "separation_margin": margin,
            "pairwise_auc": auc,
            "midpoint_accuracy": 0.75,
        },
        "rows": [
            _score_row("pair_000", "positive", positive_mean),
            _score_row("pair_000", "negative", negative_mean),
            _score_row("pair_001", "positive", positive_mean),
            _score_row("pair_001", "negative", negative_mean),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _score_row(pair_id: str, label: str, cosine: float) -> JsonObject:
    return {
        "pair_id": pair_id,
        "label": label,
        "anchor_id": f"{pair_id}_ref",
        "candidate_id": f"{pair_id}_{label}",
        "anchor_group": "green_guard",
        "candidate_group": "green_guard" if label == "positive" else "red_mage",
        "cosine": cosine,
    }


def _write_c097_row(
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
        "prompt": "mrcolor_panel_style, hard shape prompt",
        "shape_group": group,
        "negative_shape_group": negative_group,
        "source_pose_pair": "front->profile",
    }
    existing = _read_jsonl(source_manifest) if source_manifest.is_file() else ()
    _write_jsonl(source_manifest, existing + (row,))
    for suffix in ("ref", "target", "negative"):
        path = image_root / f"{prefix}_{suffix}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 40), "green").save(path)


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    parsed: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            parsed.append(raw)
    return tuple(parsed)
