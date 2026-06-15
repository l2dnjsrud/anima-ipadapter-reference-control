from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.c104_expanded_qwen_target_siglip_probe import (
    C104ProbeConfig,
    C104ProbeError,
    build_c104_probe_manifest,
    write_c104_probe_summary,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_build_c104_probe_manifest_expands_positive_and_negative_rows(tmp_path: Path) -> None:
    source_manifest = tmp_path / "c097.jsonl"
    image_root = tmp_path / "root"
    _write_c097_row(source_manifest, image_root, index=0, group="frog", negative_group="goblin")
    _write_c097_row(source_manifest, image_root, index=1, group="frog", negative_group="oni")

    config = C104ProbeConfig(
        source_manifest=source_manifest,
        image_root=image_root,
        out_dir=tmp_path / "eval",
    )
    summary = build_c104_probe_manifest(config)

    rows = _read_jsonl(config.probe_manifest)
    assert summary.selected_rows == 2
    assert summary.positive_rows == 2
    assert summary.explicit_negative_rows == 2
    assert summary.token_probe_rows == 4
    assert summary.heldout_rows_used == 0
    assert summary.missing_path_count == 0
    assert [row["label"] for row in rows] == ["positive", "negative", "positive", "negative"]
    assert rows[0]["anchor_id"] == "c097_hard_shape/pair_000_ref"
    assert rows[1]["candidate_id"] == "c097_hard_shape/pair_000_negative"


def test_build_c104_probe_manifest_blocks_missing_images(tmp_path: Path) -> None:
    source_manifest = tmp_path / "c097.jsonl"
    _write_jsonl(
        source_manifest,
        (
            {
                "ref_id": "c097_hard_shape/pair_000_ref",
                "tgt_id": "c097_hard_shape/pair_000_target",
                "neg_id": "c097_hard_shape/pair_000_negative",
                "prompt": "prompt",
            },
        ),
    )

    with pytest.raises(C104ProbeError, match="missing c104 image paths"):
        build_c104_probe_manifest(
            C104ProbeConfig(
                source_manifest=source_manifest,
                image_root=tmp_path / "missing_root",
                out_dir=tmp_path / "eval",
            )
        )


def test_write_c104_probe_summary_requires_margin_above_qwen_baseline(tmp_path: Path) -> None:
    score_path = tmp_path / "token_scores.json"
    score_path.write_text(
        json.dumps(
            {
                "summaries": {
                    "pooled": {
                        "separation_margin": 0.11,
                        "pairwise_auc": 0.91,
                        "positive_mean": 0.60,
                        "negative_mean": 0.49,
                    },
                    "mean_max_token": {"separation_margin": 0.03, "pairwise_auc": 0.75},
                    "topk_token": {"separation_margin": 0.02, "pairwise_auc": 0.70},
                },
                "rows": [
                    _score_row("pair_000", "positive", 0.60),
                    _score_row("pair_000", "negative", 0.49),
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = write_c104_probe_summary(
        token_score_path=score_path,
        out_dir=tmp_path / "eval",
        manifest_summary_path=tmp_path / "manifest_summary.json",
        c098_best_mean_uplift=0.0865313863,
        qwen_baseline_mean_uplift=0.1089544056,
    )

    assert summary["decision"] == "c104_probe_pass_prepare_training"
    assert summary["best_siglip_metric"] == "pooled"
    assert summary["rows_evaluated"] == 1
    assert (tmp_path / "eval" / "per_row_metrics.jsonl").is_file()
    assert (tmp_path / "eval" / "report.md").is_file()


def test_write_c104_probe_summary_rejects_weak_margin(tmp_path: Path) -> None:
    score_path = tmp_path / "token_scores.json"
    score_path.write_text(
        json.dumps(
            {
                "summaries": {
                    "pooled": {"separation_margin": 0.04, "pairwise_auc": 0.90},
                    "mean_max_token": {"separation_margin": 0.05, "pairwise_auc": 0.84},
                    "topk_token": {"separation_margin": 0.03, "pairwise_auc": 0.80},
                },
                "rows": [
                    _score_row("pair_000", "positive", 0.54),
                    _score_row("pair_000", "negative", 0.50),
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = write_c104_probe_summary(
        token_score_path=score_path,
        out_dir=tmp_path / "eval",
        manifest_summary_path=tmp_path / "manifest_summary.json",
        c098_best_mean_uplift=0.0865313863,
        qwen_baseline_mean_uplift=0.1089544056,
    )

    assert summary["decision"] == "c104_probe_not_enough_signal"
    assert summary["next_branch"] == "stronger_encoder_checkpoint_or_manual_external_annotation"


def _score_row(pair_id: str, label: str, pooled: float) -> JsonObject:
    return {
        "pair_id": pair_id,
        "label": label,
        "anchor_id": f"{pair_id}_ref",
        "candidate_id": f"{pair_id}_{label}",
        "scores": {
            "pooled": pooled,
            "mean_max_token": pooled,
            "topk_token": pooled,
        },
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
