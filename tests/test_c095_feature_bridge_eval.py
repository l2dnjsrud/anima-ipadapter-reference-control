from __future__ import annotations

from tools.c095_feature_bridge_eval import c095_siglip_variants
from tools.c095_feature_bridge_metrics import summarize_c095_shape_rows


def test_c095_variants_include_c094_and_bridge_weight_sweep() -> None:
    labels = tuple(variant.label for variant in c095_siglip_variants())

    assert len(labels) == 12
    assert "c092_qwen_target_w14" in labels
    assert "c093_anti_collapse_w14" in labels
    assert "c094_shape_supervised_w14" in labels
    assert labels[-4:] == (
        "c095_feature_bridge_w08",
        "c095_feature_bridge_w10",
        "c095_feature_bridge_w12",
        "c095_feature_bridge_w14",
    )


def test_c095_summary_accepts_broader_gate_candidate() -> None:
    summary = summarize_c095_shape_rows(
        _rows(c095=0.115, c094=0.095, qwen=0.120, heldout=0.090),
        diversity_by_variant={
            "c095_feature_bridge_w14": 0.055,
            "c094_shape_supervised_w14": 0.040,
        },
        pixel_audit=_pixel_audit(),
    )

    assert summary["best_c095_variant"] == "c095_feature_bridge_w14"
    assert summary["heldout07"]["best_c095_uplift"] == 0.090
    assert summary["decision"] == "c095_candidate_for_broader_shape_gate"


def test_c095_summary_rejects_when_not_close_to_qwen_baseline() -> None:
    summary = summarize_c095_shape_rows(
        _rows(c095=0.095, c094=0.088, qwen=0.120, heldout=0.055),
        diversity_by_variant={
            "c095_feature_bridge_w14": 0.055,
            "c094_shape_supervised_w14": 0.040,
        },
        pixel_audit=_pixel_audit(),
    )

    assert summary["decision"] == (
        "c095_feature_bridge_not_promoted_requires_siglip_encoder_finetune_or_data_expansion"
    )


def test_c095_summary_rejects_blank_like_bridge_rows() -> None:
    audit = _pixel_audit()
    audit["rows"] = [{"name": "heldout07_c095_feature_bridge_w14", "nonblank": False}]

    summary = summarize_c095_shape_rows(
        _rows(c095=0.115, c094=0.095, qwen=0.120, heldout=0.055),
        diversity_by_variant={
            "c095_feature_bridge_w14": 0.055,
            "c094_shape_supervised_w14": 0.040,
        },
        pixel_audit=audit,
    )

    assert summary["c095_blank_like_rows"] == ["heldout07_c095_feature_bridge_w14"]
    assert summary["decision"].startswith("c095_feature_bridge_not_promoted")


def _rows(
    *,
    c095: float,
    c094: float,
    qwen: float,
    heldout: float,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for sample in ("crop_pair00", "crop_pair01", "crop_pair02", "crop_pair03"):
        rows.extend(_sample_rows(sample, c095, c094, qwen))
    rows.extend(_sample_rows("heldout07", heldout, c094 - 0.03, qwen - 0.03))
    return tuple(rows)


def _sample_rows(
    sample: str,
    c095: float,
    c094: float,
    qwen: float,
) -> list[dict[str, object]]:
    return [
        {"sample": sample, "variant": "c092_qwen_target_w14", "uplift": c094 - 0.01},
        {"sample": sample, "variant": "c093_anti_collapse_w14", "uplift": c094 - 0.005},
        {"sample": sample, "variant": "c094_shape_supervised_w14", "uplift": c094},
        {"sample": sample, "variant": "c095_feature_bridge_w14", "uplift": c095},
        {"sample": sample, "variant": "c087_expanded_crop_positive_w14", "uplift": qwen},
    ]


def _pixel_audit() -> dict[str, object]:
    return {"rows": [{"name": "crop_pair00_no_ip", "nonblank": False}]}
