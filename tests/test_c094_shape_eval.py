from __future__ import annotations

from tools.c094_shape_eval import c094_siglip_variants
from tools.c094_shape_metrics import summarize_c094_shape_rows


def test_c094_variants_include_expected_baselines_and_weight_sweep() -> None:
    labels = tuple(variant.label for variant in c094_siglip_variants())

    assert len(labels) == 11
    assert labels[:2] == ("no_ip", "siglip_pilot_w14")
    assert "c092_qwen_target_w14" in labels
    assert "c093_anti_collapse_w14" in labels
    assert labels[-4:] == (
        "c094_shape_supervised_w08",
        "c094_shape_supervised_w10",
        "c094_shape_supervised_w12",
        "c094_shape_supervised_w14",
    )


def test_c094_summary_accepts_full_threshold_candidate() -> None:
    summary = summarize_c094_shape_rows(
        _rows(c094=0.12, c093=0.09, c092=0.08, heldout=0.06),
        diversity_by_variant={
            "c094_shape_supervised_w14": 0.050,
            "c093_anti_collapse_w14": 0.040,
        },
        pixel_audit=_pixel_audit(),
    )

    assert summary["best_c094_variant"] == "c094_shape_supervised_w14"
    assert summary["heldout07"]["best_c094_uplift"] == 0.06
    assert summary["decision"] == "c094_candidate_for_broader_shape_gate"


def test_c094_summary_exhausts_when_heldout_fails() -> None:
    summary = summarize_c094_shape_rows(
        _rows(c094=0.12, c093=0.09, c092=0.08, heldout=0.02),
        diversity_by_variant={
            "c094_shape_supervised_w14": 0.050,
            "c093_anti_collapse_w14": 0.040,
        },
        pixel_audit=_pixel_audit(),
    )

    assert summary["variant_summaries"]["c094_shape_supervised_w14"]["mean_uplift"] >= 0.095
    assert summary["decision"] == "c094_shape_supervision_exhausted_requires_encoder_training"


def test_c094_summary_exhausts_when_c094_output_is_blank_like() -> None:
    audit = _pixel_audit()
    audit["rows"] = [{"name": "heldout07_c094_shape_supervised_w14", "nonblank": False}]

    summary = summarize_c094_shape_rows(
        _rows(c094=0.12, c093=0.09, c092=0.08, heldout=0.06),
        diversity_by_variant={
            "c094_shape_supervised_w14": 0.050,
            "c093_anti_collapse_w14": 0.040,
        },
        pixel_audit=audit,
    )

    assert summary["c094_blank_like_rows"] == ["heldout07_c094_shape_supervised_w14"]
    assert summary["decision"] == "c094_shape_supervision_exhausted_requires_encoder_training"


def _rows(
    *,
    c094: float,
    c093: float,
    c092: float,
    heldout: float,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for sample in ("crop_pair00", "crop_pair01", "crop_pair02", "crop_pair03"):
        rows.extend(_sample_rows(sample, c094, c093, c092))
    rows.extend(_sample_rows("heldout07", heldout, c093 - 0.05, c092 - 0.05))
    return tuple(rows)


def _sample_rows(
    sample: str,
    c094: float,
    c093: float,
    c092: float,
) -> list[dict[str, object]]:
    return [
        {"sample": sample, "variant": "c092_qwen_target_w14", "uplift": c092},
        {"sample": sample, "variant": "c093_anti_collapse_w14", "uplift": c093},
        {"sample": sample, "variant": "c094_shape_supervised_w14", "uplift": c094},
        {"sample": sample, "variant": "c087_expanded_crop_positive_w14", "uplift": 0.11},
    ]


def _pixel_audit() -> dict[str, object]:
    return {"rows": [{"name": "crop_pair00_no_ip", "nonblank": False}]}
