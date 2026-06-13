from __future__ import annotations

from tools.c098_siglip_encoder_lora_eval import summarize_c098_shape_rows


def test_summarize_c098_promotes_when_mean_and_heldout_beat_baselines() -> None:
    summary = summarize_c098_shape_rows(
        (
            *_rows("crop_pair00", c098=0.15, c096=0.11, c094=0.09, c095=0.08, qwen=0.10),
            *_rows("crop_pair01", c098=0.14, c096=0.10, c094=0.09, c095=0.08, qwen=0.10),
            *_rows("heldout07", c098=0.08, c096=0.04, c094=0.03, c095=0.02, qwen=0.04),
        ),
        {"rows": []},
    )

    assert summary["best_c098_variant"] == "c098_lora_c094_w14"
    assert summary["best_c096_variant"] == "c096_lora_c094_w14"
    assert summary["decision"] == "c098_encoder_lora_quality_candidate_for_larger_gate"


def test_summarize_c098_rejects_blank_outputs() -> None:
    summary = summarize_c098_shape_rows(
        (*_rows("heldout07", c098=0.12, c096=0.04, c094=0.03, c095=0.02, qwen=0.04),),
        {"rows": [{"name": "heldout07_c098_lora_c094_w14", "nonblank": False}]},
    )

    assert summary["c098_blank_like_rows"] == ["heldout07_c098_lora_c094_w14"]
    assert summary["decision"] == "c098_encoder_lora_not_promoted_blank_outputs"


def _rows(
    sample: str,
    *,
    c098: float,
    c096: float,
    c094: float,
    c095: float,
    qwen: float,
) -> tuple[dict[str, object], ...]:
    return (
        {"sample": sample, "variant": "c098_lora_c094_w14", "uplift": c098},
        {"sample": sample, "variant": "c096_lora_c094_w14", "uplift": c096},
        {"sample": sample, "variant": "c094_shape_supervised_w14", "uplift": c094},
        {"sample": sample, "variant": "c095_feature_bridge_w14", "uplift": c095},
        {"sample": sample, "variant": "c087_expanded_crop_positive_w14", "uplift": qwen},
    )
