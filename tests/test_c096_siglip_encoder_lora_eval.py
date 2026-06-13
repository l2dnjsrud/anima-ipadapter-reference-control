from __future__ import annotations

from tools.c096_siglip_encoder_lora_eval import summarize_c096_shape_rows


def test_summarize_c096_promotes_when_mean_and_heldout_improve() -> None:
    summary = summarize_c096_shape_rows(
        (
            *_rows("crop_pair00", c096=0.13, c094=0.09, c095=0.08, qwen=0.10),
            *_rows("crop_pair01", c096=0.12, c094=0.09, c095=0.08, qwen=0.10),
            *_rows("heldout07", c096=0.06, c094=0.02, c095=0.01, qwen=0.04),
        ),
        {"rows": []},
    )

    assert summary["best_c096_variant"] == "c096_lora_c094_w14"
    assert summary["decision"] == "c096_encoder_lora_candidate_for_larger_gate"


def test_summarize_c096_rejects_blank_outputs() -> None:
    summary = summarize_c096_shape_rows(
        (*_rows("heldout07", c096=0.09, c094=0.02, c095=0.01, qwen=0.04),),
        {"rows": [{"name": "heldout07_c096_lora_c094_w14", "nonblank": False}]},
    )

    assert summary["c096_blank_like_rows"] == ["heldout07_c096_lora_c094_w14"]
    assert summary["decision"] == "c096_encoder_lora_not_promoted_blank_outputs"


def _rows(sample: str, *, c096: float, c094: float, c095: float, qwen: float) -> tuple[dict[str, object], ...]:
    return (
        {"sample": sample, "variant": "c096_lora_c094_w14", "uplift": c096},
        {"sample": sample, "variant": "c094_shape_supervised_w14", "uplift": c094},
        {"sample": sample, "variant": "c095_feature_bridge_w14", "uplift": c095},
        {"sample": sample, "variant": "c087_expanded_crop_positive_w14", "uplift": qwen},
    )
