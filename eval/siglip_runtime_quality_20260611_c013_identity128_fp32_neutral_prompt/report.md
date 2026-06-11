# SigLIP Identity128 FP32 3072-Step Neutral-Prompt Sweep 2026-06-11

## Decision

`fp32_training_moves_weights_but_quality_still_fail`

- Pair sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c013_identity128_fp32_neutral_prompt/reference_output_pairs.jpg`
- Summary: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c013_identity128_fp32_neutral_prompt/summary.json`
- Checkpoint: `anima_siglip_ip_adapter_identity128_fp32_3072_20260611.safetensors`

| Variant | In Train | Mean Abs vs No-IP | Mean Abs vs Ref |
|---|---:|---:|---:|
| train000_w10 | True | 62.9 | 77.8 |
| train064_w10 | True | 55.3 | 74.4 |
| train127_w10 | True | 51.7 | 78.3 |
| heldout0112_w10 | False | 67.3 | 120.7 |
| heldout0560_w10 | False | 68.7 | 105.2 |
| heldout1008_w10 | False | 57.0 | 74.2 |

## Visual Review

The fp32 continuation produces visible adapter influence, but it does not reach
usable reference-control quality. Outputs collapse toward a similar two-character
court/interior scene across most references. Train references are not faithfully
memorized, and held-out references do not preserve identity, layout, or distinct
palette. The checkpoint is therefore useful as diagnosis evidence, not as a
ready reference-control model.
