# SigLIP Identity128 Neutral-Prompt Sweep 2026-06-11

## Decision

`pending_visual_review`

This rerun removes gender/identity pressure from the prompt and removes the earlier female/woman/girl negative tokens so held-out references are not suppressed by prompt text.

## Evidence

- Pair sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c011_identity128_neutral_prompt/reference_output_pairs.jpg`
- Summary: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c011_identity128_neutral_prompt/summary.json`

## Metrics

| Variant | In Train | Weight | Mean Abs vs No-IP | Mean Abs vs Ref |
|---|---:|---:|---:|---:|
| train000_w08 | True | 0.8 | 58.9 | 93.1 |
| train000_w10 | True | 1.0 | 66.0 | 91.1 |
| train000_w12 | True | 1.2 | 67.2 | 80.5 |
| train064_w10 | True | 1.0 | 58.5 | 69.0 |
| train127_w10 | True | 1.0 | 53.1 | 70.7 |
| heldout0112_w08 | False | 0.8 | 59.7 | 110.8 |
| heldout0112_w10 | False | 1.0 | 64.0 | 114.1 |
| heldout0112_w12 | False | 1.2 | 68.4 | 114.7 |
| heldout0560_w10 | False | 1.0 | 65.1 | 121.6 |
| heldout1008_w10 | False | 1.0 | 58.3 | 68.7 |
