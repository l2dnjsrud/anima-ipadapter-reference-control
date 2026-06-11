# SigLIP Identity128 Contrastive 512-Step Neutral-Prompt Sweep 2026-06-11

## Decision

`contrastive_improves_reference_distance_but_quality_still_fail`

- Pair sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c014_identity128_contrastive_neutral_prompt/reference_output_pairs.jpg`
- Summary: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c014_identity128_contrastive_neutral_prompt/summary.json`
- Checkpoint: `anima_siglip_ip_adapter_identity128_contrastive_0512_20260611.safetensors`

| Variant | In Train | Mean Abs vs No-IP | Mean Abs vs Ref |
|---|---:|---:|---:|
| train000_w10 | True | 74.9 | 68.6 |
| train064_w10 | True | 56.6 | 63.9 |
| train127_w10 | True | 50.2 | 67.8 |
| heldout0112_w10 | False | 73.7 | 118.9 |
| heldout0560_w10 | False | 76.5 | 99.5 |
| heldout1008_w10 | False | 59.9 | 66.2 |

## Visual Review

The contrastive 512-step checkpoint increases reference-dependent variation and
reduces mean pixel distance to several references compared with c013. It also
breaks the earlier two-character conversation collapse in multiple rows.

It still does not pass the usable reference-control bar. Train references are
not faithfully reconstructed, held-out identity is not preserved, and panel
layout/character count still drift toward dataset-average court scenes. Treat
this as evidence that a reference-swap objective helps, but is insufficient for
high-quality frozen-SigLIP2 control.
