# SigLIP Identity128 Reference Sweep 2026-06-11

## Decision

`pending_visual_review`

The 128-row color self-reconstruction continuation completed and generated a real ComfyUI API comparison sheet. This report records the raw evidence; final pass/fail depends on visual review against the contact sheet.

## Setup

- Checkpoint: `anima_siglip_ip_adapter_identity128_1024_20260611.safetensors`
- Init checkpoint: `checkpoints/anima_siglip_ip_adapter_identity8_1024_20260611.safetensors`
- Manifest: `training/manifests/local_color_self_identity128_20260611.jsonl`
- Rows: 128
- Steps: 1024
- Learning rate: 2e-5
- API server: `http://127.0.0.1:8115` isolated ComfyUI base

## Evidence

- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c010_identity128_reference_sweep/contact_sheet.jpg`
- Summary: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c010_identity128_reference_sweep/summary.json`
- Reference selection: `eval/identity128_reference_selection_20260611.json`

## Metrics

| Variant | In Train | Weight | Mean Abs vs No-IP | Mean Abs vs Ref |
|---|---:|---:|---:|---:|
| train000_w08 | True | 0.8 | 58.8 | 86.5 |
| train000_w10 | True | 1.0 | 62.3 | 84.2 |
| train000_w12 | True | 1.2 | 62.2 | 77.6 |
| train064_w10 | True | 1.0 | 65.0 | 69.3 |
| train127_w10 | True | 1.0 | 60.8 | 69.6 |
| heldout0112_w08 | False | 0.8 | 59.4 | 108.3 |
| heldout0112_w10 | False | 1.0 | 59.2 | 112.5 |
| heldout0112_w12 | False | 1.2 | 63.4 | 111.6 |
| heldout0560_w10 | False | 1.0 | 61.7 | 114.2 |
| heldout1008_w10 | False | 1.0 | 62.9 | 71.8 |

## Initial Interpretation

This run must be compared visually to c009. Numeric pixel distance only proves the adapter changes output, not that identity/control quality is sufficient.
