# QwenVL Identity128 64-Step Native ComfyUI Weight Sweep

- Checkpoint: `anima_qwenvl_ip_adapter_identity128_0064_20260611.safetensors`
- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_runtime_quality_20260611_c002_identity128_weight_sweep/contact_sheet.jpg`
- Columns: reference / no_ip / qwenvl_w035 / qwenvl_w07 / qwenvl_w1.

| label | weight | d_no_ip | d_ref | stddev |
| --- | ---: | ---: | ---: | ---: |
| train000 | 0.35 | 39.79 | 95.53 | 78.25 |
| train000 | 0.70 | 43.15 | 94.58 | 77.78 |
| train000 | 1.00 | 42.95 | 94.79 | 66.91 |
| train064 | 0.35 | 40.06 | 74.55 | 78.02 |
| train064 | 0.70 | 42.54 | 72.41 | 75.65 |
| train064 | 1.00 | 42.95 | 62.76 | 66.36 |
| train127 | 0.35 | 39.88 | 80.04 | 77.88 |
| train127 | 0.70 | 42.61 | 78.79 | 75.54 |
| train127 | 1.00 | 42.74 | 71.42 | 66.77 |
| heldout0112 | 0.35 | 39.19 | 124.32 | 77.90 |
| heldout0112 | 0.70 | 42.88 | 123.96 | 77.68 |
| heldout0112 | 1.00 | 43.16 | 128.57 | 67.49 |
| heldout0560 | 0.35 | 39.62 | 125.66 | 77.87 |
| heldout0560 | 0.70 | 43.13 | 125.11 | 78.03 |
| heldout0560 | 1.00 | 42.83 | 127.98 | 65.93 |
| heldout1008 | 0.35 | 40.16 | 81.57 | 78.19 |
| heldout1008 | 0.70 | 43.30 | 80.62 | 78.06 |
| heldout1008 | 1.00 | 42.41 | 71.77 | 67.58 |

## Decision

`qwenvl_denoising_changes_outputs_but_collapses_scene_average`

The 16-step generation baseline is clear enough for visual review. The QwenVL
adapter changes the image strongly relative to no-IP, but it does not preserve
reference identity, panel layout, character count, or palette. Across both
train and held-out references, outputs collapse toward a similar yellow-robed
interior/crowd scene. This is not usable reference control.
