# QwenVL Identity128 64-Step Native ComfyUI Evaluation

- Checkpoint: `anima_qwenvl_ip_adapter_identity128_0064_20260611.safetensors`
- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_runtime_quality_20260611_c001_identity128/contact_sheet.jpg`
- Columns: reference / no_ip / qwenvl_w1.

| label | d_no_ip | d_ref | stddev |
| --- | ---: | ---: | ---: |
| train000 | 29.65 | 92.00 | 42.90 |
| train064 | 29.04 | 53.20 | 42.78 |
| train127 | 29.89 | 54.45 | 43.22 |
| heldout0112 | 29.65 | 131.86 | 43.07 |
| heldout0560 | 29.45 | 132.32 | 42.46 |
| heldout1008 | 28.90 | 59.03 | 42.57 |

## Decision

`adapter_changes_pixels_generation_quality_invalid`

This 4-step smoke proves that the QwenVL adapter branch changes the generated
pixels relative to no-IP. It is not a valid quality gate because the no-IP
baseline itself is blurry at 4 steps. A 16-step weight sweep is recorded in
`eval/qwenvl_runtime_quality_20260611_c002_identity128_weight_sweep/`.
