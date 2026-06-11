# QwenVL Identity128 Calibrated Contrastive 64-Step Native ComfyUI Weight Sweep

- Checkpoint: `anima_qwenvl_ip_adapter_identity128_calibrated_contrastive_0064_20260611.safetensors`
- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_runtime_quality_20260611_c004_calibrated_contrastive_weight_sweep/contact_sheet.jpg`
- Columns: reference / no_ip / qwenvl_w035 / qwenvl_w07 / qwenvl_w1.
- Decision: `qwen_feature_calibration_changes_outputs_but_reference_collapse_remains`

Visual read: the calibrated QwenVL checkpoint clearly changes pixels relative
to no-IP, but almost every reference converges to the same yellow-robed
street/interior conversation scene. The generated images do not preserve
reference-specific color, identity, character count, panel layout, or held-out
composition. Increasing weight strengthens the generic scene pattern instead of
recovering the reference.

| label | weight | d_no_ip | d_ref | stddev |
| --- | ---: | ---: | ---: | ---: |
| train000 | 0.35 | 39.36 | 99.17 | 74.55 |
| train000 | 0.70 | 42.95 | 102.51 | 71.88 |
| train000 | 1.00 | 45.97 | 104.42 | 72.31 |
| train064 | 0.35 | 40.44 | 69.15 | 73.56 |
| train064 | 0.70 | 43.28 | 65.28 | 71.63 |
| train064 | 1.00 | 46.85 | 65.77 | 72.88 |
| train127 | 0.35 | 40.39 | 77.67 | 73.87 |
| train127 | 0.70 | 43.13 | 78.14 | 71.51 |
| train127 | 1.00 | 46.54 | 78.93 | 72.26 |
| heldout0112 | 0.35 | 39.96 | 128.67 | 73.47 |
| heldout0112 | 0.70 | 43.06 | 134.56 | 71.54 |
| heldout0112 | 1.00 | 46.62 | 137.99 | 72.44 |
| heldout0560 | 0.35 | 40.51 | 129.37 | 74.14 |
| heldout0560 | 0.70 | 43.12 | 134.65 | 71.30 |
| heldout0560 | 1.00 | 46.54 | 137.10 | 72.14 |
| heldout1008 | 0.35 | 39.29 | 76.26 | 74.39 |
| heldout1008 | 0.70 | 43.09 | 74.05 | 72.09 |
| heldout1008 | 1.00 | 46.54 | 74.62 | 72.85 |
