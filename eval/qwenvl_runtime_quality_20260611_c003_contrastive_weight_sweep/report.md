# QwenVL Identity128 Contrastive 64-Step Native ComfyUI Weight Sweep

- Checkpoint: `anima_qwenvl_ip_adapter_identity128_contrastive_0064_20260611.safetensors`
- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_runtime_quality_20260611_c003_contrastive_weight_sweep/contact_sheet.jpg`
- Columns: reference / no_ip / qwenvl_w035 / qwenvl_w07 / qwenvl_w1.

| label | weight | d_no_ip | d_ref | stddev |
| --- | ---: | ---: | ---: | ---: |
| train000 | 0.35 | 37.67 | 98.26 | 78.34 |
| train000 | 0.70 | 48.17 | 98.47 | 75.75 |
| train000 | 1.00 | 48.66 | 100.06 | 76.12 |
| train064 | 0.35 | 36.75 | 75.57 | 78.21 |
| train064 | 0.70 | 48.70 | 69.17 | 74.32 |
| train064 | 1.00 | 50.39 | 69.35 | 76.31 |
| train127 | 0.35 | 36.45 | 81.09 | 78.03 |
| train127 | 0.70 | 46.46 | 76.15 | 72.59 |
| train127 | 1.00 | 48.71 | 79.05 | 75.37 |
| heldout0112 | 0.35 | 38.34 | 127.09 | 79.01 |
| heldout0112 | 0.70 | 45.64 | 127.09 | 72.40 |
| heldout0112 | 1.00 | 49.83 | 131.80 | 75.92 |
| heldout0560 | 0.35 | 37.43 | 127.41 | 78.51 |
| heldout0560 | 0.70 | 44.42 | 126.18 | 71.19 |
| heldout0560 | 1.00 | 49.16 | 129.47 | 75.85 |
| heldout1008 | 0.35 | 38.18 | 82.38 | 79.01 |
| heldout1008 | 0.70 | 48.56 | 78.30 | 77.60 |
| heldout1008 | 1.00 | 48.49 | 77.30 | 75.72 |

## Decision

`qwenvl_contrastive_changes_outputs_but_not_quality_pass`

The reference-swap contrastive objective increases the adapter's ability to move
the image away from no-IP, especially at weights `0.7` and `1.0`. It still does
not recover reference-specific identity, color palette, layout, or character
count. Visually, most rows remain variants of the same yellow-robed
interior/conversation scene.

Compared with c002, mean distance from no-IP rises at higher weights, but mean
distance to the references does not improve enough to justify more short
adapter-only continuation. The next credible path is not another small
adapter-only run; it needs either trainable image-encoder/calibrator adaptation
or a higher-fidelity teacher/control objective.
