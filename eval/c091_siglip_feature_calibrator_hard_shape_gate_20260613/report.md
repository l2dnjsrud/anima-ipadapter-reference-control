# c091 SigLIP Feature-Calibrator Hard-Shape Gate

- Decision: `c091_matches_c089_not_qwen_baseline`
- Best c091: `c091_feature_calibrator_w14`
- Best c089 baseline: `c089_shape_w14`
- Best Qwen baseline: `c087_expanded_crop_positive_w14`
- Contact sheet: `eval/c091_siglip_feature_calibrator_hard_shape_gate_20260613/contact_sheet_hard_shape.jpg`
- Pixel audit: `eval/c091_siglip_feature_calibrator_hard_shape_gate_20260613/pixel_nonblank_audit.json`
- Visual audit: `eval/c091_siglip_feature_calibrator_hard_shape_gate_20260613/visual_audit.md`
- Low-variance / blank-like count: `2`

| variant | mean uplift | improved rate | cases |
| --- | ---: | ---: | ---: |
| `blend_species_face` | `-0.04365191457847577` | `0.36363636363636365` | `11` |
| `c086_hard_negative_w14` | `0.050174909082205704` | `0.7272727272727273` | `11` |
| `c087_expanded_crop_positive_w14` | `0.10895440559772807` | `0.9090909090909091` | `11` |
| `c089_shape_w14` | `0.024921435615903164` | `0.7272727272727273` | `11` |
| `c091_feature_calibrator_w10` | `0.0022065424565951386` | `0.6363636363636364` | `11` |
| `c091_feature_calibrator_w14` | `0.024088173699299265` | `0.7272727272727273` | `11` |
| `no_ip` | `0.0` | `0.0` | `11` |
| `siglip_pilot_w14` | `-0.06624003835926072` | `0.18181818181818182` | `11` |
