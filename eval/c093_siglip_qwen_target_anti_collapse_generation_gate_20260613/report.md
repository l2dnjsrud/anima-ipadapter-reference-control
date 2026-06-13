# c093 SigLIP Anti-Collapse Hard-Shape Gate

- Decision: `c093_anti_collapse_not_promoted`
- Best c093: `c093_anti_collapse_w14`
- Best c092 baseline: `c092_qwen_target_w14`
- Best c089 baseline: `c089_shape_w14`
- Best Qwen baseline: `c087_expanded_crop_positive_w14`
- Heldout07: `{"best_c093_variant": "c093_anti_collapse_w14", "best_c093_uplift": 0.004518958388039396, "c092_w14_uplift": 0.0009986629992987384}`
- Diversity proxy: `{"best_c093": 0.08459516101413303, "c092_qwen_target_w14": 0.07203983200920952, "by_variant": {"siglip_pilot_w14": 0.385618472761578, "c089_shape_w14": 0.16404802269405788, "c091_feature_calibrator_w14": 0.16145913733376396, "c092_qwen_target_w10": 0.08962256378597683, "c092_qwen_target_w14": 0.07203983200920952, "c093_anti_collapse_w08": 0.11075587670008341, "c093_anti_collapse_w10": 0.09035154051250882, "c093_anti_collapse_w12": 0.08574090931150648, "c093_anti_collapse_w14": 0.08459516101413303}}`
- Contact sheet: `eval/c093_siglip_qwen_target_anti_collapse_generation_gate_20260613/contact_sheet_hard_shape.jpg`
- Pixel audit: `eval/c093_siglip_qwen_target_anti_collapse_generation_gate_20260613/pixel_nonblank_audit.json`
- Low-variance / blank-like count: `1`

| variant | mean uplift | improved rate | cases |
| --- | ---: | ---: | ---: |
| `blend_species_face` | `-0.04365191457847577` | `0.36363636363636365` | `11` |
| `c086_hard_negative_w14` | `0.050174909082205704` | `0.7272727272727273` | `11` |
| `c087_expanded_crop_positive_w14` | `0.10895440559772807` | `0.9090909090909091` | `11` |
| `c089_shape_w14` | `0.024921435615903164` | `0.7272727272727273` | `11` |
| `c091_feature_calibrator_w14` | `0.024088173699299265` | `0.7272727272727273` | `11` |
| `c092_qwen_target_w10` | `0.07382548763785077` | `0.9090909090909091` | `11` |
| `c092_qwen_target_w14` | `0.08526816531384505` | `1.0` | `11` |
| `c093_anti_collapse_w08` | `0.05401194838203038` | `0.8181818181818182` | `11` |
| `c093_anti_collapse_w10` | `0.08153196775281521` | `0.9090909090909091` | `11` |
| `c093_anti_collapse_w12` | `0.08405881214678951` | `1.0` | `11` |
| `c093_anti_collapse_w14` | `0.08637357797831002` | `1.0` | `11` |
| `no_ip` | `0.0` | `0.0` | `11` |
| `siglip_pilot_w14` | `-0.06624003835926072` | `0.18181818181818182` | `11` |
