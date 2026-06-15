# c092 Qwen-Target SigLIP Hard-Shape Gate

- Decision: `c092_improves_c089_but_not_qwen_baseline`
- Best c092: `c092_qwen_target_w14`
- Best c091 baseline: `c091_feature_calibrator_w14`
- Best c089 baseline: `c089_shape_w14`
- Best Qwen baseline: `c087_expanded_crop_positive_w14`
- Contact sheet: `eval/c092_qwen_target_siglip_generation_gate_20260613/contact_sheet_hard_shape.jpg`
- Pixel audit: `eval/c092_qwen_target_siglip_generation_gate_20260613/pixel_nonblank_audit.json`
- Visual audit: `eval/c092_qwen_target_siglip_generation_gate_20260613/visual_audit.md`
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
| `no_ip` | `0.0` | `0.0` | `11` |
| `siglip_pilot_w14` | `-0.06624003835926072` | `0.18181818181818182` | `11` |
