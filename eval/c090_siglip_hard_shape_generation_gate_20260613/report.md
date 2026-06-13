# c090 SigLIP Hard-Shape Generation Gate

- Decision: `c089_improves_prior_siglip_but_not_qwen_baseline`
- Contact sheet: `eval/c090_siglip_hard_shape_generation_gate_20260613/contact_sheet_hard_shape.jpg`
- Pixel audit: `eval/c090_siglip_hard_shape_generation_gate_20260613/pixel_nonblank_audit.json`
- Low-variance / blank-like count: `2`

| variant | mean uplift | improved rate | cases |
| --- | ---: | ---: | ---: |
| `blend_species_face` | `-0.04365191457847577` | `0.36363636363636365` | `11` |
| `c086_hard_negative_w14` | `0.050174909082205704` | `0.7272727272727273` | `11` |
| `c087_expanded_crop_positive_w14` | `0.10895440559772807` | `0.9090909090909091` | `11` |
| `c089_shape_w10` | `0.0027134619528290964` | `0.5454545454545454` | `11` |
| `c089_shape_w14` | `0.024921435615903164` | `0.7272727272727273` | `11` |
| `no_ip` | `0.0` | `0.0` | `11` |
| `siglip_pilot_w14` | `-0.06624003835926072` | `0.18181818181818182` | `11` |
