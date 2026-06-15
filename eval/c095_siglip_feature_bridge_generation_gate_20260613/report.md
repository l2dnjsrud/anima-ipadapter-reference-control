# c095 SigLIP Feature-Bridge Hard-Shape Gate

- Decision: `c095_feature_bridge_not_promoted_requires_siglip_encoder_finetune_or_data_expansion`
- Best c095: `c095_feature_bridge_w14`
- Best c094 baseline: `c094_shape_supervised_w14`
- Best c093 baseline: `c093_anti_collapse_w14`
- Best c092 baseline: `c092_qwen_target_w14`
- Best Qwen baseline: `c087_expanded_crop_positive_w14`
- Heldout07: `{"best_c095_variant": "c095_feature_bridge_w14", "best_c095_uplift": 0.01154170337811311, "c094_w14_uplift": 0.008562351254518985, "c093_w14_uplift": 0.004518958388039396, "c092_w14_uplift": 0.0009986629992987384}`
- C095 blank-like rows: `[]`
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
| `c093_anti_collapse_w14` | `0.08637357797831002` | `1.0` | `11` |
| `c094_shape_supervised_w14` | `0.08788329540499396` | `0.9090909090909091` | `11` |
| `c095_feature_bridge_w08` | `0.058369752000306405` | `0.8181818181818182` | `11` |
| `c095_feature_bridge_w10` | `0.08022350590108185` | `0.9090909090909091` | `11` |
| `c095_feature_bridge_w12` | `0.08289596753891405` | `0.8181818181818182` | `11` |
| `c095_feature_bridge_w14` | `0.08652233470140815` | `0.9090909090909091` | `11` |
| `no_ip` | `0.0` | `0.0` | `11` |
| `siglip_pilot_w14` | `-0.06624003835926072` | `0.18181818181818182` | `11` |
