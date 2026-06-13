# c096 SigLIP Encoder LoRA Hard-Shape Gate

- Decision: `c096_encoder_lora_not_promoted_requires_data_expansion_or_deeper_encoder_training`
- Best c096: `c096_lora_c094_w14`
- Best c094 baseline: `c094_shape_supervised_w14`
- Best c095 baseline: `c095_feature_bridge_w14`
- Best Qwen baseline: `c087_expanded_crop_positive_w14`
- Heldout07: `{"best_c096_uplift": 0.005656986085992688, "c094_w14_uplift": 0.008562351254518985, "c095_w14_uplift": 0.01154170337811311}`
- C096 blank-like rows: `[]`
- Low-variance / blank-like count: `1`

| variant | mean uplift | improved rate | cases |
| --- | ---: | ---: | ---: |
| `blend_species_face` | `-0.04365191457847577` | `0.36363636363636365` | `11` |
| `c086_hard_negative_w14` | `0.050174909082205704` | `0.7272727272727273` | `11` |
| `c087_expanded_crop_positive_w14` | `0.10895440559772807` | `0.9090909090909091` | `11` |
| `c094_shape_supervised_w14` | `0.08788329540499396` | `0.9090909090909091` | `11` |
| `c095_feature_bridge_w14` | `0.08652233470140815` | `0.9090909090909091` | `11` |
| `c096_lora_c094_w08` | `0.057639721463887655` | `0.7272727272727273` | `11` |
| `c096_lora_c094_w10` | `0.07860397621758765` | `0.8181818181818182` | `11` |
| `c096_lora_c094_w12` | `0.08308309275453575` | `0.9090909090909091` | `11` |
| `c096_lora_c094_w14` | `0.08808495528618471` | `0.9090909090909091` | `11` |
| `no_ip` | `0.0` | `0.0` | `11` |
