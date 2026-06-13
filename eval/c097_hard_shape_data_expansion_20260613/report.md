# C097 hard-shape data expansion

- decision: `data_gate_pass_for_deeper_siglip_encoder_training`
- selected_rows: `56`
- explicit_negative_rows: `56`
- heldout_rows_used: `0`
- heldout_rows_rejected: `0`
- groups: `{"c082_frog_yokai_guard": 16, "c082_goblin_mage": 16, "c082_green_oni_scout": 16, "c082_jade_lizard_monk": 8}`
- manifest: `training/manifests/c097_siglip_hard_shape_expanded_pairs_20260613.jsonl`
- review_sheet: `eval/c097_hard_shape_data_expansion_20260613/pair_review_sheet.jpg`

C096은 10행 shallow encoder-LoRA라 hard-shape collapse를 깨지 못했다. C097은 더 깊은 SigLIP encoder adaptation 전에
4개 비인간/마스코트 계열 crop-pair를 확장하고, 각 row에 다른 shape group negative를 붙여 다음 학습 입력을 만든다.
