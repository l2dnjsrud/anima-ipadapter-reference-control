# QwenVL c062 Calibrator Distillation Training

- Decision: `pending_generation_gate`
- Dataset: `training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.jsonl`
- Heldout rows used for training: `0`
- Init checkpoint: `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- Output checkpoint: `checkpoints/anima_qwenvl_ip_adapter_c062_calibrator_distill_b128_0096_20260612.safetensors`
- Calibrator bottleneck dim: `128`
- Instruction: c061 `species_face` instruction

| metric | value |
| --- | --- |
| steps | `96` |
| rows_loaded | `154` |
| first_loss | `0.24025645852088928` |
| final_loss | `0.18788939714431763` |
| mean_loss | `0.23382606574644646` |
| mean_base_loss | `0.17600078376320502` |
| mean_contrastive_loss | `0.04994656960479915` |
| mean_retrieval_loss | `0.20171990835418305` |
| finite_loss | `True` |
| checkpoint.loadable | `True` |
| checkpoint.pe_checkpoint_rejected | `True` |

Next: expose this checkpoint in isolated ComfyUI and compare it against `no_ip` and the current best `blend_species_face` runtime preset on the clean32+heldout8 gate.
