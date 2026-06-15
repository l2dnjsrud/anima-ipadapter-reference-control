# QwenVL c063 Calibrator-Only Training

- Decision: `not_promoted_after_generation_gate`
- Dataset: `training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.jsonl`
- Heldout rows used for training: `0`
- Init checkpoint: `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- Output checkpoint: `checkpoints/anima_qwenvl_ip_adapter_c063_calibrator_only_b128_0128_20260612.safetensors`
- Calibrator bottleneck dim: `128`
- Train calibrator only: `True`
- Instruction: c061 `species_face` instruction

| metric | value |
| --- | --- |
| steps | `128` |
| rows_loaded | `154` |
| first_loss | `0.33138132095336914` |
| final_loss | `0.1937534660100937` |
| mean_loss | `0.23729280498810112` |
| mean_base_loss | `0.18053316185250878` |
| mean_contrastive_loss | `0.05005351372528821` |
| mean_retrieval_loss | `0.1962045612744987` |
| finite_loss | `True` |
| trainable_parameters | `528384` |
| frozen_base_parameters | `4947838963` |
| checkpoint.loadable | `True` |
| checkpoint.pe_checkpoint_rejected | `True` |

Generation gate result: c063 was loadable and active in isolated ComfyUI, but it did not beat the current `blend_species_face` baseline on PE or visual heldout identity. See `eval/qwenvl_c063_calibrator_only_gate_20260612/report.md`.
