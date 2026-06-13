# C098 SigLIP Encoder LoRA Training

- manifest: `training/manifests/c097_siglip_hard_shape_expanded_pairs_20260613.jsonl`
- image_root: `.tmp/c097_siglip_hard_shape_expanded_root`
- checkpoint: `checkpoints/anima_siglip_encoder_lora_c098_rank8_layer4_0224_20260613.safetensors`
- steps: `224`
- rows_loaded: `56`
- explicit_negative_rows: `56`
- final_loss: `0.0023478984367102385`
- finite_loss: `true`
- checkpoint_loadable: `true`
- rank: `8`
- alpha: `8.0`
- trainable_parameter_count: `24`
- module_count: `12`

## Decision

Training gate passed. This checkpoint can proceed to the C098 hard-shape ComfyUI generation gate.
