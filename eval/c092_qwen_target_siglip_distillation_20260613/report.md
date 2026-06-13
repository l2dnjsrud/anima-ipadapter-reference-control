# c092 Qwen-Target SigLIP Training

- Decision: `proceed_to_c092_generation_gate`
- Init checkpoint: `checkpoints/anima_siglip_ip_adapter_c089_shape_pe_teacher_0032_20260613.safetensors`
- Output checkpoint: `checkpoints/anima_siglip_ip_adapter_c092_qwen_target_0064_20260613.safetensors`
- Steps / rows: `64` / `10`
- First loss: `0.15548069775104523`
- Final loss: `0.07929979264736176`
- Loss delta: `-0.07618090510368347`
- Finite loss: `True`
- Trainable parameters: `217369756`
- Frozen base parameters: `2913827059`
- Checkpoint loadable: `True`
- PE checkpoint rejected by SigLIP loader: `True`

## Interpretation

The c092 training gate passed. This is a full SigLIP adapter continuation from c089 using materialized Qwen c087 hard-shape outputs as target images. It excludes `heldout07` from training. Quality remains undecided until the isolated ComfyUI hard-shape generation gate compares c092 with c089, c091, and the recorded Qwen baseline.
