# c096 SigLIP Encoder LoRA Training

- Decision: `proceed_to_c096_generation_gate`
- Checkpoint: `checkpoints/anima_siglip_encoder_lora_c096_rank8_0096_20260613.safetensors`
- Rows loaded: `10`
- Explicit negative rows: `10`
- Heldout rows: `[]`
- Loss: `0.018323052674531937` -> `0.007415413856506348`
- Mean positive similarity: `0.8003282845020294`
- Mean negative similarity: `0.7492882957061132`
- Rank/alpha: `8` / `8.0`

## Trainable Parameters

- `vision_model.encoder.layers.10.self_attn.v_proj.lora_down.weight`
- `vision_model.encoder.layers.10.self_attn.v_proj.lora_up.weight`
- `vision_model.encoder.layers.10.self_attn.q_proj.lora_down.weight`
- `vision_model.encoder.layers.10.self_attn.q_proj.lora_up.weight`
- `vision_model.encoder.layers.10.self_attn.out_proj.lora_down.weight`
- `vision_model.encoder.layers.10.self_attn.out_proj.lora_up.weight`
- `vision_model.encoder.layers.11.self_attn.v_proj.lora_down.weight`
- `vision_model.encoder.layers.11.self_attn.v_proj.lora_up.weight`
- `vision_model.encoder.layers.11.self_attn.q_proj.lora_down.weight`
- `vision_model.encoder.layers.11.self_attn.q_proj.lora_up.weight`
- `vision_model.encoder.layers.11.self_attn.out_proj.lora_down.weight`
- `vision_model.encoder.layers.11.self_attn.out_proj.lora_up.weight`
