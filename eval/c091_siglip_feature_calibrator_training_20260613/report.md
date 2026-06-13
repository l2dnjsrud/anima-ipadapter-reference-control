# c091 SigLIP Feature-Calibrator Training

- Decision: `proceed_to_c091_generation_gate`
- Init checkpoint: `checkpoints/anima_siglip_ip_adapter_c089_shape_pe_teacher_0032_20260613.safetensors`
- Output checkpoint: `checkpoints/anima_siglip_ip_adapter_c091_feature_calibrator_b64_0064_20260613.safetensors`
- Steps / rows: `64` / `64`
- Feature calibrator bottleneck: `64`
- Train calibrator only: `true`
- Trainable parameters: `199680`
- First loss: `0.2653019428253174`
- Final loss: `0.1954212784767151`
- Loss delta: `-0.0698806643486023`
- Finite loss: `true`
- Checkpoint loadable: `true`
- PE checkpoint rejected by SigLIP loader: `true`
- Checkpoint class: `CalibratedIPAdapterSigLIP`
- Feature calibrator key count: `8`

## Interpretation

c091 successfully converts the c089 SigLIP hard-shape checkpoint into a calibrated checkpoint and trains only the `feature_calibrator.*` parameters. This is the intended feature-side adaptation surface rather than another full frozen-SigLIP adapter continuation. The training gate is cleared, but quality is undecided until the ComfyUI hard-shape generation gate compares c091 against `siglip_pilot_w14`, `c089_shape_w14`, and the recorded QwenVL hard-shape baselines.
