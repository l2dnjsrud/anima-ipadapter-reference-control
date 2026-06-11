# SigLIP c018 PE-Teacher Weight Sweep

- Checkpoint: `anima_siglip_ip_adapter_identity128_pe_teacher_0064_20260611.safetensors`
- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c018_pe_teacher_weight_sweep/contact_sheet.jpg`
- References: `train064_w10`, `heldout0560_w10`, `heldout1008_w10`
- Weights: `0.35`, `0.7`, `1.0`, `1.4`, `2.0`
- Decision: `weight_sweep_does_not_rescue_pe_teacher_checkpoint`

Visual read: increasing weight strengthens reference-dependent changes, but the
changes are not controlled enough to be useful. Lower weights drift toward the
same no-IP scene; higher weights introduce distorted or generic character
groups and do not recover the reference color, identity, or panel composition.

Conclusion: the c018 failure is not mainly a ComfyUI runtime weight issue. The
frozen SigLIP2 adapter path needs a stronger encoder signal or trainable
image-encoder adaptation before another broad training run is justified.
