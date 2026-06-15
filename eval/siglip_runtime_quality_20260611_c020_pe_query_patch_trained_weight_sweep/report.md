# SigLIP c020 PE-Style Query Patch Retrained Weight Sweep

- Checkpoint:
  `anima_siglip_ip_adapter_identity128_pe_query_patch_0064_20260611.safetensors`
  (local ignored artifact)
- Init checkpoint:
  `anima_siglip_ip_adapter_identity128_pe_teacher_0064_20260611.safetensors`
- Training: 64-step PE-teacher continuation after changing native/training
  SigLIP patching to prefer the Anima cross-attention `compute_qkv` query.
- Training summary:
  - rows loaded: `16`
  - first/final loss: `0.23008` / `0.60042`
  - mean loss: `0.25248`
  - mean base loss: `0.22360`
  - mean contrastive loss: `0.04329`
  - mean teacher loss: `0.02746`
  - finite loss: `true`
  - checkpoint loadable: `true`
  - PE checkpoint rejected by SigLIP loader: `true`
- Contact sheet:
  `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c020_pe_query_patch_trained_weight_sweep/contact_sheet.jpg`
- Columns: reference / no_ip / `w=0.35` / `w=0.7` / `w=1.0` / `w=1.4` / `w=2.0`.
- Decision: `short_query_patch_retrain_not_quality_pass`

Visual read: the retrained checkpoint preserves the stronger reference
influence introduced by the query patch, but does not solve the target quality
bar. The held-out face row gets more portrait-like at high weights, while the
blue held-out row still fails to recover the blue palette/layout. Higher weights
increase figure distortion and generic court/interior scenes.

Conclusion: PE-style query patching is the correct runtime/training geometry
for the native path and should be kept. A 64-step continuation is not enough to
turn the SigLIP checkpoint into a reliable reference-control model. Next work
needs a stronger objective or longer run under this corrected patch surface,
with explicit held-out reference gates.
