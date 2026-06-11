# SigLIP c019 PE-Style Query Patch Weight Sweep

- Checkpoint: `anima_siglip_ip_adapter_identity128_pe_teacher_0064_20260611.safetensors`
- Runtime change: native SigLIP/QwenVL patch now prefers the same
  cross-attention query produced by Anima `compute_qkv`, instead of treating
  pre-attention hidden states as a separate IP query stream.
- Contact sheet:
  `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c019_pe_query_patch_weight_sweep/contact_sheet.jpg`
- Columns: reference / no_ip / `w=0.35` / `w=0.7` / `w=1.0` / `w=1.4` / `w=2.0`.
- Decision: `query_patch_increases_reference_influence_but_not_quality_pass`

Visual read: the PE-style query path is a real improvement over the earlier
generic yellow-robed collapse. Different references produce more different
gender, pose, color, and crop tendencies. It is still not a production
reference-control result: the train row does not recover the palace composition,
the blue held-out row does not recover the blue palette/layout, and higher
weights often distort figures or add unrelated speech bubbles.

Conclusion: the native patch geometry was a real bottleneck. Fixing it makes
the adapter influence stronger, but the existing checkpoint remains undertrained
and does not pass the high-quality reference-control gate.
