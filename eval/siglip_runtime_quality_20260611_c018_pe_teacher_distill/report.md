# SigLIP c018 PE-Teacher Distillation Evaluation

- Checkpoint: `anima_siglip_ip_adapter_identity128_pe_teacher_0064_20260611.safetensors`
- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c018_pe_teacher_distill/contact_sheet.jpg`
- Columns: reference / no_ip / c016 / c017 / c018 teacher.
- Decision: `pe_teacher_distillation_changes_outputs_but_not_quality_pass`
- Visual read: c018 produces reference-dependent images, but the dependency is
  not reliable reference control. The held-out rows do not preserve reference
  color, panel layout, identity, or composition well enough to use as a
  production reference-control model.

| label | d_no_ip | d_ref | d_c016 | d_c017 | stddev |
| --- | ---: | ---: | ---: | ---: | ---: |
| train000_w10 | 89.90 | 69.55 | 32.32 | 58.06 | 69.95 |
| train064_w10 | 58.62 | 66.77 | 21.74 | 48.64 | 70.18 |
| train127_w10 | 55.92 | 74.30 | 40.01 | 53.22 | 71.14 |
| heldout0112_w10 | 83.72 | 111.01 | 38.21 | 78.59 | 81.21 |
| heldout0560_w10 | 89.45 | 90.94 | 39.96 | 76.16 | 81.86 |
| heldout1008_w10 | 63.22 | 76.93 | 37.76 | 54.56 | 73.58 |
