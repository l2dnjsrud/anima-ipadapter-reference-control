# SigLIP Calibrated Contrastive 576-Step Neutral-Prompt Sweep 2026-06-11

## Decision

`longer_calibrated_contrastive_training_overfits_scene_average`

- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c017_calibrated_contrastive0576_neutral_prompt/contact_sheet.jpg`
- Summary: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c017_calibrated_contrastive0576_neutral_prompt/summary.json`
- Checkpoint: `anima_siglip_ip_adapter_identity128_calibrated_contrastive_0576_20260611.safetensors`
- Init checkpoint: `anima_siglip_ip_adapter_identity128_calibrated_contrastive_0064_20260611.safetensors`

| Reference | Mean Abs vs No-IP | Mean Abs vs Ref | Mean Abs vs c014 | Mean Abs vs c016 | Notes |
|---|---:|---:|---:|---:|---|
| train000_w10 | 80.4 | 63.9 | 54.3 | 53.0 | collapses to sunset crowd/sky, not the monk reference |
| train064_w10 | 49.6 | 55.3 | 43.6 | 44.5 | numeric dRef improves, visual layout drifts into dim room |
| train127_w10 | 44.8 | 64.9 | 35.7 | 42.9 | remains top-down court/table-like, but blurred and generic |
| heldout0112_w10 | 54.3 | 110.6 | 66.8 | 80.8 | loses the promising c016 portrait signal |
| heldout0560_w10 | 59.9 | 110.0 | 68.6 | 75.8 | moves away from reference profile into generic group scene |
| heldout1008_w10 | 68.7 | 70.6 | 49.8 | 55.1 | blue palette persists, identity/layout remain weak |

## Visual Review

The 576-step continuation proves the calibration path keeps training and moves
both adapter and calibration tensors. It does not improve the real quality gate.
Compared with c016, longer training often blurs faces, increases average group
scenes, and weakens held-out identity. The best current calibrated evidence is
c016, not c017.

Next training needs a stronger objective or encoder signal rather than simply
more steps on the same frozen-SigLIP contrastive setup.
