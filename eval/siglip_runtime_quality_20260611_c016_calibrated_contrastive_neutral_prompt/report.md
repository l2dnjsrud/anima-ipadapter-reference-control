# SigLIP Calibrated Contrastive 64-Step Neutral-Prompt Sweep 2026-06-11

## Decision

`calibration_path_executes_and_improves_some_rows_but_not_quality_pass`

- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c016_calibrated_contrastive_neutral_prompt/contact_sheet.jpg`
- Summary: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c016_calibrated_contrastive_neutral_prompt/summary.json`
- Checkpoint: `anima_siglip_ip_adapter_identity128_calibrated_contrastive_0064_20260611.safetensors`
- Base checkpoint: `anima_siglip_ip_adapter_identity128_contrastive_0512_20260611.safetensors`

| Reference | Mean Abs vs No-IP | Mean Abs vs Ref | Mean Abs vs c014 | Notes |
|---|---:|---:|---:|---|
| train000_w10 | 77.4 | 60.7 | 23.1 | stronger reference color pressure, but washed-out sky/crowd scene |
| train064_w10 | 55.0 | 59.5 | 29.4 | improved character staging vs c014, still not the reference layout |
| train127_w10 | 48.1 | 67.6 | 19.3 | keeps top-down room/table pressure, identity still weak |
| heldout0112_w10 | 86.3 | 110.0 | 53.5 | cleaner face output, but not the side-profile reference identity |
| heldout0560_w10 | 91.6 | 89.2 | 56.1 | cleaner portrait pressure, not the female profile reference |
| heldout1008_w10 | 66.3 | 69.5 | 41.0 | blue/court palette remains weakly reflected |

## Visual Review

The calibrated checkpoint loads through the normal SigLIP ComfyUI loader and
produces outputs distinct from both no-IP and c014. The 64-step run is a useful
positive signal: the added `feature_calibrator.*` tensors move, the API workflow
executes, and several outputs are cleaner than c014.

This is not a usable quality pass. The result still transfers broad palette,
scene, or crowd pressure more reliably than identity, exact pose, or panel
layout. Treat this as evidence that trainable feature calibration is worth
continuing, not as evidence that the current checkpoint is ready.
