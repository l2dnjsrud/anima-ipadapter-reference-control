# SigLIP Color Pilot Evaluation

**Decision:** `scale_after_siglip_workflow_eval`
**Quality proven:** `False`

## Method

This is a proxy evaluation, not a final visual quality pass. The pilot checkpoint
is compared against the one-step SigLIP smoke checkpoint by tensor deltas, then
anchored to the existing PE/no-IP contact-sheet baseline. A real SigLIP UI/API
image workflow is still required before calling the model usable.

## Checkpoint Delta

- Smoke: `checkpoints/anima_siglip_ip_adapter_smoke_20260610.safetensors`
- Pilot: `checkpoints/anima_siglip_ip_adapter_pilot_20260610.safetensors`
- Key match: `True`
- Common tensors: `255`
- Changed tensors: `142`
- Relative L2 delta: `0.0008295683`
- Mean abs delta: `0.0000040928`
- Max abs delta: `0.0002288818`

## PE Baseline Anchor

- Summary: `eval/comfy_pe_full_contactsheet_20260610/summary.json`
- PE/no-IP contact-sheet pass: `True`
- Best PE scale: `1.0`
- Generated images: `40`
- Best mean uplift: `0.0937323346734047`
- Best improved rate: `0.875`

## Scale Decision

- Scale next: `True`
- Reason: pilot moved from smoke, but SigLIP image-generation workflow is not evaluated yet

## Required Next Gate

Create a native SigLIP ComfyUI/API workflow and generate contact sheets against
no-IP and PE-Core baselines before any `quality` or `ready to use` claim.
