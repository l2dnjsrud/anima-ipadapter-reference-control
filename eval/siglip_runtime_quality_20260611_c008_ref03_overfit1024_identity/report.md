# SigLIP Ref03 One-Image Overfit Gate 2026-06-11

## Decision

`overfit_pass_generalization_required`

The native SigLIP runtime and adapter architecture can carry visible reference
identity when the training task is reduced to a one-image self-reconstruction
overfit. This means the SigLIP route is not proven impossible. The current
general checkpoints still are not ready-to-trust reference-control models.

## Setup

- Init checkpoint:
  `checkpoints/anima_siglip_ip_adapter_self512_continue_20260611.safetensors`
- Output checkpoint:
  `checkpoints/anima_siglip_ip_adapter_ref03_overfit1024_20260611.safetensors`
- Training rows: 1
- Steps: 1024
- Resolution: 256
- Learning rate: 5e-5
- Reference/target image:
  `/data/ai/comfyui02/input/codex_contact_ref03.png`
- Training/eval prompt intentionally removed direct identity tokens such as
  old, bald, white beard, and prayer beads.

## API Evaluation

- Server: isolated ComfyUI at `http://127.0.0.1:8115`
- Custom node source:
  `/home/wktwin/anima-ipadapter-reference-control`
- Model selector checkpoint:
  `anima_siglip_ip_adapter_ref03_overfit1024_20260611.safetensors`
- Evidence:
  `eval/siglip_runtime_quality_20260611_c008_ref03_overfit1024_identity/contact_sheet.jpg`
  and `summary.json`

## Result

The no-IP baseline generated unrelated martial-arts figures. The overfit
checkpoint recovered the bald monk face, red prayer beads, robe color, speech
bubble, and panel crop even though those identity details were absent from the
prompt.

Best numeric reference distances:

| Variant | Mean Abs vs No-IP | Mean Abs vs Reference |
|---|---:|---:|
| no_ip | 0.0 | 108.2 |
| overfit_w06 | 97.0 | 61.4 |
| overfit_w08 | 97.2 | 52.0 |
| overfit_w10 | 96.0 | 46.9 |
| overfit_w12 | 94.8 | 48.7 |

Visual sweet spot:

- `overfit_w08` to `overfit_w12` strongly recover the reference.
- `overfit_w10` has the lowest pixel distance to the reference in this single
  run.
- Higher weights become unstable; `overfit_w24` and `overfit_w32` collapse or
  blur.

## Interpretation

This proves the current native SigLIP path can learn reference identity in a
controlled overfit setting. The failure of `color64`, `self64`, and `self512`
therefore points to insufficient or incorrect training data, insufficient
training duration, uncached slow training, scale instability, or lack of a
proper identity-aware validation set rather than a hard runtime impossibility.

Next work should focus on cached feature training and a small multi-reference
identity set before any broad Wenaka-scale launch.
