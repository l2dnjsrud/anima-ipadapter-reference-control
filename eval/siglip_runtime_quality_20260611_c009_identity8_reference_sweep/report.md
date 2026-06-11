# SigLIP Identity8 Reference Sweep 2026-06-11

## Decision

`reference_influence_seen_identity_generalization_incomplete`

The one-image overfit result was not just a single broken prompt path: after
training on 8 generic self-reconstruction references, the native SigLIP
checkpoint changes the generated image differently for different reference
images. It still does not reliably reproduce held-out identity.

## Setup

- Init checkpoint:
  `checkpoints/anima_siglip_ip_adapter_self512_continue_20260611.safetensors`
- Output checkpoint:
  `checkpoints/anima_siglip_ip_adapter_identity8_1024_20260611.safetensors`
- Manifest:
  `training/manifests/local_color_self_identity8_20260611.jsonl`
- Rows: 8
- Steps: 1024
- Resolution: 256
- Learning rate: 3e-5
- Runtime optimization: repeated-row feature caching for up to 16 rows.

## API Evaluation

- Server: isolated ComfyUI at `http://127.0.0.1:8115`
- Checkpoint selector:
  `anima_siglip_ip_adapter_identity8_1024_20260611.safetensors`
- Weight: 1.0
- Prompt: same generic martial-arts-master prompt used by the identity tests.
- References:
  `eval/identity8_reference_selection_20260611.json`
- Evidence:
  `eval/siglip_runtime_quality_20260611_c009_identity8_reference_sweep/contact_sheet.jpg`
  and `summary.json`

## Result

The no-IP baseline is stable and unrelated to the reference images. With the
identity8 checkpoint, outputs change by reference:

- training references can push old/bald monk features, robes, and speech-bubble
  placement into the result;
- held-out references also shift color, pose, and composition, but do not
  faithfully recover the held-out character identity;
- the model still prefers prompt-prior wuxia figures when the reference is
  outside the tiny 8-image set.

Mean abs difference versus no-IP:

| Variant | In Train | Mean Abs vs No-IP |
|---|---:|---:|
| train00_w10 | true | 70.2 |
| train03_w10 | true | 70.2 |
| train07_w10 | true | 64.4 |
| heldout0112_w10 | false | 66.1 |
| heldout0560_w10 | false | 70.0 |
| heldout1008_w10 | false | 73.6 |

## Interpretation

This is better than the earlier `self512` identity failure because reference
variation now causes distinct, visible image changes. It is still below the
goal of a high-quality generalized IP-Adapter. The next step should scale the
same cached self-reconstruction setup from 8 references to a larger, more
balanced identity/color set and add a held-out validation sheet before pushing
any checkpoint as usable.
