# QwenVL c057 Runtime Weight/Blend Visual Audit

Decision: `runtime_blend_prev14_c05504_best_so_far_larger_gate_required`

## Summary

- Generated images: `56`
- Blank images: `0`
- Best single runtime recipe: `blend_prev14_c05504`
- PE mean uplift winner: `blend_prev14_c05504`
- QwenVL mean uplift winner: `prev_w14`, by only `0.0002`
- Final high-quality pass: `not_yet`

c057 is the strongest runtime result so far. Applying the previous retrieval checkpoint at `1.4` and then applying c055 at `0.4` gives the best PE score and keeps QwenVL similarity essentially tied with the previous retrieval baseline. This is a useful ComfyUI recipe candidate, but not the end state because exact reference pose, props, and panel-specific details are still inconsistent.

## Metric Summary

| metric | prev_w14 | c055_w06 | c055_w08 | c055_w12 | blend_prev10_c05506 | blend_prev14_c05504 |
|---|---:|---:|---:|---:|---:|---:|
| PE mean uplift | `+0.0983` | `+0.0543` | `+0.0282` | `+0.0583` | `+0.0768` | `+0.1064` |
| PE improved rate | `0.875` | `0.875` | `0.625` | `0.750` | `0.750` | `0.875` |
| QwenVL mean uplift | `+0.0377` | `+0.0096` | `+0.0136` | `+0.0289` | `+0.0357` | `+0.0375` |
| QwenVL improved rate | `0.875` | `0.500` | `0.750` | `0.750` | `0.875` | `0.875` |

## Case Notes

| sample | visual read |
|---|---|
| `train00` | Blend variants restore more arm/action pressure than `prev_w14`, but hands and exact pose remain imperfect. |
| `train07` | Weak. All variants remain generic angry close-ups; runtime blending does not recover page-specific framing. |
| `train14` | `c055_w06` is best for the exaggerated old bearded face. `blend_prev14_c05504` is stable but less expressive. |
| `train23` | `c055_w06/w08` restore the fan better than the blend. The best global recipe still misses this prop. |
| `heldout00` | c055 and blend variants improve side-facing dark-armored villain traits over the previous retrieval baseline. |
| `heldout02` | `blend_prev14_c05504` is strong: bald head, white beard, and monk silhouette survive cleanly. |
| `heldout05` | `blend_prev14_c05504` wins PE and gives a strong shouting side profile, but speech bubble and exact beard/mouth shape remain off. |
| `heldout07` | `blend_prev14_c05504` wins QwenVL for this row and keeps green skin/red eye while improving full-head silhouette. |

## Decision

Use `blend_prev14_c05504` as the current best runtime recipe candidate. Do not call it a finished high-quality reference-control checkpoint yet. The next loop should either run this recipe on a larger heldout suite and make a UI workflow, or train/distill a single adapter that preserves the blend's aggregate stability while recovering c055's special-case trait wins.
