# QwenVL c054 Visual Audit

Decision: `qwen_c052_partial_visual_improvement_metric_regression_not_quality_pass`

## Summary

- Generated images: `32`
- Blank images: `0`
- c053 best or competitive: `4/8`
- Clear c053 losses: `2/8`
- Mixed: `2/8`

c053 is a real generation improvement in several hard visual cases, especially `heldout00`, `heldout02`, and `heldout07`. It is not a full quality pass because the previous retrieval checkpoint still wins the aggregate PE and QwenVL pooled metrics, and c053 regresses on `train14` and does not clearly solve `heldout05`.

## Metric Summary

| metric | qwen_prev_retrieval_w14 | qwen_c052_w1 | qwen_c052_w14 |
|---|---:|---:|---:|
| PE mean uplift | `+0.0983` | `+0.0377` | `+0.0394` |
| PE improved rate | `0.875` | `0.625` | `0.625` |
| QwenVL mean uplift | `+0.0377` | `+0.0174` | `+0.0231` |
| QwenVL improved rate | `0.875` | `0.500` | `0.625` |

## Case Notes

| sample | visual read |
|---|---|
| `train00` | Mixed. c053 keeps robe/palette but misses the arm-crossing action. |
| `train07` | Mixed. c053 w14 is sharp and angry, but still generic. |
| `train14` | Loss. c053 drifts away from the exaggerated old bearded reference. |
| `train23` | c053 competitive/better. Glasses, hat, and scholar framing survive, fan fidelity incomplete. |
| `heldout00` | c053 better visually: pale side-facing villain, dark armor, long black hair. |
| `heldout02` | c053 w14 better: bald head, white beard, red beads, monk palette. |
| `heldout05` | Not better. Shouting profile remains mostly prompt-driven. |
| `heldout07` | c053 better visually: green demon skin and red eye survive; expression still softened. |

## Decision

Do not scale c053 as-is. It proves that the c052 reviewed identity seed can improve special visual traits, but the adapter regresses aggregate metrics. The next loop should preserve the special-trait wins while restoring metric strength, likely through a targeted continuation, negative/hard-negative contrastive mix, or a small QwenVL feature calibrator before another generation gate.
