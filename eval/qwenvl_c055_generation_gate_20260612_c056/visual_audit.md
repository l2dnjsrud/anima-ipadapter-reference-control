# QwenVL c056 Visual Audit

Decision: `qwen_c055_improves_c052_not_quality_pass_prev_retrieval_still_best`

## Summary

- Generated images: `40`
- Blank images: `0`
- c055 best or competitive by visual review: `6/8`
- Clear c055 weak/loss cases: `2/8`
- Metric judgment: c055 improves over c052, but previous retrieval still wins aggregate PE and QwenVL mean uplift.

c055 is a useful continuation. It broadens the c052 special-trait wins and improves QwenVL improved rate to `0.875` at weight `1.4`. The result is still not a high-quality pass because key reference-specific pose, prop, and panel details are not controlled reliably, and the earlier retrieval checkpoint remains stronger overall.

## Metric Summary

| metric | qwen_prev_retrieval_w14 | qwen_c052_w14 | qwen_c055_w1 | qwen_c055_w14 |
|---|---:|---:|---:|---:|
| PE mean uplift | `+0.0983` | `+0.0394` | `+0.0502` | `+0.0460` |
| PE improved rate | `0.875` | `0.625` | `0.750` | `0.750` |
| QwenVL mean uplift | `+0.0377` | `+0.0231` | `+0.0257` | `+0.0339` |
| QwenVL improved rate | `0.875` | `0.625` | `0.750` | `0.875` |

## Case Notes

| sample | visual read |
|---|---|
| `train00` | Loss/weak. c055 keeps black-red costume and long hair but misses the reference arm/action pose; previous retrieval and c052 are not solved either, but c055 is not a clear improvement. |
| `train07` | Loss/weak. All IP variants produce a generic angry close-up. c055 is clean but does not restore the reference-specific page framing or background cues. |
| `train14` | c055 better. c055 w1 restores the exaggerated old bearded expression better than c052 and previous retrieval, though the exact robe/background remain simplified. |
| `train23` | Competitive. c055 keeps hat, glasses, and scholar silhouette, but the fan and speech bubble are still missing; QwenVL metric prefers c055 w1, PE prefers previous retrieval. |
| `heldout00` | c055 visually strong. c055 w1/w14 preserve pale purple skin, side-facing dark-armored villain, and long black hair better than c052; QwenVL metric also prefers c055 w14. |
| `heldout02` | Competitive. c055 preserves bald head, white beard, red beads, and monk palette, but c052/previous retrieval remain close by metric and visual read. |
| `heldout05` | c055 better than c052 visually. The shouting side profile and dark hair survive more clearly, but mouth/face shape and speech bubble are still not exact. |
| `heldout07` | c055 competitive/better visually. Green demon skin and red eye survive, but snout/face silhouette remains imperfect and PE/QwenVL metrics still prefer previous retrieval. |

## Decision

Do not promote c055 as the final reference-control checkpoint. Use it as evidence that mixed continuation can recover c052 special-trait wins without completely losing metric strength. The next loop should first test runtime weight/blend combinations against the same c056 gate before spending more training time, because c055 w1.4 is already close to previous retrieval on QwenVL improved rate but still weaker on PE and some visual details.
