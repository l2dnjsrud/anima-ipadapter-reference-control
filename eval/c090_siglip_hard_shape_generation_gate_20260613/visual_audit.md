# c090 Visual Audit

## Decision

`c089_partial_siglip_improvement_not_promoted_escalate_encoder_side`

## Evidence

- Generated set: 11 hard-shape samples x 4 SigLIP variants = 44 PNGs.
- Contact sheet: `eval/c090_siglip_hard_shape_generation_gate_20260613/contact_sheet_hard_shape.jpg`.
- c089 improves the prior SigLIP pilot in the shape metric: `c089_shape_w14` mean uplift `0.024921`, improved rate `0.727273`; `siglip_pilot_w14` mean uplift `-0.066240`, improved rate `0.181818`.
- c089 remains below the current hard-shape QwenVL baseline: `c087_expanded_crop_positive_w14` mean uplift `0.108954`, improved rate `0.909091`.
- Pixel audit flags 2 low-variance / blank-like collapsed outputs: `crop_pair00_no_ip` and `crop_pair00_c089_shape_w10`.

## Visual Judgment

c089 is not the same as the earlier broken SigLIP pilot. In several frog/yokai rows, especially `c089_shape_w14`, it produces coherent green character shapes instead of pure prompt drift. However, it still often turns non-human references into ordinary green humanoid faces, loses the toy/chibi silhouette, and sometimes produces repeated/collage-like figures. On `heldout07`, it does not preserve the side-profile monster identity; it becomes a different dark fantasy character.

The result is useful as evidence that the c089 PE-teacher checkpoint learned some hard-shape signal, but it is not strong enough to promote as a practical reference-control model. The next loop should escalate to stronger encoder-side/feature adaptation rather than simply relying on frozen SigLIP adapter tuning.
