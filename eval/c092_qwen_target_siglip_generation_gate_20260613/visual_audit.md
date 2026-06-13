# c092 Qwen-Target SigLIP Visual Audit

Contact sheet: `eval/c092_qwen_target_siglip_generation_gate_20260613/contact_sheet_hard_shape.jpg`

## Verdict

`c092_qwen_target_w14` is not a final quality promotion, but it is the strongest SigLIP hard-shape result so far. It clearly improves over c089/c091 on aggregate metrics and visible green-character control, while still failing the harder requirement: preserving non-human/chibi body shape and heldout side-profile monster identity.

## What Improved

- c092 beats c089 and c091 by a large margin: c092 w14 mean uplift `0.08526816531384505` vs c089 `0.024921435615903164` and c091 `0.024088173699299265`.
- c092 w14 has improved rate `1.0`, so every sample is above its no-IP baseline by the shape metric.
- On crop pairs 00-05 and 07-09, c092 produces a strong green face/character signal instead of the prior SigLIP over-zoom, blur, or unrelated wuxia portrait drift.
- c092 w10 is also useful: mean uplift `0.07382548763785077`, improved rate `0.9090909090909091`. This suggests the Qwen-target supervision itself is doing useful work, not just high adapter weight.

## What Failed

- c092 does not match the Qwen teacher baseline: Qwen c087 mean uplift is `0.10895440559772807`, still higher than c092 w14.
- The contact sheet shows a strong collapse toward bald green human faces. This improves broad green/non-human color but reduces frog/chibi body diversity.
- `heldout07` was excluded from training and remains weak. c092 w14 heldout uplift is only `0.0009986629992987384`, below c089 `0.019646282913627522`, c091 `0.020851579146191956`, and Qwen c087 `0.09803324868423646`.
- c092 still does not preserve the reference's monster side-profile, red eye, and exaggerated jaw silhouette.

## Decision

`c092_improves_c089_but_not_qwen_baseline`

c092 proves that Qwen-generated target images are a much stronger SigLIP supervision signal than the c091 feature-calibrator-only route. The next loop should keep the Qwen-target direction, but add anti-collapse pressure: more diverse Qwen targets, heldout-free hard negatives, body/silhouette preservation loss, or a teacher mix that does not reduce all references to green human face templates.
