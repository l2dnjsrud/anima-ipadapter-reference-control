# c091 SigLIP Feature-Calibrator Visual Audit

Contact sheet: `eval/c091_siglip_feature_calibrator_hard_shape_gate_20260613/contact_sheet_hard_shape.jpg`

## Verdict

`c091_feature_calibrator_w14` is not promoted. It recovers the same broad green-character signal as c089, but it does not materially improve reference fidelity and remains far below the QwenVL expanded target-positive baseline.

## What Improved

- Compared with `siglip_pilot_w14`, c091 avoids several over-zoomed or blurred face outputs and more often produces a readable green character.
- `c091_feature_calibrator_w14` is visually close to `c089_shape_w14`, especially on frog/yokai crop pairs where the image remains character-shaped instead of drifting into unrelated wuxia portraits.

## What Failed

- `c091_feature_calibrator_w10` is too weak/unstable. `crop_pair00_c091_feature_calibrator_w10` collapses into a low-variance green block.
- `c091_feature_calibrator_w14` does not beat c089 numerically: c091 mean uplift is `0.024088173699299265`, while c089 is `0.024921435615903164`.
- The QwenVL baseline `c087_expanded_crop_positive_w14` remains clearly stronger with mean uplift `0.10895440559772807` and improved rate `0.9090909090909091`.
- Chibi/frog mascot proportions, exact non-human body shape, and side-profile monster identity are still not stable. `heldout07` continues to drift away from the reference silhouette.

## Decision

`c091_matches_c089_not_qwen_baseline`

The feature-calibrator-only route is useful as a trainability proof, but it is not enough for the high-quality reference-control target. The next loop should stop treating frozen SigLIP feature calibration as the main lever and move to a stronger encoder-side checkpoint, supervised shape/identity feature adaptation, or a QwenVL/SigLIP hybrid route with hard-shape positive/negative supervision.
