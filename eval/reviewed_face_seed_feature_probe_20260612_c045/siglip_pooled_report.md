# Identity Feature Probe

- Encoder: `google/siglip2-base-patch16-512`
- Manifest: `eval/reviewed_face_seed_feature_probe_20260612_c045/pair_probe_manifest.jsonl`
- Pairs: `23`
- Positive mean: `0.9231412187218666`
- Negative mean: `0.9072680036226909`
- Separation margin: `0.01587321509917572`
- Pairwise AUC: `0.65`
- Midpoint accuracy: `0.6086956521739131`
- Decision: `feature_not_sufficiently_separated`

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
