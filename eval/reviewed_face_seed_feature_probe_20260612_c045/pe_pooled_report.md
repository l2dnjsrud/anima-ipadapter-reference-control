# Identity Feature Probe

- Encoder: `pe`
- Manifest: `eval/reviewed_face_seed_feature_probe_20260612_c045/pair_probe_manifest.jsonl`
- Pairs: `23`
- Positive mean: `0.9253924265503883`
- Negative mean: `0.8806189815203349`
- Separation margin: `0.04477344503005343`
- Pairwise AUC: `0.7833333333333333`
- Midpoint accuracy: `0.6956521739130435`
- Decision: `feature_not_sufficiently_separated`

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
