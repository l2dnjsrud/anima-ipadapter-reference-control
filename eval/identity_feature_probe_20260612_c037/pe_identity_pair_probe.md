# Identity Feature Probe

- Encoder: `pe`
- Manifest: `eval/identity_feature_probe_20260612_c037/identity_pair_probe_manifest.jsonl`
- Pairs: `128`
- Positive mean: `0.8559971693903208`
- Negative mean: `0.8404057277366519`
- Separation margin: `0.01559144165366888`
- Pairwise AUC: `0.58056640625`
- Midpoint accuracy: `0.578125`
- Decision: `feature_not_sufficiently_separated`

Positive pairs are a weak same-SG proxy, not verified same-character labels.
This gate only decides whether an encoder deserves a stricter identity-pair run.
