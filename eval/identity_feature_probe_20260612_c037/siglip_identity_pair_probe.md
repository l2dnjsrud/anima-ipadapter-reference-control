# Identity Feature Probe

- Encoder: `google/siglip2-base-patch16-512`
- Manifest: `eval/identity_feature_probe_20260612_c037/identity_pair_probe_manifest.jsonl`
- Pairs: `128`
- Positive mean: `0.8932049730792642`
- Negative mean: `0.8800096744671464`
- Separation margin: `0.013195298612117767`
- Pairwise AUC: `0.575927734375`
- Midpoint accuracy: `0.5625`
- Decision: `feature_not_sufficiently_separated`

Positive pairs are a weak same-SG proxy, not verified same-character labels.
This gate only decides whether an encoder deserves a stricter identity-pair run.
