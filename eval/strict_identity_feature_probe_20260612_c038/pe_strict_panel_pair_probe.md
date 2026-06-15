# Identity Feature Probe

- Encoder: `pe`
- Manifest: `eval/strict_identity_feature_probe_20260612_c038/strict_panel_pair_probe_manifest.jsonl`
- Pairs: `128`
- Positive mean: `0.9988965038210154`
- Negative mean: `0.8585073910653591`
- Separation margin: `0.14038911275565624`
- Pairwise AUC: `0.999755859375`
- Midpoint accuracy: `0.9140625`
- Decision: `feature_separates_proxy_pairs`

Positive pairs are a weak same-SG proxy, not verified same-character labels.
This gate only decides whether an encoder deserves a stricter identity-pair run.
