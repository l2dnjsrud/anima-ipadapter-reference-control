# Identity Feature Probe

- Encoder: `google/siglip2-base-patch16-512`
- Manifest: `eval/strict_identity_feature_probe_20260612_c038/strict_panel_pair_probe_manifest.jsonl`
- Pairs: `128`
- Positive mean: `0.9995568273589015`
- Negative mean: `0.8937714975327253`
- Separation margin: `0.10578532982617617`
- Pairwise AUC: `1.0`
- Midpoint accuracy: `0.96875`
- Decision: `feature_separates_proxy_pairs`

Positive pairs are a weak same-SG proxy, not verified same-character labels.
This gate only decides whether an encoder deserves a stricter identity-pair run.
