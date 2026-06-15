# Identity Feature Probe

- Encoder: `google/siglip2-base-patch16-512`
- Manifest: `eval/reviewed_seed_feature_probe_20260612_c042/pair_probe_manifest.jsonl`
- Pairs: `7`
- Positive mean: `0.9203879237174988`
- Negative mean: `0.9175261855125427`
- Separation margin: `0.0028617382049560547`
- Pairwise AUC: `0.4166666666666667`
- Midpoint accuracy: `0.5714285714285714`
- Decision: `feature_not_sufficiently_separated`

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
