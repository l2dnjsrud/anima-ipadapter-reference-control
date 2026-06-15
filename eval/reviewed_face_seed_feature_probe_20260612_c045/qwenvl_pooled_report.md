# Identity Feature Probe

- Encoder: `Qwen/Qwen3-VL-Embedding-2B`
- Manifest: `eval/reviewed_face_seed_feature_probe_20260612_c045/pair_probe_manifest.jsonl`
- Pairs: `23`
- Positive mean: `0.8839093372225761`
- Negative mean: `0.817700719833374`
- Separation margin: `0.06620861738920214`
- Pairwise AUC: `0.7916666666666666`
- Midpoint accuracy: `0.6521739130434783`
- Decision: `feature_separates_proxy_pairs`

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
