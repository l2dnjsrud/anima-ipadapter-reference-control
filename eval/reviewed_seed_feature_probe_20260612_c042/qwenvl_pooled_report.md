# Identity Feature Probe

- Encoder: `Qwen/Qwen3-VL-Embedding-2B`
- Manifest: `eval/reviewed_seed_feature_probe_20260612_c042/pair_probe_manifest.jsonl`
- Pairs: `7`
- Positive mean: `0.8724207431077957`
- Negative mean: `0.8484058181444804`
- Separation margin: `0.024014924963315365`
- Pairwise AUC: `0.6666666666666666`
- Midpoint accuracy: `0.7142857142857143`
- Decision: `feature_not_sufficiently_separated`

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
