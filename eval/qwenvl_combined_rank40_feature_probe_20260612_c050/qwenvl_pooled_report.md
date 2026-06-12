# Identity Feature Probe

- Encoder: `Qwen/Qwen3-VL-Embedding-2B`
- Manifest: `eval/qwenvl_combined_rank40_feature_probe_20260612_c050/pair_probe_manifest.jsonl`
- Pairs: `36`
- Positive mean: `0.90378548283326`
- Negative mean: `0.8221866102779612`
- Separation margin: `0.08159887255529874`
- Pairwise AUC: `0.9009287925696594`
- Midpoint accuracy: `0.8611111111111112`
- Decision: `feature_separates_proxy_pairs`

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
