# Identity Feature Probe

- Encoder: `Qwen/Qwen3-VL-Embedding-2B`
- Manifest: `eval/qwenvl_combined_diverse_feature_probe_20260612_c052/pair_probe_manifest.jsonl`
- Pairs: `58`
- Positive mean: `0.8977140311537117`
- Negative mean: `0.8255111862873209`
- Separation margin: `0.07220284486639084`
- Pairwise AUC: `0.9131985731272295`
- Midpoint accuracy: `0.8620689655172413`
- Decision: `feature_separates_proxy_pairs`

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
