# Identity Feature Probe

- Encoder: `Qwen/Qwen3-VL-Embedding-2B`
- Manifest: `eval/qwenvl_combined_seed_feature_probe_20260612_c048/pair_probe_manifest.jsonl`
- Pairs: `33`
- Positive mean: `0.9053295155366262`
- Negative mean: `0.817700719833374`
- Separation margin: `0.08762879570325222`
- Pairwise AUC: `0.9074074074074074`
- Midpoint accuracy: `0.8181818181818182`
- Decision: `feature_separates_proxy_pairs`

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
