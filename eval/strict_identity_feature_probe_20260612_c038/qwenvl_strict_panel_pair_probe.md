# Identity Feature Probe

- Encoder: `Qwen/Qwen3-VL-Embedding-2B`
- Manifest: `eval/strict_identity_feature_probe_20260612_c038/strict_panel_pair_probe_manifest.jsonl`
- Pairs: `128`
- Positive mean: `0.9991699857637286`
- Negative mean: `0.7930740043520927`
- Separation margin: `0.20609598141163588`
- Pairwise AUC: `1.0`
- Midpoint accuracy: `0.96875`
- Decision: `feature_separates_proxy_pairs`

Positive pairs are a weak same-SG proxy, not verified same-character labels.
This gate only decides whether an encoder deserves a stricter identity-pair run.
