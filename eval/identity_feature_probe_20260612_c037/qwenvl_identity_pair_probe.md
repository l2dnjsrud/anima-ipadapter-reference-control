# Identity Feature Probe

- Encoder: `Qwen/Qwen3-VL-Embedding-2B`
- Manifest: `eval/identity_feature_probe_20260612_c037/identity_pair_probe_manifest.jsonl`
- Pairs: `128`
- Positive mean: `0.7893099104985595`
- Negative mean: `0.7567050401121378`
- Separation margin: `0.03260487038642168`
- Pairwise AUC: `0.59130859375`
- Midpoint accuracy: `0.5703125`
- Decision: `feature_not_sufficiently_separated`

Positive pairs are a weak same-SG proxy, not verified same-character labels.
This gate only decides whether an encoder deserves a stricter identity-pair run.
