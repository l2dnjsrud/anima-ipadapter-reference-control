# SigLIP Token Pair Probe

- Encoder: `google/siglip2-base-patch16-512`
- Layer: `-1`
- Top-k: `64`
- Manifest: `eval/strict_identity_feature_probe_20260612_c038/strict_panel_pair_probe_manifest.jsonl`

| metric | positive mean | negative mean | margin | pairwise AUC | decision |
|---|---:|---:|---:|---:|---|
| `pooled` | 0.999446 | 0.865676 | 0.133770 | 1.000000 | `token_feature_separates_pairs` |
| `mean_max_token` | 0.994342 | 0.677364 | 0.316978 | 1.000000 | `token_feature_separates_pairs` |
| `topk_token` | 0.999956 | 0.998010 | 0.001946 | 0.991699 | `token_feature_not_sufficiently_separated` |

This probe compares hidden-token similarity, not generated image quality.
