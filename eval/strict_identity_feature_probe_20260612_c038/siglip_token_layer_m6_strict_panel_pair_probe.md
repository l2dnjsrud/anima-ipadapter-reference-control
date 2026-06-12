# SigLIP Token Pair Probe

- Encoder: `google/siglip2-base-patch16-512`
- Layer: `-6`
- Top-k: `64`
- Manifest: `eval/strict_identity_feature_probe_20260612_c038/strict_panel_pair_probe_manifest.jsonl`

| metric | positive mean | negative mean | margin | pairwise AUC | decision |
|---|---:|---:|---:|---:|---|
| `pooled` | 0.998710 | 0.524800 | 0.473910 | 0.999756 | `token_feature_separates_pairs` |
| `mean_max_token` | 0.995232 | 0.715541 | 0.279691 | 1.000000 | `token_feature_separates_pairs` |
| `topk_token` | 0.999892 | 0.994004 | 0.005889 | 0.997314 | `token_feature_not_sufficiently_separated` |

This probe compares hidden-token similarity, not generated image quality.
