# SigLIP Token Pair Probe

- Encoder: `google/siglip2-base-patch16-512`
- Layer: `-6`
- Top-k: `64`
- Manifest: `eval/reviewed_face_seed_feature_probe_20260612_c045/pair_probe_manifest.jsonl`

| metric | positive mean | negative mean | margin | pairwise AUC | decision |
|---|---:|---:|---:|---:|---|
| `pooled` | 0.632742 | 0.624929 | 0.007813 | 0.491667 | `token_feature_not_sufficiently_separated` |
| `mean_max_token` | 0.765671 | 0.736944 | 0.028728 | 0.708333 | `token_feature_not_sufficiently_separated` |
| `topk_token` | 0.994048 | 0.996207 | -0.002159 | 0.291667 | `token_feature_not_sufficiently_separated` |

This probe compares hidden-token similarity, not generated image quality.
