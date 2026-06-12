# SigLIP Token Pair Probe

- Encoder: `google/siglip2-base-patch16-512`
- Layer: `-6`
- Top-k: `64`
- Manifest: `eval/reviewed_seed_feature_probe_20260612_c042/pair_probe_manifest.jsonl`

| metric | positive mean | negative mean | margin | pairwise AUC | decision |
|---|---:|---:|---:|---:|---|
| `pooled` | 0.514571 | 0.880698 | -0.366127 | 0.083333 | `token_feature_not_sufficiently_separated` |
| `mean_max_token` | 0.776755 | 0.733530 | 0.043225 | 0.916667 | `token_feature_not_sufficiently_separated` |
| `topk_token` | 0.994644 | 0.994293 | 0.000351 | 0.750000 | `token_feature_not_sufficiently_separated` |

This probe compares hidden-token similarity, not generated image quality.
