# SigLIP Token Pair Probe

- Encoder: `google/siglip2-base-patch16-512`
- Layer: `-6`
- Top-k: `64`
- Manifest: `eval/c104_expanded_qwen_target_siglip_probe_20260613/probe_manifest.jsonl`

| metric | positive mean | negative mean | margin | pairwise AUC | decision |
|---|---:|---:|---:|---:|---|
| `pooled` | 0.827202 | 0.816592 | 0.010610 | 0.505740 | `token_feature_not_sufficiently_separated` |
| `mean_max_token` | 0.804426 | 0.784451 | 0.019975 | 0.722895 | `token_feature_not_sufficiently_separated` |
| `topk_token` | 0.999084 | 0.999085 | -0.000001 | 0.476881 | `token_feature_not_sufficiently_separated` |

This probe compares hidden-token similarity, not generated image quality.
