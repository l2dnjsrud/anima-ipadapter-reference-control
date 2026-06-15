# Identity Feature Probe

- Encoder: `Qwen/Qwen3-VL-Embedding-2B`
- Manifest: `training/manifests/c066_direct_green_non_human_pairs_20260612.jsonl`
- Pairs: `156`
- Positive mean: `0.7372685724344009`
- Negative mean: `0.7379715748322315`
- Separation margin: `-0.0007030023978306099`
- Pairwise AUC: `0.5096153846153846`
- Midpoint accuracy: `0.48717948717948717`
- Decision: `feature_not_sufficiently_separated`

## Anchor Group Summaries

| group | positive mean | negative mean | margin | AUC | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| direct_green_pixel_candidate | 0.7157958835363388 | 0.6810271546244622 | 0.034768728911876656 | 0.54375 | feature_not_sufficiently_separated |
| fang_profile_proxy | 0.7791795169605928 | 0.8703067618257859 | -0.09112724486519308 | 0.31833910034602075 | feature_not_sufficiently_separated |
| pale_non_human_proxy | 0.7399252382191744 | 0.7313991568305276 | 0.008526081388646856 | 0.512396694214876 | feature_not_sufficiently_separated |
| red_eye_proxy | 0.7489883899688721 | 0.7480090975761413 | 0.0009792923927307573 | 0.55 | feature_not_sufficiently_separated |

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
