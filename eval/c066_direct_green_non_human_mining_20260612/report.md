# c066 Direct Green / Non-Human Mining Report

- Decision: `direct_green_data_insufficient_attribute_teacher_required`
- Candidate manifest: `training/manifests/c066_direct_green_non_human_candidates_20260612.jsonl`
- Candidate summary: `training/manifests/c066_direct_green_non_human_candidates_20260612.summary.json`
- Pair manifest: `training/manifests/c066_direct_green_non_human_pairs_20260612.jsonl`
- Review sheet: `eval/c066_direct_green_non_human_mining_20260612/green_top16_probe_sheet.jpg`

## Candidate Mining

- Total candidates: `120`
- Positive candidates: `78`
- Negative candidates: `42`
- Direct green character attribute positives: `0`
- Direct green pixel candidates: `40`
- Non-human proxy positives: `38`
- Heldout rows used: `0`
- Missing paths: `0`
- Sidecar caption keyword hits: `0`

Source bucket counts:

| bucket | count |
| --- | ---: |
| direct_green_pixel_candidate | 40 |
| fang_profile_proxy | 17 |
| human_negative | 20 |
| old_headwear_negative | 22 |
| pale_non_human_proxy | 11 |
| red_eye_proxy | 10 |

## Feature Separation

| encoder | margin | AUC | midpoint | decision |
| --- | ---: | ---: | ---: | --- |
| qwenvl | -0.000703 | 0.509615 | 0.487179 | feature_not_sufficiently_separated |
| siglip | 0.008130 | 0.537229 | 0.532051 | feature_not_sufficiently_separated |
| pe | 0.009635 | 0.523915 | 0.506410 | feature_not_sufficiently_separated |

## Bucket Notes

### qwenvl

| bucket | margin | AUC | decision |
| --- | ---: | ---: | --- |
| direct_green_pixel_candidate | 0.034769 | 0.543750 | feature_not_sufficiently_separated |
| fang_profile_proxy | -0.091127 | 0.318339 | feature_not_sufficiently_separated |
| pale_non_human_proxy | 0.008526 | 0.512397 | feature_not_sufficiently_separated |
| red_eye_proxy | 0.000979 | 0.550000 | feature_not_sufficiently_separated |

### siglip

| bucket | margin | AUC | decision |
| --- | ---: | ---: | --- |
| direct_green_pixel_candidate | 0.029138 | 0.600000 | feature_not_sufficiently_separated |
| fang_profile_proxy | -0.042651 | 0.269896 | feature_not_sufficiently_separated |
| pale_non_human_proxy | 0.011578 | 0.586777 | feature_not_sufficiently_separated |
| red_eye_proxy | 0.006634 | 0.560000 | feature_not_sufficiently_separated |

### pe

| bucket | margin | AUC | decision |
| --- | ---: | ---: | --- |
| direct_green_pixel_candidate | 0.039530 | 0.571250 | feature_not_sufficiently_separated |
| fang_profile_proxy | -0.051312 | 0.290657 | feature_not_sufficiently_separated |
| pale_non_human_proxy | 0.009071 | 0.652893 | feature_not_sufficiently_separated |
| red_eye_proxy | -0.005712 | 0.550000 | feature_not_sufficiently_separated |

## Interpretation

The local color dataset does contain images with green pixels, but the top reviewed candidates are dominated by leaves, backgrounds, rooms, cups, and other non-character regions. The only explicit `green monster face with red glowing eye` selected-attribute sample remains in clean32 heldout and was excluded.

The green-pixel bucket also does not pass the feature-separation gate. Its best margin is PE `0.039530`, below the `0.05` threshold, and its best AUC is SigLIP2 `0.600000`, below the `0.70` threshold. Red-eye, pale, and fang/profile proxy buckets also fail across QwenVL, SigLIP2, and PE.

Therefore c066 does not justify launching an encoder-side checkpoint from the mined data. The next loop should build an explicit attribute teacher/reranker or run a direct caption/annotation stage to obtain true train-split green non-human positives before any expensive checkpoint training.
