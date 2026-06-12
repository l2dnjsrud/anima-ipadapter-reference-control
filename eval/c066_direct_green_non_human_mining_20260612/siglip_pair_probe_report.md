# Identity Feature Probe

- Encoder: `google/siglip2-base-patch16-512`
- Manifest: `training/manifests/c066_direct_green_non_human_pairs_20260612.jsonl`
- Pairs: `156`
- Positive mean: `0.8758517564871372`
- Negative mean: `0.8677215522680527`
- Separation margin: `0.008130204219084503`
- Pairwise AUC: `0.5372287968441815`
- Midpoint accuracy: `0.532051282051282`
- Decision: `feature_not_sufficiently_separated`

## Anchor Group Summaries

| group | positive mean | negative mean | margin | AUC | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| direct_green_pixel_candidate | 0.8583049952983857 | 0.8291669651865959 | 0.029138030111789726 | 0.6 | feature_not_sufficiently_separated |
| fang_profile_proxy | 0.9023895193548763 | 0.9450406782767352 | -0.04265115892185889 | 0.2698961937716263 | feature_not_sufficiently_separated |
| pale_non_human_proxy | 0.8770068071105264 | 0.8654286211187189 | 0.011578185991807488 | 0.5867768595041323 | feature_not_sufficiently_separated |
| red_eye_proxy | 0.8996540486812592 | 0.8930196106433869 | 0.006634438037872337 | 0.56 | feature_not_sufficiently_separated |

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
