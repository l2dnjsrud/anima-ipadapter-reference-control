# Identity Feature Probe

- Encoder: `pe`
- Manifest: `training/manifests/c065_failure_attribute_pairs_20260612.jsonl`
- Pairs: `126`
- Positive mean: `0.8252357395868453`
- Negative mean: `0.8516979510821994`
- Separation margin: `-0.026462211495354104`
- Pairwise AUC: `0.41219450743260266`
- Midpoint accuracy: `0.42857142857142855`
- Decision: `feature_not_sufficiently_separated`

## Anchor Group Summaries

| group | positive mean | negative mean | margin | AUC | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| beard_headwear_crop | 0.824760840697722 | 0.8710685480724681 | -0.0463077073747461 | 0.3615702479338843 | feature_not_sufficiently_separated |
| non_human_red_pale_profile_proxy | 0.8271742008981251 | 0.82859529483886 | -0.0014210939407348633 | 0.4897959183673469 | feature_not_sufficiently_separated |
| old_face_crop | 0.8237227439880371 | 0.85464808344841 | -0.030925339460372903 | 0.385 | feature_not_sufficiently_separated |

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
