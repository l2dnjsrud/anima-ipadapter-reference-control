# Identity Feature Probe

- Encoder: `Qwen/Qwen3-VL-Embedding-2B`
- Manifest: `training/manifests/c065_failure_attribute_pairs_20260612.jsonl`
- Pairs: `126`
- Positive mean: `0.7481812390070113`
- Negative mean: `0.7534571223788791`
- Separation margin: `-0.0052758833718677955`
- Pairwise AUC: `0.46056941295036535`
- Midpoint accuracy: `0.4126984126984127`
- Decision: `feature_not_sufficiently_separated`

## Anchor Group Summaries

| group | positive mean | negative mean | margin | AUC | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| beard_headwear_crop | 0.7520805190909993 | 0.7486226043917916 | 0.0034579146992076426 | 0.4772727272727273 | feature_not_sufficiently_separated |
| non_human_red_pale_profile_proxy | 0.7358976063274202 | 0.7573974444752648 | -0.02149983814784462 | 0.41496598639455784 | feature_not_sufficiently_separated |
| old_face_crop | 0.7567898452281951 | 0.7546377539634704 | 0.0021520912647247092 | 0.51 | feature_not_sufficiently_separated |

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
