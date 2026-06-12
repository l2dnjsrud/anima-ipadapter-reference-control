# Identity Feature Probe

- Encoder: `google/siglip2-base-patch16-512`
- Manifest: `training/manifests/c065_failure_attribute_pairs_20260612.jsonl`
- Pairs: `126`
- Positive mean: `0.8891768247362167`
- Negative mean: `0.8792919762550838`
- Separation margin: `0.009884848481132913`
- Pairwise AUC: `0.5782312925170068`
- Midpoint accuracy: `0.5952380952380952`
- Decision: `feature_not_sufficiently_separated`

## Anchor Group Summaries

| group | positive mean | negative mean | margin | AUC | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| beard_headwear_crop | 0.8937719653953206 | 0.8789924383163452 | 0.01477952707897534 | 0.5888429752066116 | feature_not_sufficiently_separated |
| non_human_red_pale_profile_proxy | 0.8791367184548151 | 0.8803147276242574 | -0.001178009169442329 | 0.5034013605442177 | feature_not_sufficiently_separated |
| old_face_crop | 0.8946642816066742 | 0.8785475790500641 | 0.01611670255661013 | 0.6225 | feature_not_sufficiently_separated |

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
