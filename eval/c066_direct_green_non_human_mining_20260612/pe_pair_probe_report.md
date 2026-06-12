# Identity Feature Probe

- Encoder: `pe`
- Manifest: `training/manifests/c066_direct_green_non_human_pairs_20260612.jsonl`
- Pairs: `156`
- Positive mean: `0.8226156303515801`
- Negative mean: `0.8129802535359676`
- Separation margin: `0.00963537681561255`
- Pairwise AUC: `0.5239151873767258`
- Midpoint accuracy: `0.5064102564102564`
- Decision: `feature_not_sufficiently_separated`

## Anchor Group Summaries

| group | positive mean | negative mean | margin | AUC | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| direct_green_pixel_candidate | 0.78487369120121 | 0.7453434702008963 | 0.03953022100031367 | 0.57125 | feature_not_sufficiently_separated |
| fang_profile_proxy | 0.8845725410124835 | 0.9358849841005662 | -0.05131244308808269 | 0.2906574394463668 | feature_not_sufficiently_separated |
| pale_non_human_proxy | 0.8290547078306024 | 0.8199832439422607 | 0.00907146388834168 | 0.6528925619834711 | feature_not_sufficiently_separated |
| red_eye_proxy | 0.8611736536026001 | 0.8668860554695129 | -0.005712401866912797 | 0.55 | feature_not_sufficiently_separated |

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
