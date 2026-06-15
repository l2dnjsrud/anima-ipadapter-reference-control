# c068 Reviewed Attribute Label Seed

- Source c067 commit: `9b53041`
- Reviewed rows: `48`
- Heldout rows used: `0`
- Direct-green target positives: `0`
- Decision: `direct_green_reviewed_seed_insufficient_new_annotation_required`

## Label Counts

- `false_positive_background_object`: `11`
- `false_positive_human_face`: `9`
- `false_positive_human_old_face`: `4`
- `false_positive_red_eye_human`: `1`
- `negative_anchor`: `8`
- `target_positive`: `1`
- `useful_proxy_positive`: `14`

The reviewed direct-green/non-human seed is not sufficient for encoder-side training unless direct-green target positives are at least 4.
Next decision: do not train encoder-side positives from this seed; collect new captioned or manually reviewed direct-green/non-human data first.
