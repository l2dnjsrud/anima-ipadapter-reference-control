# c067 Attribute Teacher / Reranker Seed

- Candidate count: `72`
- Query count: `6`
- Heldout rows used: `0`
- Scorer status: `scored`
- Decision: `candidate_teacher_seed_requires_manual_review`

## Attribute Queries

- `direct_green_non_human_face` (target_positive): green non-human demon face with red glowing eye, colored skin, monster portrait
- `red_glowing_eye` (target_positive): single red glowing demonic eye on a fantasy martial arts character
- `side_profile_silhouette` (target_positive): side profile portrait silhouette, sharp nose profile, character face in profile
- `beard_headwear_crop` (target_positive): old bearded martial arts master or court official with black hat, close-up crop
- `human_negative` (negative_anchor): ordinary human wuxia martial artist face, natural skin, normal human portrait
- `background_object_green` (false_positive_guard): green background objects, leaves, plants, palace decor, not a green character

## Scoring Result

- Scorer: `Qwen/Qwen3-VL-Embedding-2B` via local image-text retrieval scorer.
- Score rows: `432` (`72` candidates x `6` queries).
- Top-k per query: `8`.
- Review sheet: `attribute_review_sheet.jpg`.
- Direct-green teacher candidates counted by score guard: `6`.

## Visual Audit

`direct_green_non_human_face` did not produce a reliable automatic positive set. The top
rows are mostly old human faces with strong shadows/headwear, a red-eyed monk-like
character, ordinary human close-ups, tea cups, or panels where green is background/object
color. This repeats the c066 failure mode: green pixel evidence and generic VL retrieval do
not cleanly isolate a green non-human character face.

The query groups for `red_glowing_eye`, `side_profile_silhouette`, and
`beard_headwear_crop` are more useful as review queues because their top rows include
visibly relevant faces and crops. `background_object_green` correctly collects many
false-positive green objects/backgrounds such as cups, leaves, palace decor, and green
background panels.

## Decision

`c067` yields an auditable attribute review queue, but it is not yet a reliable
encoder-side supervised dataset for the direct green / non-human face failure. The next
training loop should not treat the direct-green top-k as clean positives. It should either:

- add manual labels for direct-green/non-human positives and background-green negatives, or
- run a stronger captioning/attribute teacher that can explicitly distinguish character skin
  color/species from background and object color before encoder-side training.
