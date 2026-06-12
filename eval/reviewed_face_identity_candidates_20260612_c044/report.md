# c044 Reviewed Face/Upper-Body Identity Candidates

Date: 2026-06-12

## Goal

Turn the c043 face/upper-body candidate pool into a conservative reviewed
identity manifest before using it for any feature gate or training decision.

## Inputs

- Candidate pool:
  `eval/broad_identity_candidate_mining_20260612_c043/kept_face_candidates.jsonl`
- Candidate count: `30`
- Review sheets:
  - `candidate_review_page_01.jpg`
  - `candidate_review_page_02.jpg`
  - `candidate_review_page_03.jpg`

## Label Policy

Labels are conservative:

- `same_character`: the same character is visible in both images.
- `different_character`: the two sides are visually different primary
  characters.
- `unclear`: same-character is possible but the crop is too partial, back-facing,
  or multi-character-heavy.
- `positive_usable=true`: both sides are clear enough for feature/training
  positive use. Same-character but noisy/multi-character/back-facing pairs are
  not marked usable.

## Results

Artifacts:

- `manual_visual_labels.jsonl`
- `reviewed_candidate_pairs.jsonl`
- `usable_positive_pairs.jsonl`
- `different_character_pairs.jsonl`
- `unclear_or_noisy_same_pairs.jsonl`
- `reviewed_candidate_sheet.jpg`

Summary:

| label | count |
|---|---:|
| reviewed rows | 30 |
| same_character | 12 |
| different_character | 15 |
| unclear | 3 |
| positive_usable | 8 |
| noisy/unclear same rows excluded from positive | 7 |

## Decision

`reviewed_face_seed_expanded_but_still_small`

c044 doubles the usable positive seed from c041's 4 pairs to 8 pairs and adds
15 reviewed hard negatives. This is enough to rerun feature separation, but it
is still too small and too concentrated for adapter training.

## Next

Run the c042-style feature probe again on:

- positive: `positive_usable=true`
- negative: `review_label=different_character`

If a raw feature passes the gate, use it for candidate ranking and larger
identity-set mining. Do not treat the result as generation quality yet.
