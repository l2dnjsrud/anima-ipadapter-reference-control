# c049 QwenVL Rank21-40 Reviewed Identity Candidates

Date: 2026-06-12

## Goal

Review ranks 21-40 from the c046 QwenVL-ranked candidate sheet. This tests how
quickly candidate quality drops after the top20 and adds harder negatives/noisy
same-character examples to the reviewed seed.

## Inputs

- Ranked source:
  `eval/qwenvl_ranked_identity_candidates_20260612_c046/qwenvl_top40_face_candidates.jsonl`
- Review input:
  `rank21_40_review_candidates.jsonl`
- Manual labels:
  `manual_visual_labels.jsonl`

## Results

Artifacts:

- `reviewed_candidate_pairs.jsonl`
- `usable_positive_pairs.jsonl`
- `different_character_pairs.jsonl`
- `unclear_or_noisy_same_pairs.jsonl`
- `reviewed_candidate_sheet.jpg`

Summary:

| label | count |
|---|---:|
| reviewed rows | 20 |
| same_character | 9 |
| different_character | 9 |
| unclear | 2 |
| positive_usable | 3 |
| same_character rows excluded from positive | 6 |
| noisy/unclear rows excluded from positive | 8 |

## Decision

`qwenvl_rank21_40_precision_drops_adds_hard_negatives`

Ranks 21-40 are much noisier than the c047 top20. They are useful mostly for
hard negatives and noisy same-character examples, not for rapidly expanding
clean positives.

## Next

Merge c049 with c048 and rerun the QwenVL pooled feature gate to test whether
the identity metric remains stable with noisier reviewed rows.
