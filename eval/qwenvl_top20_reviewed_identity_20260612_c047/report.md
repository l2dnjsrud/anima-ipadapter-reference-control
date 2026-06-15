# c047 QwenVL Top20 Reviewed Identity Candidates

Date: 2026-06-12

## Goal

Manually review the top 20 candidates from c046 to confirm whether QwenVL
ranking improves positive mining efficiency.

## Inputs

- Ranked source:
  `eval/qwenvl_ranked_identity_candidates_20260612_c046/qwenvl_top40_face_candidates.jsonl`
- Review input:
  `top20_review_candidates.jsonl`
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
| same_character | 18 |
| different_character | 0 |
| unclear | 2 |
| positive_usable | 14 |
| noisy/unclear same rows excluded from positive | 6 |

## Decision

`qwenvl_top20_review_precision_good`

QwenVL top20 ranking is far more efficient than the unranked c043 pool. It
produces 14 usable positives from 20 reviewed rows. This is still not a complete
training set, but it is useful for expanding the reviewed identity seed.

## Next

Combine c047 positives with c044 hard negatives and rerun the QwenVL pooled
feature gate on the larger reviewed seed.
