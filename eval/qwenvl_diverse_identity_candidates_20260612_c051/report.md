# c051 QwenVL Diverse Reviewed Identity Candidates

Date: 2026-06-12

## Goal

Expand the reviewed identity seed without simply lowering the c046 rank window.
The selection prioritizes new `SG-*` pages so the seed is less dominated by
previous protagonist-heavy pages.

## Inputs

- Data root:
  `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`
- Face score source:
  `eval/qwenvl_ranked_identity_candidates_20260612_c046/scored_face_candidates.jsonl`
- Reviewed exclusion set:
  `eval/qwenvl_combined_rank40_feature_probe_20260612_c050/combined_reviewed_candidate_pairs.jsonl`

## Pipeline

1. Reused c046 face/upper-body scores and lowered the face threshold from
   `0.08` to `0.07`.
2. Ranked the resulting `109` candidate pairs with QwenVL pooled image
   embeddings.
3. Selected review candidates with `target_count=32`, `max_per_sg_page=1`,
   `min_face_score=0.07`, and `min_similarity=0.78`.
4. Manually labeled the selected candidates as `same_character`,
   `different_character`, or `unclear`, with `positive_usable=true` only for
   clean same-character positives.

## Artifacts

- `face070_candidate_pairs.jsonl`
- `qwenvl_ranked_face070_candidates.jsonl`
- `qwenvl_top109_face070_candidates.jsonl`
- `qwenvl_top109_face070_candidate_sheet.jpg`
- `diverse_selection_summary.json`
- `diverse_review_candidates.jsonl`
- `diverse_review_candidate_sheet.jpg`
- `diverse_review_page_01.jpg` through `diverse_review_page_04.jpg`
- `manual_visual_labels.jsonl`
- `reviewed_candidate_pairs.jsonl`
- `reviewed_candidate_sheet.jpg`
- `usable_positive_pairs.jsonl`
- `different_character_pairs.jsonl`
- `unclear_or_noisy_same_pairs.jsonl`

## Selection Summary

| metric | value |
|---|---:|
| face070 candidate rows | 109 |
| face070 SG pages | 79 |
| eligible rows after exclusions/gates | 45 |
| selected rows | 32 |
| unique selected SG pages | 32 |
| new-page rows | 32 |
| old-page rows | 0 |
| skipped already-reviewed pair ids | 52 |

## Review Summary

| label | count |
|---|---:|
| reviewed rows | 32 |
| same_character | 17 |
| different_character | 12 |
| unclear | 3 |
| positive_usable | 10 |
| noisy/unclear same rows excluded from positive | 10 |

## Decision

`qwenvl_diverse_sampling_improves_seed_diversity`

The diverse sampler produced fewer positives than the c047 top20 but more than
the c049 rank21-40 continuation, while adding `32` new SG pages. This is useful
for broadening the reviewed identity seed before training.

## Next

Combine c051 with c050 and rerun the QwenVL pooled feature gate. If the gate
remains stable, use the larger reviewed seed for the next adapter or metric-head
training decision.
