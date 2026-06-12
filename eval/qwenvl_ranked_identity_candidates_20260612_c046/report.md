# c046 QwenVL-Ranked Identity Candidate Mining

Date: 2026-06-12

## Goal

Use the c045 result, where QwenVL pooled passed the small reviewed identity
proxy, to rank a broader face/upper-body candidate pool for the next reviewed
identity manifest.

## Inputs

- Data root:
  `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`
- Raw same-page non-duplicate candidates:
  `raw_candidate_pairs.jsonl`
- Raw candidate count: `372`

The raw miner requested limit `640`, but the dataset produced `372` available
same-page non-duplicate candidate pairs.

## Pipeline

1. Build all available same-page non-duplicate candidates.
2. Filter both sides with the c043 QwenVL face/upper-body image-text filter.
3. Rank kept pairs by QwenVL pooled image embedding cosine similarity.
4. Emit the top 40 candidates and review sheets.

## Results

Artifacts:

- `raw_candidate_pairs.jsonl`
- `scored_face_candidates.jsonl`
- `kept_face_candidates.jsonl`
- `kept_face_candidate_sheet.jpg`
- `qwenvl_ranked_face_candidates.jsonl`
- `qwenvl_top40_face_candidates.jsonl`
- `qwenvl_top40_candidate_sheet.jpg`
- `qwenvl_top40_review_page_01.jpg` through `qwenvl_top40_review_page_04.jpg`

Counts:

| stage | count |
|---|---:|
| raw same-page candidates | 372 |
| face/upper-body kept, threshold 0.08 | 65 |
| QwenVL-ranked top set | 40 |
| SG pages in top40 | 27 |

QwenVL similarity:

| metric | value |
|---|---:|
| max similarity | 0.963218 |
| min similarity among 65 ranked pairs | 0.652421 |
| top10 similarity range | 0.9155 to 0.9632 |
| top20 lower bound | 0.8801 |

## Visual Review

The top 10 are mostly clean same-character pairs. The top 20 remain useful but
start to include multi-character or back-facing crops. Rank 21 onward has more
noise: different characters, group panels, and partial crops appear more often.

This confirms that QwenVL pooled ranking improves review efficiency, but it
does not remove the need for manual labels.

## Decision

`qwenvl_ranking_improves_candidate_precision_top20`

The next reviewed loop should label the top 20 first, then optionally continue
down the top40 if more positives are needed. Do not auto-promote top40 to
positive labels.
