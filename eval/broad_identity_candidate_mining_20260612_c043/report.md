# c043 Broad Face/Upper-Body Identity Candidate Mining

Date: 2026-06-12

## Goal

Expand beyond the tiny c041 reviewed seed before launching any stronger encoder
or adapter training. The target was not to auto-label identity. The target was
to mine a larger, reviewable set of same-page non-duplicate pairs where both
sides are likely to contain a clear face or upper-body character crop.

## Inputs

- Data root:
  `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`
- Raw candidate manifest:
  `eval/broad_identity_candidate_mining_20260612_c043/raw_candidate_pairs.jsonl`
- Raw candidate count: `160`

The raw manifest was generated with the same-page non-duplicate candidate miner
at a wider limit than c039.

## Method

Added `tools/filter_face_upper_body_candidates.py`.

The filter scores each side with Qwen3-VL image-text retrieval:

- Positive prompts: clear face close-up, upper-body portrait, single martial
  arts bust shot, solo manhwa face-and-shoulders.
- Negative prompts: torso without face, wide group scene, background/building,
  prop without face, tiny distant character.

For each image:

```text
face_upper_score = max(positive_text_scores) - max(negative_text_scores)
```

A pair is kept only when both sides have `face_upper_score >= threshold`.

## Calibration

At `threshold=0.0`, the filter kept `152 / 160` pairs, which was too loose.
The minimum side-score distribution showed:

| threshold | kept pairs |
|---:|---:|
| -0.05 | 160 |
| 0.00 | 152 |
| 0.02 | 132 |
| 0.05 | 91 |
| 0.08 | 30 |
| 0.10 | 4 |

`threshold=0.08` was selected because it keeps a reviewable 30-pair set without
collapsing to a tiny seed.

## Results

Final artifacts:

- `scored_face_candidates.jsonl`
- `kept_face_candidates.jsonl`
- `kept_face_candidate_sheet.jpg`

Final count:

| metric | value |
|---|---:|
| input pairs | 160 |
| kept pairs | 30 |
| threshold | 0.08 |
| distinct SG pages in kept set | 22 |
| kept min side-score range | 0.080176 to 0.114322 |

Visual inspection of the contact sheet confirms that the filter mostly keeps
face or upper-body character crops. It does not solve identity labeling:
different characters and partial/crop ambiguities remain.

## Decision

`face_upper_body_filter_expands_review_pool_not_identity_labels`

c043 is useful as a candidate expansion step. It is not a training gate and not
a proof that SigLIP/QwenVL can already separate same-character identity. The
next loop should manually review the 30 kept pairs into `same_character`,
`different_character`, and `unclear`, then rerun the c042 feature separation
probe on the larger reviewed set.

## Verification

```bash
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python -m pytest \
  tests/test_face_upper_body_candidate_filter.py -q
```

Result: `4 passed`.
