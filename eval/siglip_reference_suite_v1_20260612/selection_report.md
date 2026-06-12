# SigLIP Reference Suite v1 Selection Report

Date: 2026-06-12

## Source

```text
/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best
```

The suite is read-only with respect to the source dataset. It is built from the
previously curated single-character color-panel manifests:

- `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- `training/manifests/local_color_single_character_clean32_20260611.jsonl`

## Selection Method

The first automatic v1 attempt scanned the local color-panel dataset and kept 32
rows by aspect-ratio and caption heuristics. Visual inspection showed that this
let through prop-only, table/interior, and distant room panels. That set was not
strong enough for a single-character reference-control gate.

The final v1 suite therefore uses:

- all 8 rows from the curated heldout8 manifest used by c034;
- the first 24 rows from the curated clean32 manifest;
- the remaining 8 clean32 rows as `heldout_unused.jsonl`.

This keeps the c035 gate focused on visible character reference behavior rather
than page layout or background reconstruction.

## Counts

| item | count |
|---|---:|
| original source images scanned by prior selector | 1,571 |
| original candidates kept by prior selector | 418 |
| c035 suite rows | 32 |
| unused clean32 rows | 8 |
| rejected/skipped by original selector | 1,153 |

## Visual Coverage Buckets

Counts are non-exclusive and come from manual inspection of `reference_sheet.jpg`.
They are coverage buckets, not mutually exclusive class labels.

| bucket | count |
|---|---:|
| young / black-haired wuxia male | 15 |
| elder, bearded, bald, or monk-like face | 8 |
| official hat, formal robe, fan, or court costume | 6 |
| female or noble costume reference | 3 |
| angry, shouting, stern, or strong expression | 14 |
| close-up / upper-body dominant composition | 25 |
| seated or full-body indoor composition | 3 |
| weapon, action, fan, or distinctive prop | 4 |
| non-human or unusual colored face | 2 |
| dark blue/black palette | 9 |
| warm red/gold palace palette | 6 |

## Artifacts

- `reference_suite_v1.jsonl`
- `heldout_unused.jsonl`
- `summary.json`
- `reference_sheet.jpg`

## Intended Use

This suite is the c035 input set. It should be used with:

- `no_ip`
- `siglip_kv_init_w14`
- `siglip_ref_retrieval_w14`

The suite is a single-character color-reference gate. It is not a line-art
colorization gate and not a full page-layout generalization gate.
