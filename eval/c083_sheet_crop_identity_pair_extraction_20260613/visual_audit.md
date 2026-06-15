# c083 sheet crop identity pair extraction visual audit

## Goal

c083 tested the next acquisition route after c082 showed that stricter prompt-only negative constraints still produced character-sheet layouts. The goal was to recover usable single-figure same-identity training targets by cropping individual foreground figures from the c082 sheet-like generations.

## Extraction

- Source manifest: `eval/c082_single_image_pair_acquisition_20260613/generation_manifest.jsonl`
- Source generated images: `24`
- Crop candidates: `104`
- Raw crop policy: raw crop PNGs stay under `.tmp/c083_sheet_crop_identity_pair_extraction/crops/`
- Review sheet: `contact_sheet.jpg`
- Heldout rows used: `0`
- Training started: `false`

## Visual Result

The automatic crop extraction recovered many usable single-figure crops from the generated sheets. This worked especially well for these identity groups:

- `c082_green_oni_scout`: action/profile/three-quarter crops preserve a green oni scout identity; stacked front-sheet crops were excluded.
- `c082_jade_lizard_monk`: front and three-quarter crops are single lizard/monk figures with consistent robe/sash cues.
- `c082_goblin_mage`: most crops are clean single goblin mage figures with purple hood/goggle cues.
- `c082_frog_yokai_guard`: most crops are single frog/yokai guard figures; the front stacked crop was excluded.

`c082_plant_dryad` had only front-view single crops and no usable cross-source diversity. `c082_serpent_dancer` had several single crops, but identity and style varied too much across source views, so it was kept as useful proxy only rather than paired same-identity supervision.

## Review Summary

- `reviewed_rows`: `104`
- `target_positive_rows`: `74`
- `approved_group_count`: `4`
- `approved_pair_rows`: `970`
- `direct_self_pair_rows`: `0`
- `decision`: `ready_for_c084_paired_training_manifest`

## Decision

Decision: `ready_for_c084_paired_training_manifest`

c083 proves that crop extraction is a better acquisition surface than repeating prompt-only single-image constraints. The next loop should build a balanced c084 training manifest from the approved c083 pairs, with group/source downsampling so the 970 directed pairs do not overfit duplicated crops from one synthetic sheet family.
