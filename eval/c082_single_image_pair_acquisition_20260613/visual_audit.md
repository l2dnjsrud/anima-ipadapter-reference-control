# c082 single-image identity pair acquisition visual audit

## Goal

c082 tested whether stricter prompt wording could turn the c081 multi-pose character-sheet outputs into usable same-identity, single-image source/target pairs.

## Generation

- API surface: ComfyUI02 `http://127.0.0.1:8102`
- Prompt package: `prompt_manifest.jsonl`
- Generated images: `24`
- Blank images: `0`
- Raw PNG policy: kept under `.tmp/c082_single_image_pair_acquisition/generated/`
- Review sheet: `contact_sheet.jpg`

## Visual Result

The stricter prompt wording improved the explicit instruction surface, but it did not solve the generator behavior. Most outputs still contain multiple repeated poses, front/back turnarounds, lineup grids, or sprite-sheet-like layouts inside a single generated image.

`c082_jade_lizard_monk_action` is the only clear target-positive candidate: it is a single green lizardfolk figure with a coherent outfit and no visible duplicate figure. It is not enough for pair training because the other views in that identity group are still multi-view sheets, so the group has fewer than two target-positive views.

The other identity groups remain useful as non-human green proxy references but not as paired target images:

- `c082_green_oni_scout`: mostly repeated full-body or head/side-view sheets.
- `c082_jade_lizard_monk`: action is usable; front/three-quarter/profile are repeated turnaround outputs.
- `c082_goblin_mage`: mostly repeated chibi sprite or pose sheets.
- `c082_frog_yokai_guard`: mostly repeated simple character sheets.
- `c082_plant_dryad`: often two-character front/back or multi-view sheets.
- `c082_serpent_dancer`: identity and pose vary, with multi-view/action-sheet layouts.

## Decision

Decision: `more_identity_pairs_required`

Final review:

- `generated_count`: `24`
- `blank_count`: `0`
- `reviewed_rows`: `24`
- `target_positive_rows`: `1`
- `approved_group_count`: `0`
- `approved_pair_rows`: `0`
- `direct_self_pair_rows`: `0`

c082 proves that prompt-only negative constraints are not sufficient for this Qwen image generation surface. The next acquisition loop should stop repeating prompt-only wording and instead change the generation surface: for example, image-to-image/crop-based extraction from generated sheets, automatic single-figure crop detection, or a different generator/conditioning route that naturally emits one figure per image.
