# c035 Visual Audit

- Run: `siglip_runtime_quality_20260612_c035_suite_v1`
- Contact sheet: `eval/siglip_runtime_quality_20260612_c035_suite_v1/contact_sheet.jpg`
- Columns: reference / no_ip / `siglip_kv_init_w14` / `siglip_ref_retrieval_w14`
- Decision: `not_ready`
- Best current SigLIP variant: `siglip_ref_retrieval_w14`

## Summary

`siglip_ref_retrieval_w14` is the best current SigLIP checkpoint in this run, but the result is not yet a reliable high-quality reference-control model.

The positive result is that it often improves broad manhwa styling, costume family, facial expression, and palette over no-IP. The blocking failure is stricter reference fidelity: many rows collapse toward a repeated black long-haired wuxia warrior, purple/night palace lighting, red-eye villain look, or generic official/elder template. This is especially visible on brown-haired faces, glasses/fan props, non-human face traits, soft female faces, and unusual blue/green/pale color identities.

## Gates

| Gate | Result |
| --- | --- |
| blank output | pass: `0` blank images |
| metric uplift | fail: best mean uplift `+0.0577`, required `>= +0.10` |
| metric improved rate | fail: best improved rate `0.65625`, required `>= 0.75` |
| palette/costume/expression/framing | pass: `31 / 32`, required `>= 24 / 32` |
| identity/distinctive trait | fail: `16 / 32`, required `>= 18 / 32` |
| non-human/special trait | fail: `0 / 1` |

## Row-Level Notes

The row-level judgments are stored in `visual_audit.json`. The important pattern is that c035 can create attractive and often relevant manhwa character images, but it is not dependable for preserving identity-level traits. This means the current SigLIP attribute-reference path should remain an experimental recipe, not a "trust and use blindly" reference-control release.
