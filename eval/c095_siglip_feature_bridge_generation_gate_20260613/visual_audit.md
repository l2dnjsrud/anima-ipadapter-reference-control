# c095 Visual Audit

## Verdict

- Decision: `not_promoted`
- Contact sheet: `eval/c095_siglip_feature_bridge_generation_gate_20260613/contact_sheet_hard_shape.jpg`
- Sheet size: `3384x3504`
- Generated images: `132`
- Blank-like rows: `1` (`crop_pair00_no_ip` only; no C095 blank-like row)

## What Improved

- The C095 bridge checkpoint loads through the native SigLIP ComfyUI node and generates nonblank outputs.
- C095 `w14` slightly improves heldout07 over C094 numerically (`0.0115417034` vs `0.0085623513`), and the heldout07 row keeps a stronger side-profile monster cue than C092/C093 in a few cells.
- Some C095 rows look more stable than the lower-weight C095 variants; `w08` is visibly too weak and more variable.

## What Failed

- C095 does not beat C094 on mean hard-shape uplift: `0.0865223347` vs C094 `0.0878832954`.
- It remains well below the Qwen baseline mean uplift `0.1089544056`.
- The key hard-shape failure remains: frog/chibi/non-human references still collapse into green human or villain-like heads.
- Heldout07 remains visually incomplete: C095 preserves some green side-profile/monster coloring, but it does not reconstruct the non-human face geometry strongly enough for reference-control use.
- Speech-bubble/crop context and prop/body shape are not materially recovered by the bridge.

## Next Decision

C095 proves that a small residual bridge after the SigLIP fused tokens is loadable and trainable, but it is not enough to solve the hard-shape/reference identity gap. The next loop should move to deeper SigLIP image-encoder checkpoint adaptation or larger anime/reference-control data expansion, not another adapter-only or tiny-bridge continuation.
