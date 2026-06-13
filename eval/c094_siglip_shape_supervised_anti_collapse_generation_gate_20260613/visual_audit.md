# c094 visual audit

## Verdict

`c094_shape_supervision_exhausted_requires_encoder_training`

C094 is operational in ComfyUI and is not blank, but it is not promoted as a high-quality reference-control checkpoint. The adapter still collapses most direct-green non-human references into a narrow green human-face prior.

## Evidence

- Contact sheet: `contact_sheet_hard_shape.jpg`
- Generated PNG count: `121`
- Samples: `11`
- Runtime variants: `11`
- Best C094 variant: `c094_shape_supervised_w14`
- Best C094 mean uplift: `0.0878832954`
- C093 w14 mean uplift: `0.0863735780`
- C094 over C093: `+0.0015097174`
- Qwen target baseline mean uplift: `0.1089544056`
- Heldout07 best C094 uplift: `0.0085623513`
- C094 blank-like rows: `[]`
- Pixel audit low-variance image: one `no_ip` baseline image, not a C094 image

## Visual Notes

The reference column contains frog/chibi/mascot and other non-human silhouettes. C094 at weights `0.8`, `1.0`, `1.2`, and `1.4` consistently produces usable images, but most rows become a similar green male head or bust. The generated face angle, hairline, jaw, and clothing vary a little by row, but the species/body-shape identity from the reference is mostly lost.

The chibi and mascot rows are the clearest failure cases. Small round eyes, tiny body proportions, frog head shapes, and simplified cartoon silhouettes do not carry into the generated outputs. C094 instead maps them into Anima's green martial-artist/headshot prior.

Heldout07 improves slightly over C092/C093 by metric, but visually it is still not a reliable match. The monster side-profile, exaggerated jaw, red eye, and rough silhouette are not preserved. The result is closer to a dark-haired human warrior portrait with some red-eye influence.

## Decision

C094 confirms that target-side latent shape supervision can produce a small incremental gain, but it does not break the collapse mode. Continuing with adapter-only C093/C094-style loss tuning is unlikely to reach the requested "trustworthy high-quality reference-control" bar. The next loop should train or adapt the image encoder / feature bridge itself, so the conditioning features preserve non-human shape and species identity before they reach the IP cross-attention layers.
