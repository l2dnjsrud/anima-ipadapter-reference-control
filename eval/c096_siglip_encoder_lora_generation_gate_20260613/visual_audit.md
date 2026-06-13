# C096 SigLIP Encoder LoRA Visual Audit

Date: 2026-06-13

## Verdict

Decision: `c096_encoder_lora_not_promoted_requires_data_expansion_or_deeper_encoder_training`

C096 generated successfully through the native ComfyUI SigLIP path and did not create blank-like C096 rows, but it does not qualify as a high-quality reference-control candidate. The best C096 variant is `c096_lora_c094_w14`, with mean uplift `0.0880849553`, essentially tied with C094 `0.0878832954` and slightly above C095 `0.0865223347`, but still clearly below the Qwen hard-shape baseline `0.1089544056`.

## Visual Findings

- C096 changes the image and remains stable, so the encoder LoRA path is functional.
- Frog, chibi, mascot, and yokai references still collapse toward a green adult humanoid face or bust.
- C096 does not preserve small body proportions, frog-like head shape, simplified mascot silhouette, or non-human profile strongly enough.
- Heldout07 remains a key failure: the reference is a monster side-profile with red-eye/jaw silhouette, but C096 produces a human warrior portrait family. Its best heldout uplift is `0.0056569861`, below C094 `0.0085623513` and C095 `0.0115417034`.
- Compared with C094/C095, the C096 w1.4 column is visually similar. It does not provide a decisive new control signal.

## Operational Notes

- Generated images: `77`
- Samples: `11`
- Variants: `7`
- Contact sheet: `contact_sheet_hard_shape.jpg`
- Contact sheet size: `2334x3504`
- C096 blank-like rows: `0`
- Low-variance / blank-like total count includes one baseline no-IP row, not a C096 output.

## Next Decision

Do not promote C096. The next loop should not repeat shallow encoder LoRA at the same data scale. Use either:

- larger reviewed hard-shape/reference-control pairs, or
- deeper SigLIP encoder adaptation / stronger encoder training with explicit hard negatives and non-human silhouette supervision.
