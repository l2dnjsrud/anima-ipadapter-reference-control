# C098 SigLIP Encoder LoRA Visual Audit

Date: 2026-06-13

## Verdict

Decision: `c098_encoder_lora_not_promoted_requires_stronger_encoder_or_better_data`

C098 generated successfully through the native ComfyUI SigLIP path and all C098 candidate rows are nonblank. It does not qualify as a high-quality reference-control candidate. The best C098 variant is `c098_lora_c094_w14`, with mean uplift `0.0865313863`, which is slightly below C096 `0.0880849553`, C094 `0.0878832954`, and clearly below the Qwen hard-shape baseline `0.1089544056`.

## Visual Findings

- C098 proves the deeper encoder-LoRA checkpoint can be loaded and applied in the native SigLIP ComfyUI workflow.
- The C098 weight sweep mostly changes strength and contrast, not reference identity fidelity.
- Frog, chibi, mascot, and yokai references still collapse toward a green adult humanoid face or bust instead of preserving small body proportions, round head/eye shapes, simplified silhouettes, or non-human profile geometry.
- `c098_lora_c094_w14` is visually close to C094/C095/C096 and does not add a decisive new control signal.
- Heldout07 remains the clearest failure: the reference is a non-human green side-profile with distinctive jaw/eye silhouette, but C098 still produces a dark human warrior portrait family. Its best heldout uplift is `0.0071462548`, below C095 `0.0115417034`.
- The only blank-like row is the known `crop_pair00_no_ip` baseline row, which also appeared in the earlier C096 audit. It is not a C098 candidate collapse.

## Operational Notes

- Generated images: `88`
- Samples: `11`
- Variants: `8`
- Contact sheet: `contact_sheet_hard_shape.jpg`
- Contact sheet size: `2544x3504`
- C098 blank-like rows: `0`
- Low-variance / blank-like total count includes one baseline no-IP row, not a C098 output.

## Next Decision

Do not promote C098. The next loop should not keep scaling the same small SigLIP encoder-LoRA recipe alone. The evidence points toward either stronger hard-shape/color reference data or a stronger encoder/objective that directly supervises non-human silhouette, body proportion, and distinctive reference attributes.
