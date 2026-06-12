# QwenVL c061 Instruction Calibration Visual Audit

Decision: `instruction_calibration_species_face_best_preset_not_quality_pass`

Compared columns:

- `reference`
- `no_ip`
- `blend_default`
- `blend_identity_exact`
- `blend_species_face`

Artifacts:

- Train contact sheet: `eval/qwenvl_c061_instruction_calibration_gate_20260612/contact_sheet_train.jpg`
- Heldout contact sheet: `eval/qwenvl_c061_instruction_calibration_gate_20260612/contact_sheet_heldout.jpg`
- PE metric: `eval/qwenvl_c061_instruction_calibration_gate_20260612/pe_similarity_metrics.json`
- QwenVL metric: `eval/qwenvl_c061_instruction_calibration_gate_20260612/qwenvl_similarity_metrics.json`

Metric outcome:

| metric | best | second | default |
| --- | ---: | ---: | ---: |
| PE mean uplift | `blend_species_face=0.060893` | `blend_identity_exact=0.054909` | `blend_default=0.049596` |
| QwenVL mean uplift | `blend_species_face=0.042190` | `blend_default=0.041589` | `blend_identity_exact=0.039557` |
| PE heldout uplift | `blend_species_face=0.053534` | `blend_default=0.039142` | `blend_identity_exact=0.035494` |
| QwenVL heldout uplift | `blend_species_face=0.026471` | `blend_default=0.022779` | `blend_identity_exact=0.016944` |

Runtime guard:

- Generated PNGs: `160`
- Blank PNGs: `0`
- Minimum pixel std: `35.883`
- API prompt guard: all non-`no_ip` variants use the same two checkpoints, weights, start/end range, seed, positive prompt, negative prompt, and sample set. The calibrated variants differ only in `AnimaQwenVLEncodeImage.instruction`.
- Cleanup: isolated ComfyUI server stopped, and port `8116` refused `/object_info` after shutdown.

Visual observations:

- `blend_species_face` is the best c061 preset candidate. It nudges PE/QwenVL similarity upward and sometimes keeps beard, hat, red-eye, costume, and dark palette cues slightly better than the default instruction.
- The improvement is small and mostly local. The three adapter columns often converge to nearly the same image, so instruction-only feature calibration is not enough to reshape the reference-control behavior.
- `heldout00` and `heldout06` show useful costume/silhouette/palette transfer, but the face structure remains generic.
- `heldout01` still loses the old square-faced man identity and produces a younger shouting martial artist.
- `heldout07` remains the critical fail: the green non-human side-profile reference collapses into a human dark-villain body template. The `species_face` instruction adds red eyes and dark palette, but not the monster head/profile identity.

Conclusion:

c061 proves that QwenVL image-embedding instructions are not completely inert: `blend_species_face` beats the default runtime blend in aggregate PE and slightly in QwenVL metrics on the same seeds/checkpoints. This can be used as the better runtime preset for the current workflow.

However, prompt-only calibration does not pass the high-quality reference-control gate. The next loop should use real encoder/feature adaptation or a distillation objective that directly teaches non-human silhouette, exact face structure, speech-bubble/crop context, and prop retention.
