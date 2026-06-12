# QwenVL c062 Calibrator Distillation Visual Audit

Decision: `not_promoted`

Compared columns:

- `reference`
- `no_ip`
- `blend_species_face`
- `c062_w14`

Artifacts:

- Train contact sheet: `eval/qwenvl_c062_calibrator_distillation_gate_20260612/contact_sheet_train.jpg`
- Heldout contact sheet: `eval/qwenvl_c062_calibrator_distillation_gate_20260612/contact_sheet_heldout.jpg`
- PE metric: `eval/qwenvl_c062_calibrator_distillation_gate_20260612/pe_similarity_metrics.json`
- QwenVL metric: `eval/qwenvl_c062_calibrator_distillation_gate_20260612/qwenvl_similarity_metrics.json`

Metric outcome:

| metric | blend_species_face | c062_w14 |
| --- | ---: | ---: |
| PE mean uplift | `0.060893` | `0.013234` |
| PE train uplift | `0.062733` | `0.017362` |
| PE heldout uplift | `0.053534` | `-0.003277` |
| QwenVL mean uplift | `0.042190` | `0.026588` |
| QwenVL train uplift | `0.046120` | `0.032966` |
| QwenVL heldout uplift | `0.026471` | `0.001077` |

Runtime guard:

- Generated PNGs: `120`
- Blank PNGs: `0`
- Minimum pixel std: `35.883`
- ComfyUI API guard: `AnimaQwenVLIPAdapterLoader`, `AnimaQwenVLEncodeImage`, and `AnimaQwenVLIPAdapterApply` were visible through `/object_info`; the c062 checkpoint was selectable.
- Cleanup: isolated ComfyUI server stopped, and port `8116` refused `/object_info` after shutdown.

Visual observations:

- `c062_w14` changes palette, pose, hand shape, and robe lighting in several rows, but it does not consistently improve exact reference identity over `blend_species_face`.
- `heldout07` remains the critical failure. The green non-human side-profile reference still collapses into a human dark-villain body template; c062 does not recover the monster head/profile identity.
- `heldout01` and `heldout05` still lose age, face structure, and speech-bubble/crop context.
- On train rows, c062 often increases dramatic purple aura or dark costume cues, but the identity signal remains weaker than the existing runtime blend.

Conclusion:

c062 proves the new checkpoint is loadable and active inside the native ComfyUI QwenVL IP-Adapter path, but it does not beat the current `blend_species_face` runtime preset. The c062 calibrator/distillation continuation should not be promoted as the high-quality reference-control model.

The next loop should move away from this checkpoint as the main branch and target stronger feature adaptation or encoder-side adaptation against the known heldout failures: non-human profile, old-face geometry, beard/headwear retention, props, speech-bubble context, and crop/layout preservation.
