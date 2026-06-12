# QwenVL c063 Calibrator-Only Visual Audit

Decision: `not_promoted`

Compared columns:

- `reference`
- `no_ip`
- `blend_species_face`
- `c063_calibrator_only_w14`

Artifacts:

- Train contact sheet: `eval/qwenvl_c063_calibrator_only_gate_20260612/contact_sheet_train.jpg`
- Heldout contact sheet: `eval/qwenvl_c063_calibrator_only_gate_20260612/contact_sheet_heldout.jpg`
- PE metric: `eval/qwenvl_c063_calibrator_only_gate_20260612/pe_similarity_metrics.json`
- QwenVL metric: `eval/qwenvl_c063_calibrator_only_gate_20260612/qwenvl_similarity_metrics.json`

Metric outcome:

| metric | blend_species_face | c063_calibrator_only_w14 |
| --- | ---: | ---: |
| PE mean uplift | `0.060893` | `0.029465` |
| PE train uplift | `0.062733` | `0.035551` |
| PE heldout uplift | `0.053534` | `0.005121` |
| QwenVL mean uplift | `0.042190` | `0.037178` |
| QwenVL train uplift | `0.046120` | `0.040380` |
| QwenVL heldout uplift | `0.026471` | `0.024371` |

Heldout focus:

| sample | PE blend | PE c063 | QwenVL blend | QwenVL c063 |
| --- | ---: | ---: | ---: | ---: |
| `heldout01` | `0.043998` | `0.023047` | `0.074187` | `0.096285` |
| `heldout05` | `0.093415` | `-0.016610` | `0.016990` | `0.000413` |
| `heldout07` | `-0.095589` | `-0.109479` | `-0.051999` | `-0.053679` |

Runtime guard:

- Generated PNGs: `120`
- Blank PNGs: `0`
- Minimum pixel std: `35.883`
- ComfyUI API guard: `AnimaQwenVLIPAdapterLoader`, `AnimaQwenVLEncodeImage`, and `AnimaQwenVLIPAdapterApply` were visible through `/object_info`; the c063 checkpoint was selectable.
- Cleanup: isolated ComfyUI server stopped, and port `8116` refused `/object_info` after shutdown.

Visual observations:

- `c063_calibrator_only_w14` is active: it changes pose, robe tone, black/red costume balance, hand shape, and purple lighting in many rows.
- The active changes are not reliably better reference identity. On train rows it often pushes toward the same dark/purple villain template instead of preserving distinctive face/costume details.
- `heldout01` gets a better QwenVL score for c063, but the visible result still misses the old-face geometry, wrinkles, speech-bubble crop context, and original facial structure.
- `heldout05` gains a black official-hat cue, but the beard/crop/context and exact face are still weak; PE and visual comparison prefer the existing blend.
- `heldout07` remains the critical failure: the green non-human side-profile monster reference still collapses into a human dark-villain body template, and c063 is slightly worse than the blend on both PE and QwenVL uplift.

Conclusion:

c063 proves the calibrator-only training path and native ComfyUI loader are functional, but it should not be promoted. The shallow feature-calibrator-only update is not enough to solve the reference-control failures. The next loop should move to deeper encoder-side adaptation or explicit failure-attribute supervision rather than another broad adapter continuation or another calibrator-only run.
