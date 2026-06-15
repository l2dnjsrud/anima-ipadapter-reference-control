# C108 Visual Audit

- Decision: `c108_generation_gate_not_promoted`
- Candidate: `c108_qwen_teacher_alignment_w14`
- Baseline: `blend_species_face`
- Nonblank: `True`

## 육안 요약

- C108 is visibly active: black/red palette, purple aura, long hair, red eyes, beard/headwear cues often move away from no_ip toward the reference.
- C108 does not clearly beat blend_species_face. Most train rows are visually close to blend, and several heldout rows are weaker on identity lock or shape specificity.
- Heldout02 bald/bearded old man is closer in blend_species_face; C108 keeps the bald head but loses beard/age/detail balance.
- Heldout05 official/bearded crop remains close to blend but is not better; both miss speech-bubble/crop context.
- Heldout07 non-human green side-profile remains unresolved: C108 still collapses into a human dark warrior rather than preserving creature snout/profile.

## 승격 차단 사유

- PE mean uplift remains below blend_species_face by about 0.0218 overall.
- QwenVL mean uplift remains below blend_species_face by about 0.0100 overall.
- Direct wins against blend are only 14/40 in PE and 13/40 in QwenVL.
- Non-human side-profile and speech-bubble/crop context are still not robustly controlled.
