# QwenVL c060 Failure-Focused Visual Audit

Decision: `c060_failure_focused_not_quality_pass_runtime_blend_remains_best`

Compared columns:

- `reference`
- `no_ip`
- `prev_w14`
- `blend_prev14_c05504`
- `c060_w14`

Artifacts:

- Train contact sheet: `eval/qwenvl_c060_failure_focused_gate_20260612/contact_sheet_train.jpg`
- Heldout contact sheet: `eval/qwenvl_c060_failure_focused_gate_20260612/contact_sheet_heldout.jpg`
- PE metric: `eval/qwenvl_c060_failure_focused_gate_20260612/pe_similarity_metrics.json`
- QwenVL metric: `eval/qwenvl_c060_failure_focused_gate_20260612/qwenvl_similarity_metrics.json`

Metric outcome:

| metric | best | second | c060 |
| --- | ---: | ---: | ---: |
| PE mean uplift | `blend_prev14_c05504=0.049596` | `prev_w14=0.029240` | `c060_w14=0.021860` |
| QwenVL mean uplift | `blend_prev14_c05504=0.041589` | `prev_w14=0.036187` | `c060_w14=0.031796` |
| PE improved rate | `prev_w14=0.750` | `blend_prev14_c05504=0.725` | `c060_w14=0.600` |
| QwenVL improved rate | `blend_prev14_c05504=0.800` | `prev_w14=0.725` | `c060_w14=0.725` |

Visual observations:

- c060 is active: it changes palette, outfit, pose emphasis, and villain aura relative to `no_ip`.
- c060 does not beat the runtime blend. The blend column is usually more stable for the black/red costume family and keeps fewer obvious identity drifts.
- On heldout samples, c060 sometimes improves local costume or beard/hat cues, but it still fails the strict identity/control target.
- `heldout01` loses the old, square-faced man identity and turns into a younger shouting character.
- `heldout07` still collapses the green monster side-profile reference into a human dark-villain template.
- `heldout02`, `heldout05`, and `heldout06` show that c060 can retrieve beard/hat/armor-like cues, but these gains are not broad enough to pass the high-quality gate.

Conclusion:

c060 proves that failure-focused continuation remains technically viable and reference-active, but this adapter-only continuation does not solve the core reference-control problem. The current best usable recipe remains `blend_prev14_c05504`. The next experiment should avoid another narrow adapter-only continuation unless it changes the encoder/feature calibration or distills the runtime blend with a stronger objective.
