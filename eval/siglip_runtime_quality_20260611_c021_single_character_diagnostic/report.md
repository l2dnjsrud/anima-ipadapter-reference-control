# SigLIP c021 Single-Character Diagnostic

- Checkpoint:
  `anima_siglip_ip_adapter_identity128_pe_query_patch_0064_20260611.safetensors`
  (local ignored artifact, loaded through repo `checkpoints/` via ComfyUI
  extra model paths)
- Runtime: isolated ComfyUI API on `127.0.0.1:8116`, GPU0, repo custom node
  symlink, PE-style query patch active.
- Prompt shape: fixed solo wuxia upper-body portrait prompt. The prompt does
  not name reference-specific colors, age, beard, hair color, or exact pose.
- References: four manually selected color-panel crops with single, visually
  dominant characters:
  - `bearded_tan_robe`
  - `blue_robed_elder`
  - `black_robe_closeup`
  - `golden_angry_face`
- Columns: reference / no-IP / `w=0.7` / `w=1.0` / `w=1.4`.
- Contact sheet:
  `eval/siglip_runtime_quality_20260611_c021_single_character_diagnostic/contact_sheet.jpg`
- Candidate selection sheet:
  `eval/siglip_runtime_quality_20260611_c021_single_character_diagnostic/candidate_sheet.jpg`
- Decision: `single_character_diagnostic_not_quality_pass`

Visual read: simplifying the reference from multi-panel pages to
single-character crops does not make the current SigLIP checkpoint reliable.
The adapter clearly changes the no-IP output and creates some
reference-dependent variation, especially around face direction, robe darkness,
and background density. It still misses the important reference attributes:

- the tan-robed bearded reference does not preserve beard, age, or tan robe;
- the blue-robed elder does not preserve the blue palette or elder identity;
- the black-robed close-up is the best row, but still drifts into generic young
  male portrait variants rather than controlled identity transfer;
- the golden angry face does not preserve the gold hair/fire palette or facial
  intensity.

Conclusion: the previous page/contact-sheet failures were not only caused by
complex multi-panel layout. Even after reducing the diagnostic to
single-character color references, the current frozen-SigLIP adapter checkpoint
does not provide production-quality reference control. The PE-style query patch
should remain because it makes the runtime respond to references, but the next
training step needs a stronger reference-discrimination objective, trainable
feature/encoder adaptation, or an anime-domain image encoder path. Simply
running more steps on the same weak target is unlikely to reach the requested
quality gate without changing the learning signal.
