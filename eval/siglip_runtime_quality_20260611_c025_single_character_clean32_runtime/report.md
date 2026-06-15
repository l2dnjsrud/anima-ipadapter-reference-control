# SigLIP c025 Single-Character Clean32 Runtime Evaluation

- Contact sheet: `eval/siglip_runtime_quality_20260611_c025_single_character_clean32_runtime/contact_sheet.jpg`
- Summary: `eval/siglip_runtime_quality_20260611_c025_single_character_clean32_runtime/summary.json`
- Runtime: isolated ComfyUI API on `127.0.0.1:8116`, GPU0, repo custom node.
- Columns: reference / no_ip / pe_base_w1 / identity4_w1 / clean32_w1 / clean32_w14.

## Inputs

- Train manifest: `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- Held-out manifest: `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- Selection sheet: `eval/siglip_runtime_quality_20260611_c024_single_character_clean32_selection/candidate_sheet.jpg`
- Clean32 checkpoint: `checkpoints/anima_siglip_ip_adapter_single_character_clean32_pe_query_patch_0512_20260611.safetensors`

The clean32 checkpoint was trained from the PE-style query-patch checkpoint with
32 visually curated single-character color panels for 512 steps. The runtime
evaluation used four train samples and four held-out samples with the same
solo-portrait prompt and matched seeds across no-IP and adapter variants.

## Visual Result

Decision: `single_character_clean32_runtime_not_quality_pass`

Single-character testing is the right diagnostic shape: the adapter influence
is much easier to see than it was on multi-panel pages. The clean32 checkpoint
does change outputs beyond no-IP and often pushes stronger dark/red/blue wuxia
palette, older stern faces, and heavier robe styling.

It still does not behave like a reliable high-quality reference-control model.
The generated images miss stable reference-specific identity and props:

- `train14` and `heldout02` do not preserve old-man beard/age details reliably.
- `train23` misses glasses, fan, and the flatter scholarly face structure.
- `heldout05` misses the cropped screaming bearded face and instead becomes a
  generic stern man at stronger weight.
- `heldout07` misses the green demon/red-eye side-profile content and collapses
  into a black-robed human male.
- Higher `clean32_w14` usually strengthens the learned template rather than
  improving reference fidelity.

## Interpretation

This proves the current native SigLIP path is not dead: the ComfyUI patch works,
the checkpoint loads, and the image reference affects generation. But the
training objective is still learning coarse dataset/style clusters more than
fine reference identity, palette, prop, and facial-structure constraints.

The next improvement should not be a long run of the same frozen-SigLIP
adapter-only recipe. The next useful path is to keep single-character
evaluation as the gate, but change the learning signal: stronger
reference-discrimination losses, trainable image-feature calibration, or an
anime/Qwen-VL style encoder stage that is explicitly optimized to retain
identity and palette before returning to multi-panel page layouts.
