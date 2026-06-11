# SigLIP c023 Single-Character Identity4 Held-Out

- Pre checkpoint:
  `anima_siglip_ip_adapter_identity128_pe_query_patch_0064_20260611.safetensors`
- Post checkpoint:
  `anima_siglip_ip_adapter_single_character_identity4_pe_query_patch_0256_20260611.safetensors`
  (local ignored artifact)
- Train source:
  `training/manifests/local_color_single_character_identity4_20260611.jsonl`
- Runtime: isolated ComfyUI API on `127.0.0.1:8116`, GPU0, repo custom node
  symlink, PE-style query patch active.
- Columns: reference / no-IP / pre `w=1.0` / post `w=1.0`.
- Contact sheet:
  `eval/siglip_runtime_quality_20260611_c023_single_character_identity4_heldout/contact_sheet.jpg`
- Decision:
  `single_character_identity4_heldout_partial_transfer_not_quality_pass`

Held-out references:

- `heldout_old_face`
- `heldout_black_profile`
- `heldout_bearded_table`
- `heldout_black_fullbody`

Visual read: the micro-trained checkpoint transfers some coarse reference
attributes beyond the exact four training images, but it also overfits the tiny
identity4 set.

- `heldout_black_profile` is the best held-out row. The post checkpoint
  preserves the side/profile direction and black/red robe palette better than
  no-IP and better than the pre checkpoint.
- `heldout_black_fullbody` improves the black/red palette and darker mood, but
  does not preserve the full-body layout.
- `heldout_bearded_table` moves toward tan/green robes and a heavier jaw, but
  it still does not reproduce a true old bearded face.
- `heldout_old_face` is a bad generalization case. The post checkpoint drifts
  toward a blue-robed side portrait and misses the glasses, mustache, age, and
  frontal face structure.

Conclusion: single-character testing is the right next gate because it reveals
the model's real behavior without page-layout noise. The identity4 checkpoint
shows partial transfer for coarse palette and pose, but not reliable identity
control. This is not an IP-Adapter impossibility proof; it is evidence that the
current frozen-SigLIP adapter objective needs more diverse single-character
data and a stronger identity/palette discrimination loss before broad training.
