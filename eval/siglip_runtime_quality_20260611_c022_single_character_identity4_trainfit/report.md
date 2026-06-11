# SigLIP c022 Single-Character Identity4 Train-Fit

- Init checkpoint:
  `anima_siglip_ip_adapter_identity128_pe_query_patch_0064_20260611.safetensors`
- Output checkpoint:
  `anima_siglip_ip_adapter_single_character_identity4_pe_query_patch_0256_20260611.safetensors`
  (local ignored artifact)
- Manifest:
  `training/manifests/local_color_single_character_identity4_20260611.jsonl`
- Runtime: isolated ComfyUI API on `127.0.0.1:8116`, GPU0, repo custom node
  symlink, PE-style query patch active.
- Columns: reference / no-IP / pre `w=1.0` / post `w=0.7` /
  post `w=1.0` / post `w=1.4`.
- Contact sheet:
  `eval/siglip_runtime_quality_20260611_c022_single_character_identity4_trainfit/contact_sheet.jpg`
- Decision:
  `single_character_identity4_trainfit_improves_palette_but_not_identity_pass`

Training summary:

```json
{
  "steps": 256,
  "rows_loaded": 4,
  "first_loss": 0.22382895648479462,
  "final_loss": 0.13005425035953522,
  "mean_loss": 0.2278406125260517,
  "mean_base_loss": 0.1967822091828566,
  "mean_contrastive_loss": 0.03507533890660852,
  "mean_teacher_loss": 0.02704146476389724,
  "finite_loss": true
}
```

Visual read: this is much easier to judge than page-level contact sheets. The
adapter clearly learns a reference signal from the four single-character rows.
The no-IP baseline remains the same generic black-robed female portrait across
rows, while the post-training adapter varies palette, face angle, robe color,
and background much more strongly.

- `blue_robed_elder` is the clearest train-fit success. Post `w=1.0` and
  `w=1.4` recover the blue robe and cool background palette better than the
  pre-training checkpoint.
- `black_robe_closeup` improves the black/red palette and side-facing portrait
  direction, especially at higher post weights.
- `bearded_tan_robe` partially improves tan/green robe tendency and heavier
  face mass at `w=1.4`, but beard and age are still not reliable.
- `golden_angry_face` remains a failure for the key attributes. The checkpoint
  does not preserve the gold/fire palette, hair shape, or extreme expression.

Conclusion: the single-character setup confirms that the corrected SigLIP
adapter can learn from a focused dataset, so the route is not a pure no-op.
It is still not a production-quality reference-control model. The training
signal is strongest for coarse palette and face direction, weaker for identity,
age, beard, and unusual color accents. A four-image micro-train is useful as a
diagnostic but too narrow to trust as the final recipe.
