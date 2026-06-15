# SigLIP Runtime Quality Recovery 2026-06-11

## Decision

`partial_pass_training_required`

The native SigLIP runtime patch now affects Anima/Qwen Image sampling, but the
current checkpoints are not reliable blind reference-identity controllers.

## What Passed

- `weight=0` is pixel-identical to no-IP, proving the wrapper skip path is clean.
- `weight>0` produces nonzero image differences, proving the previous no-effect
  runtime bug is fixed.
- Prompt-aligned reference generation can produce good-looking comic panels.
  Best visual set in this run:
  `eval/siglip_runtime_quality_20260611_c004_color64_matched/contact_sheet.jpg`.
- The best prompt-aligned continuation checkpoint tested was:
  `checkpoints/anima_siglip_ip_adapter_color64_continue_20260611.safetensors`.

## What Failed

The stricter identity test intentionally removed direct prompt hints such as
`old`, `bald`, `white beard`, and `prayer beads`.

- `color64` changed the scene strongly but did not carry the old bearded monk identity.
- `self64` did not fix identity retention.
- `self512` with shuffled 1,024-row self-reconstruction also did not recover
  the reference identity. It continued to generate younger martial artists
  rather than the reference monk.

Evidence:

- Color64 matched prompt: `eval/siglip_runtime_quality_20260611_c004_color64_matched/contact_sheet.jpg`
- Color64 identity test: `eval/siglip_runtime_quality_20260611_c005_color64_identity/contact_sheet.jpg`
- Self64 identity test: `eval/siglip_runtime_quality_20260611_c006_self64_identity/contact_sheet.jpg`
- Self512 identity test: `eval/siglip_runtime_quality_20260611_c007_self512_identity/contact_sheet.jpg`

## Training Runs

- `color64_continue`: continued from the 16-step pilot for 64 steps on
  adjacent color-panel pairs. It improved prompt-aligned visual strength.
- `self64_continue`: continued from `color64_continue` for 64 steps on
  `ref_id == tgt_id` self-reconstruction rows. Identity did not improve.
- `self512_continue`: continued from `self64_continue` for 512 steps on
  1,024 shuffled self-reconstruction rows. Identity still did not pass.

## Conclusion

This branch is usable as a prompt-aligned style/composition influence testbed,
not yet as a high-confidence IP-Adapter reference-control model. The next real
quality step is not more short local tuning on the current manifest. It needs a
proper reference-control training set and gate:

- same-character or same-image reconstruction pairs at larger scale,
- captions that do not leak identity tokens during identity evaluation,
- validation prompts that remove identity words and require the image encoder to
  carry identity,
- longer training with cached SigLIP/text features and held-out reference sheets.
