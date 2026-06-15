# SigLIP c027 Single-Character PE-Token Anchor Runtime Evaluation

- Contact sheet: `eval/siglip_runtime_quality_20260611_c027_single_character_pe_token_anchor_runtime/contact_sheet.jpg`
- Summary: `eval/siglip_runtime_quality_20260611_c027_single_character_pe_token_anchor_runtime/summary.json`
- Runtime: isolated ComfyUI API on `127.0.0.1:8116`, GPU0, repo custom node.
- Columns: reference / no_ip / clean32_w1 / clean32_w14 / pe_token_w1 / pe_token_w14.

## Inputs

- Train manifest: `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- Held-out manifest: `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- Seed checkpoint: `checkpoints/anima_siglip_ip_adapter_single_character_clean32_pe_query_patch_0512_20260611.safetensors`
- PE-token-anchor checkpoint: `checkpoints/anima_siglip_ip_adapter_single_character_clean32_pe_token_anchor_0256_20260611.safetensors`

The run added `pe_token_alignment_loss`, which compares pooled per-block K/V
descriptors from the native SigLIP adapter against the frozen PE teacher's
K/V descriptors. This is a more direct semantic anchor than only matching the
denoiser prediction.

Observed training summary:

- rows loaded: `32`
- first/final loss: `0.3050529956817627` / `0.21933485567569733`
- mean loss: `0.26382563475635834`
- mean base loss: `0.1882707678596489`
- mean contrastive loss: `0.043121844122651964`
- mean teacher loss: `0.022011573988493183`
- mean token loss: `0.6074657544959337`
- mean PE-token loss: `0.13854585964872967`
- finite loss: `true`
- trainable parameters: `336650396`

## Visual Result

Decision: `single_character_pe_token_anchor_not_quality_pass`

The PE-token anchor improves visual stability compared with token separation:
outputs are cleaner, less grotesque, and more consistently manhwa-like. It is
not enough for high-quality reference control. The model still misses
reference-specific identity and props:

- `train14` and `heldout02` still miss the aged/bald/bearded elder traits.
- `train23` still misses glasses, fan, and scholar hat.
- `heldout05` still misses the cropped screaming expression.
- `heldout07` still misses the green demon/red-eye non-human face.
- Stronger `1.4` weight mostly strengthens a young/stern wuxia male template
  rather than preserving the reference.

## Interpretation

Pooled K/V alignment is a real improvement over token-separation-only because
it gives the native SigLIP path a PE-family target. But the pooling discards too
much token structure and does not force the SigLIP encoder/resampler to produce
PE-like reference tokens. The next branch should support a PE-token-space
adapter, where SigLIP emits 1024-d image tokens and the native path can reuse
the PE adapter's trained K/V projections directly.
