# SigLIP c028 Single-Character PE-Space Init Runtime Evaluation

- Contact sheet: `eval/siglip_runtime_quality_20260611_c028_single_character_pe_space_init_runtime/contact_sheet.jpg`
- Summary: `eval/siglip_runtime_quality_20260611_c028_single_character_pe_space_init_runtime/summary.json`
- Runtime: isolated ComfyUI API on `127.0.0.1:8116`, GPU0, repo custom node.
- Columns: reference / no_ip / clean32_w1 / clean32_w14 / pe_space_w1 / pe_space_w14.

## Inputs

- Train manifest: `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- Held-out manifest: `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- Seed checkpoint: `checkpoints/anima_siglip_ip_adapter_single_character_clean32_pe_query_patch_0512_20260611.safetensors`
- PE-space checkpoint: `checkpoints/anima_siglip_ip_adapter_single_character_clean32_pe_space_init_0512_20260611.safetensors`

This run changed the native SigLIP architecture to support asymmetric
dimensions: `dit_dim=2048` for Anima query hidden states and `ip_hidden_dim=1024`
for PE-token-space image tokens. The training initializer copies compatible
base SigLIP weights, then copies the PE adapter's trained `to_k_ip`,
`to_v_ip`, and gate values into the native SigLIP adapter.

Observed training summary:

- rows loaded: `32`
- steps: `512`
- first/final loss: `0.5026331543922424` / `0.18201853334903717`
- mean loss: `0.22448686529241968`
- mean base loss: `0.19500588972005062`
- mean contrastive loss: `0.0499068612116389`
- mean teacher loss: `0.016648300783572267`
- mean token loss: `0.78202638507355`
- mean PE-token loss: `0.01605273197731094`
- finite loss: `true`
- trainable parameters: `218159260`

## Visual Result

Decision: `single_character_pe_space_init_not_quality_pass`

The PE-space checkpoint loads and runs through the ComfyUI native SigLIP node,
which proves the asymmetric checkpoint path works. Visually, the result is
cleaner and sharper than several earlier SigLIP attempts, but it collapses
toward a narrow handsome black-haired wuxia male template.

Important failures:

- `train14` and `heldout02` lose old/bald/bearded elder features.
- `train23` loses glasses, fan, hat, and scholarly face.
- `heldout05` loses the screaming cropped bearded face.
- `heldout07` loses the green demon/red-eye non-human identity.
- `pe_space_w14` strengthens the same young-male template instead of improving
  reference fidelity.

## Interpretation

This is not a final quality pass. It is still useful progress: the native
SigLIP loader/runtime can now handle PE-token-space checkpoints, and PE K/V
initialization is mechanically valid. The failure mode changed from noisy or
weak reference influence to a clean but over-regularized identity template.

The next improvement should not be another adapter-only run with frozen SigLIP.
The evidence now points at the encoder/resampler side: SigLIP features are not
carrying enough anime identity detail into PE-token-space. The next branch
should train a stronger image encoder/calibrator or reference retrieval/ID
objective before adapter denoising, or use Qwen/PE teacher features to supervise
identity/palette/prop tokens directly instead of only pooled K/V descriptors.
