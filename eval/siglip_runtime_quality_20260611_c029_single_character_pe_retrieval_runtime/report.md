# SigLIP c029 Single-Character PE Retrieval Runtime Evaluation

- Contact sheet: `eval/siglip_runtime_quality_20260611_c029_single_character_pe_retrieval_runtime/contact_sheet.jpg`
- Summary: `eval/siglip_runtime_quality_20260611_c029_single_character_pe_retrieval_runtime/summary.json`
- Runtime: isolated ComfyUI API on `127.0.0.1:8116`, GPU0, repo custom node.
- Columns: reference / no_ip / pe_space_w1 / pe_space_w14 / retrieval_w1 / retrieval_w14.

## Inputs

- Train manifest: `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- Held-out manifest: `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- Seed checkpoint: `checkpoints/anima_siglip_ip_adapter_single_character_clean32_pe_space_init_0512_20260611.safetensors`
- Retrieval checkpoint: `checkpoints/anima_siglip_ip_adapter_single_character_clean32_pe_retrieval_0128_20260611.safetensors`

This branch added a PE-token retrieval loss. It makes the native SigLIP
PE-space image tokens prefer the matching frozen PE tokens over a deterministic
wrong-reference PE token set. The goal was to push the resampler toward a
reference-retrieval signal before relying on denoiser loss.

Observed 128-step training summary:

- rows loaded: `32`
- first/final loss: `0.22356687486171722` / `0.4331858456134796`
- mean loss: `0.3135300036519766`
- mean base loss: `0.19486997387139127`
- mean contrastive loss: `0.04941573436371982`
- mean teacher loss: `0.01949366783082951`
- mean token loss: `0.7987985340878367`
- mean PE-token loss: `0.0016404704861088248`
- mean PE-retrieval loss: `0.20024060737341642`
- finite loss: `true`
- trainable parameters: `218159260`

## Visual Result

Decision: `single_character_pe_retrieval_not_quality_pass`

The checkpoint loads and runs through the native SigLIP ComfyUI node. The
retrieval branch remains visually clean, but it does not solve reference
control:

- `train14` and `heldout02` still lose old/bald/bearded elder features.
- `train23` still loses glasses, fan, hat, and scholar identity.
- `heldout05` still loses the cropped screaming bearded expression.
- `heldout07` still loses the green demon/red-eye non-human identity.
- `retrieval_w14` mostly strengthens a handsome young black-haired wuxia male
  template, similar to or stronger than `pe_space_w14`.

## Interpretation

This is a useful negative result. PE-token-space K/V initialization plus
pairwise PE-token retrieval is not enough when the frozen SigLIP feature path
does not encode the anime identity attributes that matter here. The mean
PE-retrieval loss also stayed near the configured margin, which suggests the
student descriptors did not meaningfully become more discriminative during this
short pilot.

The next branch should stop adding adapter-only objectives around frozen
SigLIP tokens. It should either train the image feature calibrator/encoder with
a stronger multi-positive retrieval objective, or use Qwen/PE teacher features
to produce explicit identity/palette/prop tokens before denoising.
