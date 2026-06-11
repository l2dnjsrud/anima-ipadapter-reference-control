# SigLIP c026 Single-Character Token-Separation Runtime Evaluation

- Contact sheet: `eval/siglip_runtime_quality_20260611_c026_single_character_token_sep_runtime/contact_sheet.jpg`
- Summary: `eval/siglip_runtime_quality_20260611_c026_single_character_token_sep_runtime/summary.json`
- Runtime: isolated ComfyUI API on `127.0.0.1:8116`, GPU0, repo custom node.
- Columns: reference / no_ip / clean32_w1 / clean32_w14 / token_w1 / token_w14.

## Inputs

- Train manifest: `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- Held-out manifest: `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- Seed checkpoint: `checkpoints/anima_siglip_ip_adapter_single_character_clean32_pe_query_patch_0512_20260611.safetensors`
- Token-separation checkpoint: `checkpoints/anima_siglip_ip_adapter_single_character_clean32_token_sep_0256_20260611.safetensors`

The token-separation checkpoint continued from the clean32 checkpoint for 256
steps. The run added a token-level reference separation term that penalizes
high cosine similarity between correct-reference and wrong-reference image
tokens.

Observed training summary:

- rows loaded: `32`
- first/final loss: `0.3259349763393402` / `0.19840562343597412`
- mean loss: `0.29138473694911227`
- mean base loss: `0.1944809732667636`
- mean contrastive loss: `0.04216717180679552`
- mean teacher loss: `0.0204733534837942`
- mean token loss: `0.15726202292717062`
- finite loss: `true`
- trainable parameters: `336650396`
- checkpoint loadable: `true`

## Visual Result

Decision: `single_character_token_sep_not_quality_pass`

Single-character testing remains the right gate because it makes adapter
influence easy to read. The token-separation run clearly changes the output and
often separates it from the clean32 template, but the change is not the desired
reference fidelity.

Observed failures:

- `train00`: token variants shift toward a bald blue-robed stern template and
  miss the reference arm pose, hair shape, and warmer palette.
- `train07`: token variants become different black-robed characters and still
  miss the reference face and composition.
- `train14`: token variants miss the scruffy beard/aged face and move toward a
  bald stern blue-robed template.
- `train23`: token variants miss glasses, fan, scholarly face, and flat comic
  palette.
- `heldout02`: token variants miss the bald elder with beard and forehead dots.
- `heldout05`: token variants miss the screaming cropped bearded face.
- `heldout07`: token variants miss the green demon/red-eye profile and collapse
  into human male templates.

## Interpretation

The token-separation loss is technically working: it makes references less
collapsed in token space and changes the ComfyUI outputs. It is not sufficient
as a quality path because it does not anchor the separated tokens to the
correct semantic identity, props, palette, or facial structure.

Do not launch a longer token-separation-only run unchanged. The next useful
training branch should keep the same single-character train/held-out gate, but
add a semantic reference anchor before scaling. Candidate anchors include an
anime/VL image-feature teacher, explicit identity/palette/prop attributes,
paired reference-target supervision, or a trainable image encoder/calibrator
that is optimized for reference retrieval before adapter training.
