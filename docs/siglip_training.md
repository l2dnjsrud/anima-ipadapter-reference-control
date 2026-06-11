# SigLIP2 Native Branch And Training Plan

## What landed locally

- Native SigLIP2/IP-Adapter scaffolding lives in `siglip_model.py`, `siglip_checkpoint.py`, and `native_siglip.py`.
- Real frozen-Anima SigLIP2 smoke training lives in `training/siglip_real_smoke.py` with small helpers under `training/siglip_smoke_*.py`.
- The upstream Wenaka `TransformerEncoderLayer(memory=...)` bug is avoided by a real cross-attention fusion layer.
- The upstream loader constructor mismatch is avoided by detecting checkpoint tensor shapes and constructing `IPAdapterSigLIP` with explicit dimensions.
- PE-Core checkpoints are rejected by the SigLIP loader with a clear message instead of being partially loaded as SigLIP.
- `AnimaSigLIPIPAdapterLoader` uses the ComfyUI `ipadapter` model selector, not a raw filesystem path.
- `AnimaSigLIPIPAdapterApply` clones the ComfyUI `MODEL` and installs a PE-style UNet/model wrapper that temporarily patches the live Anima DiT `cross_attn.forward` calls during sampling. The previous `attn2_patch`-only path can execute as a node but is not sufficient for the Anima/Qwen workflow surface.
- A one-step smoke checkpoint was written to `checkpoints/anima_siglip_ip_adapter_smoke_20260610.safetensors` and loads through `siglip_checkpoint.load_siglip_adapter`.

## Dataset facts checked on 2026-06-10

Primary source: `https://huggingface.co/datasets/Wenaka/anima-ip-adapter-dataset`

- Dataset id: `Wenaka/anima-ip-adapter-dataset`
- Commit: `687b09ba08d8fda8c415b43cc0aacf6fc1d2661c`
- Public and ungated.
- Hub API `usedStorage`: `39,190,230,897` bytes, about `36.5 GiB`.
- Direct HEAD requests for `images_00.tar` through `images_13.tar` total `34,190,796,800` bytes, about `31.843 GiB`; the difference is Hub storage metadata and non-tar overhead.
- Files: `.gitattributes` plus `images_00.tar` through `images_13.tar`.
- Dataset Viewer split: `default/train`.
- Preview features: `jpg` image, `__key__` string, `__url__` string.
- Preview rows show 1024 x 1024 JPG images inside the tar shards.
- Dataset Viewer reported preview support, but no full viewer/search/filter/statistics support; `/size` returned a failed `config-size` result, so the Hub file metadata is the reliable size source.

Commands used:

```bash
curl 'https://datasets-server.huggingface.co/is-valid?dataset=Wenaka/anima-ip-adapter-dataset'
curl 'https://datasets-server.huggingface.co/splits?dataset=Wenaka/anima-ip-adapter-dataset'
curl 'https://datasets-server.huggingface.co/first-rows?dataset=Wenaka/anima-ip-adapter-dataset&config=default&split=train'
curl 'https://huggingface.co/api/datasets/Wenaka/anima-ip-adapter-dataset'
curl 'https://huggingface.co/api/datasets/Wenaka/anima-ip-adapter-dataset/tree/main?recursive=false&expand=true'
python - <<'PY'
import urllib.request
total = 0
for idx in range(14):
    request = urllib.request.Request(
        f"https://huggingface.co/datasets/Wenaka/anima-ip-adapter-dataset/resolve/main/images_{idx:02d}.tar",
        method="HEAD",
    )
    with urllib.request.urlopen(request) as response:
        total += int(response.headers["Content-Length"])
print(total)
PY
```

## Dry-run proof

The proof script does not download data and does not start full Anima training. It validates optional Wenaka-style pair rows and runs one synthetic optimization step through:

`SigLIP deep+shallow features -> CrossLayerEncoder -> TimeResampler -> IPCrossAttn`

Run with the existing Anima virtualenv:

```bash
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_proof.py
```

With local paired metadata:

```bash
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_proof.py \
  --pairs-path /path/to/training_pairs_final2.jsonl \
  --image-dir /path/to/extracted/images \
  --rows-to-check 8
```

Expected pair row shape:

```json
{"ref_id": "4279862", "tgt_id": "4279889", "prompt": "clean character reference prompt"}
```

## Local color-panel real smoke

The first real, bounded SigLIP2 smoke was run on 2026-06-10 with the curated
local color-panel dataset:

```text
/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best
```

Generated manifest:

```text
training/manifests/local_color_pairs_pilot_20260610.jsonl
training/manifests/local_color_pairs_pilot_20260610.summary.json
```

Manifest audit summary:

- 1,537 pair rows
- 1,460 train rows and 77 validation rows by deterministic sorted 95/5 split
- every row has `ref_id`, `tgt_id`, and `prompt`
- every referenced JPG and TXT sidecar exists
- malformed missing-caption input fails safely

Real smoke command:

```bash
HF_HUB_DISABLE_XET=1 /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_real_smoke.py \
  --manifest-path training/manifests/local_color_pairs_pilot_20260610.jsonl \
  --image-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  --steps 1 \
  --resolution 256 \
  --device cuda:0 \
  --output-path checkpoints/anima_siglip_ip_adapter_smoke_20260610.safetensors \
  --max-rows 4
```

Observed result:

```json
{
  "steps": 1,
  "rows_loaded": 4,
  "first_loss": 0.1699729710817337,
  "final_loss": 0.1699729710817337,
  "finite_loss": true,
  "trainable_parameters": 335860892,
  "frozen_base_parameters": 2913827059,
  "checkpoint": {
    "output_path": "checkpoints/anima_siglip_ip_adapter_smoke_20260610.safetensors",
    "loadable": true,
    "pe_checkpoint_rejected": true
  }
}
```

The canonical write target
`/data/ai/models/ipadapter/anima_siglip_ip_adapter_smoke_20260610.safetensors`
was attempted after the real training step reached checkpoint save, but the
directory is `root:root` with mode `755`, so `safetensors` returned
`Permission denied (os error 13)`. The repo-local checkpoint is the current
loadable smoke artifact.

## Local color-panel pilot

A bounded 16-step pilot was run on 2026-06-10 from the same local color-panel
manifest, using 32 rows at 256 px resolution:

```bash
HF_HUB_DISABLE_XET=1 /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_real_smoke.py \
  --manifest-path training/manifests/local_color_pairs_pilot_20260610.jsonl \
  --image-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  --steps 16 \
  --resolution 256 \
  --device cuda:0 \
  --output-path checkpoints/anima_siglip_ip_adapter_pilot_20260610.safetensors \
  --max-rows 32
```

Observed result:

- final_loss: `0.1321239024400711`
- mean_loss: `0.22480368381366134`
- finite_loss: `true`
- trainable_parameters: `335860892`
- checkpoint: `checkpoints/anima_siglip_ip_adapter_pilot_20260610.safetensors`
- SigLIP loader: loadable as `IPAdapterSigLIP`
- PE-Core checkpoint: rejected by the SigLIP loader, as intended

Proxy evaluation artifacts:

```text
eval/siglip_color_pilot_20260610/metrics.json
eval/siglip_color_pilot_20260610/report.md
```

Proxy result:

- key_match: `true`
- common_tensors: `255`
- changed_tensors: `142`
- relative_l2_delta: `0.0008295682970535739`
- decision: `scale_after_siglip_workflow_eval`

This proves the pilot moved away from the one-step smoke checkpoint and stayed
checkpoint-compatible. It does not prove visual quality yet because the native
SigLIP ComfyUI/API image-generation workflow has not produced contact sheets.

## What blocks real full training

- No high-quality trained SigLIP2 TimeResampler/IPCrossAttn checkpoint is present in this repo. The current SigLIP checkpoints are a one-step smoke artifact and a 16-step pilot artifact only.
- The Hugging Face dataset is about 36.5 GiB; I did not download it without explicit approval.
- The public dataset appears to contain image tar shards only. Wenaka's training script expects a paired `training_pairs_final2.jsonl` with `ref_id`, `tgt_id`, and `prompt`, which is not exposed in the dataset preview or file list.
- Full training still needs explicit runtime approval, a target output path that the current user can write, and a native SigLIP reference-control contact-sheet quality gate.

## Full-training outline

1. Choose the approved dataset scope: local color-panel pairs, downloaded Wenaka shards, or both.
2. Generate paired metadata with `ref_id`, `tgt_id`, and `prompt`.
3. Run the dry-run proof and local manifest audit against the selected rows.
4. Run bounded real smoke through the frozen Anima DiT/VAE/text-encoder loss path.
5. Save a SigLIP checkpoint with `resampler.time_proj.*`, `resampler.layers.*`, `intermediate_encoder.*`, `ip_cross_attns.*`, and `ip_scales.*` keys.
6. Load it through `AnimaSigLIPIPAdapterLoader`; do not use the PE-Core checkpoint with this loader.
7. Wire the native SigLIP ComfyUI/API workflow, then evaluate against no-IP and PE-Core baselines with contact sheets before calling it usable.

## 2026-06-11 runtime and tuning result

The native SigLIP apply path was changed from an `attn2_patch`-only node to a
PE-style sampling wrapper that temporarily patches the live Anima DiT
`cross_attn.forward` calls. This fixed the zero-effect bug:

- no-IP and `weight=0` are pixel-identical,
- `weight>0` changes generated pixels,
- the node can produce good-looking prompt-aligned comic panels.

However, this does **not** prove a finished reference-control model. The latest
quality run is documented in:

```text
eval/siglip_runtime_quality_20260611_c007_self512_identity/report.md
```

Summary:

- `color64_continue` improved prompt-aligned visual strength when the prompt
  explicitly contained old monk / white beard / prayer beads hints.
- `color64_continue`, `self64_continue`, and `self512_continue` all failed the
  stricter identity test where those identity tokens were removed from the
  prompt.
- The best matched-prompt contact sheet is:
  `eval/siglip_runtime_quality_20260611_c004_color64_matched/contact_sheet.jpg`.
- The identity failure evidence is:
  `eval/siglip_runtime_quality_20260611_c007_self512_identity/contact_sheet.jpg`.

Therefore the current local SigLIP checkpoints should be treated as research
artifacts, not a ready-to-trust IP-Adapter. More short tuning on adjacent panel
pairs is unlikely to solve identity control. The next training step needs
identity-aware or same-image reconstruction data, cached feature training for
longer runs, and a validation gate that removes identity words from the prompt.

## 2026-06-11 one-image overfit gate

A one-image overfit gate was run after the `self512` identity failure. It used
the same `codex_contact_ref03.png` monk image as both reference and target, but
the training and evaluation prompt intentionally removed direct identity words
such as old, bald, white beard, and prayer beads.

Result:

- Checkpoint:
  `checkpoints/anima_siglip_ip_adapter_ref03_overfit1024_20260611.safetensors`
  (local ignored artifact)
- Evidence:
  `eval/siglip_runtime_quality_20260611_c008_ref03_overfit1024_identity/contact_sheet.jpg`
- Report:
  `eval/siglip_runtime_quality_20260611_c008_ref03_overfit1024_identity/report.md`
- Decision: `overfit_pass_generalization_required`

The overfit checkpoint visibly recovered the bald monk face, red beads, robe
color, speech bubble, and crop without prompt-side identity hints. This means
the native SigLIP route is not a proven dead end. The remaining problem is
generalization and scale stability: the current local color/self checkpoints do
not carry identity across held-out prompts, and high overfit weights can still
collapse.

Training code now caches repeated one-row target latents, text embeddings, and
SigLIP features so the next focused overfit/ablation runs do not waste most of
their time recomputing fixed inputs.

## 2026-06-11 identity8 reference sweep

The cache was expanded to precompute repeated target latents, text embeddings,
and SigLIP features for up to 16 rows. This made a small 8-reference
self-reconstruction run practical:

- Manifest:
  `training/manifests/local_color_self_identity8_20260611.jsonl`
- Checkpoint:
  `checkpoints/anima_siglip_ip_adapter_identity8_1024_20260611.safetensors`
  (local ignored artifact)
- Report:
  `eval/siglip_runtime_quality_20260611_c009_identity8_reference_sweep/report.md`
- Contact sheet:
  `eval/siglip_runtime_quality_20260611_c009_identity8_reference_sweep/contact_sheet.jpg`
- Decision: `reference_influence_seen_identity_generalization_incomplete`

The checkpoint changes generated images differently for different references,
including held-out references, so reference influence is no longer limited to
the single-image overfit case. It still does not faithfully recover held-out
identity. The next useful scale-up is a larger balanced self-reconstruction set
with a held-out validation sheet, not another adjacent-panel-only run.

## 2026-06-11 identity128 and fp32 continuation

A 128-reference color self-reconstruction manifest was built from the local
`image_dataset_color_panel_style_v5_best` set:

- Manifest:
  `training/manifests/local_color_self_identity128_20260611.jsonl`
- Selection:
  `eval/identity128_reference_selection_20260611.json`
- bf16 continuation checkpoints:
  `checkpoints/anima_siglip_ip_adapter_identity128_1024_20260611.safetensors`
  and
  `checkpoints/anima_siglip_ip_adapter_identity128_2048_20260611.safetensors`
  (local ignored artifacts)
- bf16 evidence:
  `eval/siglip_runtime_quality_20260611_c010_identity128_reference_sweep/report.md`,
  `eval/siglip_runtime_quality_20260611_c011_identity128_neutral_prompt/report.md`,
  and
  `eval/siglip_runtime_quality_20260611_c012_identity128_2048_neutral_prompt/report.md`

The bf16 identity128 continuations lowered finite losses and changed pixels, but
`ip_scales` stayed effectively unchanged from the earlier identity8 checkpoint.
That made the poor visual result ambiguous: the route might have been weak, or
the trainable adapter might have been numerically stalled.

The training path was then changed to keep the trainable adapter in fp32 during
bounded CUDA smoke training. A fp32 1024-step continuation from the identity128
2048-step checkpoint produced:

- Checkpoint:
  `checkpoints/anima_siglip_ip_adapter_identity128_fp32_3072_20260611.safetensors`
  (local ignored artifact)
- Evidence:
  `eval/siglip_runtime_quality_20260611_c013_identity128_fp32_neutral_prompt/report.md`
- Pair sheet:
  `eval/siglip_runtime_quality_20260611_c013_identity128_fp32_neutral_prompt/reference_output_pairs.jpg`
- Decision: `fp32_training_moves_weights_but_quality_still_fail`

Runtime inspection showed the fp32 run moved 253 of 255 adapter tensors, with
relative L2 movement `0.01685` from the previous checkpoint versus `0.00440` for
the preceding bf16 continuation. This confirms there was a real training
precision/stability problem. However, the c013 sheet still collapses most
references into similar two-character court/interior scenes and does not recover
train or held-out reference identity/layout reliably.

Current conclusion: the native SigLIP route is not impossible in principle,
because the one-image overfit passed and fp32 training can move the adapter. But
the current frozen SigLIP2 encoder plus this small adapter/objective is not a
ready path to high-quality generalized Anima reference control. Further work
should shift to a stronger anime/VL image encoder or a pair objective that trains
the image encoder/adapter together, rather than only extending short local
SigLIP2 adapter tuning.

## 2026-06-11 reference-swap contrastive objective

The next ablation added a bounded reference-swap objective. For the same
target/noise/prompt, the model is trained so prediction with the correct
reference is closer to the target than prediction with a deterministic wrong
reference:

```text
loss = correct_reference_mse
     + contrastive_weight * relu(correct_mse - wrong_mse + margin)
```

Implementation:

- `training/siglip_reference_loss.py`
- `training/siglip_prepared_cache.py`
- `training/siglip_contrastive_smoke.py`
- `training/siglip_contrastive_cli.py`
- `tests/test_siglip_reference_loss.py`

Smoke evidence:

- `checkpoints/anima_siglip_ip_adapter_identity128_contrastive_0064_20260611.safetensors`
  (local ignored artifact)
  - `steps=64`, `rows_loaded=16`, `mean_contrastive_loss=0.04606`,
    checkpoint loadable.
- `checkpoints/anima_siglip_ip_adapter_identity128_contrastive_0512_20260611.safetensors`
  (local ignored artifact)
  - `steps=512`, `rows_loaded=32`, `mean_contrastive_loss=0.09111`,
    checkpoint loadable.
  - Tensor movement from the fp32 baseline: 253/255 tensors changed,
    relative L2 `0.00723`, max scale movement `0.00056`.

Visual evidence:

- `eval/siglip_runtime_quality_20260611_c014_identity128_contrastive_neutral_prompt/report.md`
- `eval/siglip_runtime_quality_20260611_c014_identity128_contrastive_neutral_prompt/reference_output_pairs.jpg`
- Decision: `contrastive_improves_reference_distance_but_quality_still_fail`

Compared with c013, c014 lowers mean pixel distance to several references and
increases reference-dependent variation. It also avoids some of the prior
two-character conversation collapse. However, it still does not faithfully
recover train or held-out identity/layout/character count. This suggests the
objective was a useful correction, but the frozen SigLIP2 adapter path still
needs either a stronger anime/VL encoder or a trainable image-encoder stage to
reach the requested high-quality reference-control target.

## 2026-06-11 contrastive weight sweep

The c014 contrastive checkpoint was evaluated through the isolated ComfyUI API at
weights `0.6`, `1.0`, `1.4`, and `1.8` across the same three train and three
held-out references:

- Evidence:
  `eval/siglip_runtime_quality_20260611_c015_contrastive_weight_sweep/report.md`
- Contact sheet:
  `eval/siglip_runtime_quality_20260611_c015_contrastive_weight_sweep/contact_sheet.jpg`
- Decision: `weight_sweep_no_usable_quality`

The sweep confirms that the checkpoint has a tunable reference-influence
channel. Higher weights can lower simple pixel distance to the reference,
especially for sky/crowd/color palette cases. The visual result still fails the
target bar: identity, layout, panel composition, and character count are not
stable, and higher weights often introduce generic group scenes or excessive
speech bubbles.

Current conclusion after c015: frozen SigLIP2 adapter tuning has learned a weak
style/composition pressure, but not the high-fidelity reference-control channel
the user wants. More inference-scale tuning on this checkpoint is not a credible
route to production quality. The next implementation path needs a trainable
image encoder/calibration stage or a stronger anime/VL encoder while preserving
the native ComfyUI patch surface.
