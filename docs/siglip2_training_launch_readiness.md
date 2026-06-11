# SigLIP2 Training Launch Readiness

Date: 2026-06-10

## Verdict

Do not start broad full-dataset SigLIP2/TimeResampler/IPCrossAttn training yet.

The native code path has passed a real one-step frozen-Anima SigLIP2 smoke
training run and a bounded 16-step color-panel pilot on the local curated
color-panel dataset. That proves the training path can load
Anima/Qwen/VAE/SigLIP2, train only the adapter modules, write loadable
SigLIP-family checkpoints, and reject the PE-Core checkpoint with the SigLIP
loader.

This is not a high-quality usable reference-control checkpoint yet. The native
runtime now affects generated pixels, but the current checkpoints fail blind
identity/reference control when prompt-side identity hints are removed. The next
training step is a targeted one-image overfit gate, not a broad full-dataset
launch. The PE IP-Adapter-only line-art colorization track is stopped; line-art
colorization needs spatial conditioning in addition to reference control.

## What Is Ready

- Native ComfyUI SigLIP nodes exist:
  - `AnimaSigLIPIPAdapterLoader`
  - `AnimaSigLIPEncodeImage`
  - `AnimaSigLIPIPAdapterApply`
- The loader uses the ComfyUI `ipadapter` model selector.
- The apply node patches the ComfyUI `MODEL` through a PE-style UNet/model
  wrapper that temporarily patches live Anima DiT cross-attention. This replaced
  the initial `attn2_patch`-only surface after live API evidence showed that
  execution alone could still match no-IP pixels.
- The synthetic proof exercises:
  - SigLIP deep and shallow feature fusion
  - `CrossLayerEncoder`
  - `TimeResampler`
  - `IPCrossAttn`
- Real smoke training exists in `training/siglip_real_smoke.py`.
- Local pair metadata was generated from:
  - `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`
- The smoke checkpoint exists at:
  - `checkpoints/anima_siglip_ip_adapter_smoke_20260610.safetensors`
- The 16-step pilot checkpoint exists at:
  - `checkpoints/anima_siglip_ip_adapter_pilot_20260610.safetensors`
- Pilot proxy evaluation exists at:
  - `eval/siglip_color_pilot_20260610/metrics.json`
  - `eval/siglip_color_pilot_20260610/report.md`
- Focused tests cover:
  - valid SigLIP checkpoint loading
  - malformed checkpoint rejection
  - PE-Core checkpoint rejection by the SigLIP loader

## Required Inputs For Real Training

The full training stage needs all of these before launch:

```text
Anima DiT checkpoint
Qwen text encoder checkpoint
Qwen image VAE checkpoint
SigLIP2 image encoder checkpoint
Wenaka/anima-ip-adapter-dataset image shards or an equivalent extracted image directory
paired training metadata with ref_id, tgt_id, and prompt
validation split metadata
GPU/runtime budget approval
```

Expected pair row shape:

```json
{"ref_id": "4279862", "tgt_id": "4279889", "prompt": "clean character reference prompt"}
```

The public Hugging Face dataset currently exposes image tar shards only in the
top-level file listing. The pair metadata used by Wenaka-style training is not
present in the visible dataset tree and was not found locally.

## Dataset Inventory

Source:

```text
https://huggingface.co/datasets/Wenaka/anima-ip-adapter-dataset
```

Checked facts:

- Dataset id: `Wenaka/anima-ip-adapter-dataset`
- Commit: `687b09ba08d8fda8c415b43cc0aacf6fc1d2661c`
- Public and ungated.
- Hub API `usedStorage`: `39,190,230,897` bytes, about `36.5 GiB`.
- Top-level files: `.gitattributes`, `images_00.tar` through `images_13.tar`.
- Tar shard total by HEAD requests: `34,190,796,800` bytes, about `31.843 GiB`.
- Local extracted image candidates exist under:
  - `/home/wktwin/anima-lora-training-bundle/image_dataset/`
- The current preferred local training source is the curated color-panel winner:
  - `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`
- Generated local color-panel metadata:
  - `training/manifests/local_color_pairs_pilot_20260610.jsonl`
  - `training/manifests/local_color_pairs_pilot_20260610.summary.json`
- The local color-panel manifest contains 1,537 pair rows with deterministic
  train/validation counts of 1,460 and 77.
- No original Wenaka `training_pairs*.jsonl`, pair CSV, or row-level
  `ref_id`/`tgt_id`/`prompt` metadata was found.

## Dry-Run Commands

Use the existing Anima virtualenv:

```bash
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_proof.py
```

With paired metadata once available:

```bash
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_proof.py \
  --pairs-path /path/to/training_pairs_final2.jsonl \
  --image-dir /path/to/extracted/images \
  --rows-to-check 8
```

Focused validation:

```bash
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python -m pytest tests/test_native_siglip.py -q
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python -m ruff check native_siglip.py siglip_checkpoint.py siglip_model.py training/siglip_proof.py tests/test_native_siglip.py
```

## Real Smoke Command

Use this for a bounded one-step training smoke, not a quality run:

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

Observed smoke summary:

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

## Bounded Pilot Result

Command:

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

Observed pilot summary:

```json
{
  "steps": 16,
  "rows_loaded": 32,
  "first_loss": 0.1699729710817337,
  "final_loss": 0.1321239024400711,
  "mean_loss": 0.22480368381366134,
  "finite_loss": true,
  "trainable_parameters": 335860892,
  "checkpoint": {
    "output_path": "checkpoints/anima_siglip_ip_adapter_pilot_20260610.safetensors",
    "loadable": true,
    "pe_checkpoint_rejected": true
  }
}
```

Proxy comparison against the smoke checkpoint:

- key_match: `true`
- common_tensors: `255`
- changed_tensors: `142`
- relative_l2_delta: `0.0008295682970535739`
- decision: `scale_after_siglip_workflow_eval`

Interpretation: scale is justified only after the native SigLIP UI/API
workflow can generate contact sheets. The pilot is a training/runtime proof and
a checkpoint-movement proof, not a visual-quality proof.

## Output Checkpoint Contract

The trained adapter should be saved as a SigLIP-family checkpoint, not a PE-Core
checkpoint. Target location for ComfyUI use:

```text
/data/ai/models/ipadapter/anima_siglip_ip_adapter_quality_YYYYMMDD.safetensors
```

Required checkpoint families and tensor groups:

```text
resampler.latents
resampler.time_proj.*
resampler.layers.*
intermediate_encoder.*
ip_cross_attns.*
ip_scales.*
```

The target ComfyUI loader is:

```text
AnimaSigLIPIPAdapterLoader
```

Current write-location note: writing the smoke checkpoint directly under
`/data/ai/models/ipadapter/` failed because that directory is `root:root` mode
`755` for this shell. The repo-local smoke checkpoint is loadable; copying or
training directly into the ComfyUI model directory needs a writable destination
or a user-run privileged copy step.

## Evaluation Gate

A checkpoint is not considered usable until it passes all of these:

- ComfyUI package import registers the SigLIP loader, encoder, and apply nodes.
- The checkpoint appears in the `ipadapter_name` model selector.
- API workflow generates nonblank images.
- UI workflow imports as a normal ComfyUI graph, not as a one-shot subprocess runner.
- Reference-control contact sheet shows a measurable improvement over no-IP baseline.
- Line-art colorization is evaluated separately with spatial control enabled.

## 2026-06-11 Native Workflow Check

Native SigLIP UI/API workflow validation was added in:

```text
workflows/anima_ipadapter_siglip_native_reference.json
eval/siglip_native_workflow_eval_20260611/report.md
```

That first run was blocked because the live `ipadapter_name` selector did not
include `anima_siglip_ip_adapter_pilot_20260610.safetensors`. A later isolated
ComfyUI run used the repo-local custom node and `tools/comfyui_extra_model_paths.yaml`
to expose repo-local checkpoints under the normal `ipadapter` model selector.

The runtime no-op was then fixed by replacing the `attn2_patch`-only path with a
PE-style model wrapper that temporarily patches the live Anima DiT
`cross_attn.forward` calls during sampling.

Evidence:

- `eval/siglip_runtime_quality_20260611_c001/contact_sheet.jpg`
  - `weight=0` is pixel-identical to no-IP.
  - `weight>0` changes generated pixels, proving the native SigLIP path is no
    longer a no-op.
- `eval/siglip_runtime_quality_20260611_c004_color64_matched/contact_sheet.jpg`
  - prompt-aligned generations can look visually strong when prompt text
    contains the reference identity attributes.
- `eval/siglip_runtime_quality_20260611_c007_self512_identity/contact_sheet.jpg`
  - blind identity/reference control still fails when direct prompt hints such
    as old, bald, white beard, and prayer beads are removed.

Latest decision:

```text
partial_pass_training_required
```

Current interpretation after `c007`: the native runtime is usable for
experimentation, but the checkpoint has not learned a reliable image-reference
channel. A one-image overfit gate was needed to determine whether the adapter
could learn reference identity at all when the prompt withholds identity
details.

The follow-up `c008` overfit gate passed:

```text
eval/siglip_runtime_quality_20260611_c008_ref03_overfit1024_identity/report.md
eval/siglip_runtime_quality_20260611_c008_ref03_overfit1024_identity/contact_sheet.jpg
```

This proves the native SigLIP route is not a hard runtime impossibility. It
does not prove generalized reference-control quality.

The follow-up `c009` identity8 sweep also shows reference-dependent image
changes across several references, including held-out references, but held-out
identity recovery remains incomplete:

```text
eval/siglip_runtime_quality_20260611_c009_identity8_reference_sweep/report.md
eval/siglip_runtime_quality_20260611_c009_identity8_reference_sweep/contact_sheet.jpg
```

The larger `identity128` runs made the limitation clearer. bf16 continuations
changed generated pixels but left `ip_scales` effectively frozen; fp32 training
fixed the parameter-movement issue, moving 253 of 255 adapter tensors and
changing scale values. The resulting c013 visual sheet still failed the quality
bar:

```text
eval/siglip_runtime_quality_20260611_c013_identity128_fp32_neutral_prompt/report.md
eval/siglip_runtime_quality_20260611_c013_identity128_fp32_neutral_prompt/reference_output_pairs.jpg
```

Interpretation: there was a real training precision problem, but fixing it did
not make frozen SigLIP2 adapter tuning produce usable generalized reference
control. The current route should be considered a research baseline, not the
main path for a high-quality checkpoint.

A reference-swap contrastive objective was then added and tested. It trains the
same target/noise/prompt against both a correct and a deterministic wrong
reference so the adapter is penalized when the wrong reference is competitive.
The 512-step c014 checkpoint moved parameters and improved several reference
distance metrics, but still failed visual identity/layout control:

```text
eval/siglip_runtime_quality_20260611_c014_identity128_contrastive_neutral_prompt/report.md
eval/siglip_runtime_quality_20260611_c014_identity128_contrastive_neutral_prompt/reference_output_pairs.jpg
```

Interpretation: objective pressure helps, but is not sufficient by itself.
Frozen SigLIP2 remains a weak main path for the requested high-quality Anima
reference-control checkpoint.

The same c014 checkpoint was also swept at inference weights `0.6`, `1.0`,
`1.4`, and `1.8`:

```text
eval/siglip_runtime_quality_20260611_c015_contrastive_weight_sweep/report.md
eval/siglip_runtime_quality_20260611_c015_contrastive_weight_sweep/contact_sheet.jpg
```

Interpretation: inference scale is not the missing fix. It can intensify style
or layout pressure, but it does not recover stable identity/layout control and
often worsens generic-scene collapse.

The next branch added an identity-initialized trainable feature calibrator in
front of the native SigLIP adapter. The checkpoint loader now preserves
backward compatibility for old SigLIP checkpoints and reconstructs calibrated
checkpoints when `feature_calibrator.*` tensors are present. Two bounded
calibrated continuations were tested:

```text
eval/siglip_runtime_quality_20260611_c016_calibrated_contrastive_neutral_prompt/report.md
eval/siglip_runtime_quality_20260611_c016_calibrated_contrastive_neutral_prompt/contact_sheet.jpg
eval/siglip_runtime_quality_20260611_c017_calibrated_contrastive0576_neutral_prompt/report.md
eval/siglip_runtime_quality_20260611_c017_calibrated_contrastive0576_neutral_prompt/contact_sheet.jpg
```

Interpretation: calibration is a valid trainable path and produces a better
early signal than pure frozen-adapter tuning, but it is not a completed quality
solution. The 64-step checkpoint improves some rows; the 576-step continuation
then drifts toward generic scene averages. A broad full-dataset run should not
start until the objective or encoder signal changes again.

The next objective change used PE-teacher distillation. A frozen PE adapter
provided the teacher denoising prediction for the same target/noise/timestep
while the native SigLIP adapter optimized base MSE, contrastive MSE, and teacher
MSE. The implementation is covered by:

```text
training/pe_teacher_features.py
training/pe_teacher_distillation.py
training/siglip_teacher_smoke.py
training/siglip_teacher_cli.py
tests/test_pe_teacher_distillation.py
```

The 64-step candidate completed with finite losses:

```text
checkpoint: checkpoints/anima_siglip_ip_adapter_identity128_pe_teacher_0064_20260611.safetensors
steps: 64
rows_loaded: 16
mean_loss: 0.23133
mean_base_loss: 0.20576
mean_contrastive_loss: 0.04206
mean_teacher_loss: 0.03011
```

ComfyUI evidence:

```text
eval/siglip_runtime_quality_20260611_c018_pe_teacher_distill/report.md
eval/siglip_runtime_quality_20260611_c018_pe_teacher_distill/contact_sheet.jpg
eval/siglip_runtime_quality_20260611_c018_pe_teacher_weight_sweep/contact_sheet.jpg
```

Interpretation: PE-teacher distillation is valid code and does affect generated
images, but it still fails the visual quality gate. Outputs become
reference-dependent, yet they do not reliably inherit reference color,
identity, panel layout, or composition. The c018 weight sweep shows no simple
runtime weight that recovers quality: low weights return toward no-IP and high
weights distort scene content. This makes a broad frozen-SigLIP2 training run a
poor use of GPU time unless the image encoder signal changes.

Qwen3-VL embedding was then probed as the next encoder signal:

```text
eval/qwen3vl_embedding_probe_20260611/report.md
eval/qwen3vl_embedding_probe_20260611/summary.json
```

The public `Qwen/Qwen3-VL-Embedding-2B` checkpoint loads locally and emits
`2048`-dim normalized image embeddings. On six identity128 color-panel
references the off-diagonal cosine range was `0.424949` to `0.737467`, with
mean `0.563116`. This is not yet adapter quality evidence, but it proves the
candidate encoder is available and does not collapse the reference set into one
generic vector. Any QwenVL adapter smoke should start from a `2048`-dim input
contract or explicitly document a projection/truncation stage.

The first QwenVL adapter scaffold now exists:

```text
qwenvl_model.py
qwenvl_checkpoint.py
tests/test_qwenvl_adapter.py
```

It intentionally uses a separate `qwenvl_family` checkpoint marker. The QwenVL
detector rejects PE-Core and SigLIP checkpoints, while the SigLIP detector
rejects QwenVL-marked checkpoints. This prevents the next ComfyUI branch from
silently accepting the wrong family and producing another no-op-looking run.

The native QwenVL ComfyUI node surface is also registered:

```text
eval/qwenvl_native_workflow_eval_20260611/report.md
eval/qwenvl_native_workflow_eval_20260611/object_info_AnimaQwenVLIPAdapterLoader.json
eval/qwenvl_native_workflow_eval_20260611/object_info_AnimaQwenVLEncodeImage.json
eval/qwenvl_native_workflow_eval_20260611/object_info_AnimaQwenVLIPAdapterApply.json
```

The nodes are visible through ComfyUI `object_info`, but this remains a
registration/runtime-surface proof only. A trained QwenVL checkpoint is still
required before no-IP vs adapter generation quality can be evaluated.

## Stop Conditions

Stop and ask before proceeding if any of these are true:

- The pair metadata is still missing.
- Dataset shard download or extraction would be required.
- The projected disk requirement is not approved.
- Full training would start a long GPU run.
- The validation split is missing.
- The checkpoint cannot be loaded through `AnimaSigLIPIPAdapterLoader`.
- A small multi-reference identity run cannot reproduce held-out references
  after the one-image overfit gate has passed; in that case this exact SigLIP
  adapter route should be paused in favor of a stronger anime/VL encoder path.

## Next Executable Step

Move from tiny reference influence to a real generalization proof. The next path
is:

1. Build a larger balanced self-reconstruction manifest with distinct
   characters, color palettes, close-up/wide panels, and a held-out split.
2. Reuse the cached target/text/SigLIP feature path for short runs, and add
   persistent feature caches before any long run.
3. Train from `self512` or `identity8` for a bounded larger run.
4. Evaluate with prompts that remove identity/color tokens and require the image
   encoder to carry those attributes.
5. Because the larger fp32 checkpoint still cannot recover held-out references,
   and PE-teacher distillation also fails the visual gate, stop short local
   frozen-SigLIP tuning as the main path and move to a Qwen-VL/anime-image-
   encoder based reference-control plan. The first Qwen branch should target
   `Qwen/Qwen3-VL-Embedding-2B` with a `2048`-dim adapter input.
6. The QwenVL image encoder node, bounded denoising smoke, and QwenVL
   contact-sheet evaluation have now been run:
   - `eval/qwenvl_runtime_quality_20260611_c001_smoke/report.md`
   - `eval/qwenvl_runtime_quality_20260611_c002_identity128_weight_sweep/contact_sheet.jpg`
   - `eval/qwenvl_runtime_quality_20260611_c003_contrastive_weight_sweep/contact_sheet.jpg`
7. The result is not a quality pass. QwenVL adapter-only tuning changes the
   output, but the outputs still collapse toward a dataset-average yellow-robed
   interior scene and do not preserve reference-specific identity/layout/color.
   Do not spend a long GPU run on the same frozen-encoder adapter-only recipe.
8. Next executable step: design a trainable image-encoder/calibrator or
   teacher-supervised objective that explicitly penalizes reference collapse
   before launching another broad run.
9. A QwenVL feature-calibration continuation has now also been tested:
   - `eval/qwenvl_runtime_quality_20260611_c004_calibrated_contrastive_weight_sweep/report.md`
   - `eval/qwenvl_runtime_quality_20260611_c004_calibrated_contrastive_weight_sweep/contact_sheet.jpg`
10. The result is still not a quality pass. The calibrator trains and changes
    images, but the contact sheet keeps collapsing references toward the same
    yellow-robed street/interior scene. Do not launch a longer broad run on
    this frozen-QwenVL adapter/calibrator recipe without a changed objective,
    stronger teacher signal, or trainable/anime-domain image encoder stage.
11. A PE-style query patch correction has now been tested for the native
    SigLIP/QwenVL runtime and SigLIP training path:
    - `eval/siglip_runtime_quality_20260611_c019_pe_query_patch_weight_sweep/report.md`
    - `eval/siglip_runtime_quality_20260611_c019_pe_query_patch_weight_sweep/contact_sheet.jpg`
    - `eval/siglip_runtime_quality_20260611_c020_pe_query_patch_trained_weight_sweep/report.md`
    - `eval/siglip_runtime_quality_20260611_c020_pe_query_patch_trained_weight_sweep/contact_sheet.jpg`
12. This correction is worth keeping because it makes the native path use the
    same cross-attention query geometry as PE-Core and increases
    reference-dependent variation. It is not a quality pass yet. The short
    c020 continuation still misses held-out palette/layout/identity and
    distorts at high weights, so the next launch should use this corrected
    patch surface but still require a stronger held-out reference gate before
    broad training.
13. A single-character diagnostic has now been run to separate page-layout
    difficulty from adapter generalization:
    - `eval/siglip_runtime_quality_20260611_c021_single_character_diagnostic/report.md`
    - `eval/siglip_runtime_quality_20260611_c021_single_character_diagnostic/contact_sheet.jpg`
14. Result: `single_character_diagnostic_not_quality_pass`. Even with
    single-character references and a fixed solo portrait prompt, the current
    SigLIP query-patch checkpoint changes outputs but does not reliably preserve
    beard/age, blue robe palette, gold hair/fire palette, or stable identity.
    This makes the issue broader than page complexity; the next training launch
    needs a stronger reference-discrimination objective, trainable
    feature/encoder adaptation, or an anime-domain image encoder path.
