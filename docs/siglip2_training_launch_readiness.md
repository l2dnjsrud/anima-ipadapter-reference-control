# SigLIP2 Training Launch Readiness

Date: 2026-06-10

## Verdict

Do not start full SigLIP2/TimeResampler/IPCrossAttn training yet.

The native code path has passed a real one-step frozen-Anima SigLIP2 smoke
training run and a bounded 16-step color-panel pilot on the local curated
color-panel dataset. That proves the training path can load
Anima/Qwen/VAE/SigLIP2, train only the adapter modules, write loadable
SigLIP-family checkpoints, and reject the PE-Core checkpoint with the SigLIP
loader.

This is not a high-quality usable reference-control checkpoint yet. Full
quality training still needs explicit runtime approval, a writable final model
location, and native SigLIP contact-sheet evaluation against no-IP and PE-Core
baselines. The PE IP-Adapter-only line-art colorization track is stopped;
line-art colorization needs spatial conditioning in addition to reference
control.

## What Is Ready

- Native ComfyUI SigLIP nodes exist:
  - `AnimaSigLIPIPAdapterLoader`
  - `AnimaSigLIPEncodeImage`
  - `AnimaSigLIPIPAdapterApply`
- The loader uses the ComfyUI `ipadapter` model selector.
- The apply node patches the ComfyUI `MODEL` through an `attn2_patch`.
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

## Stop Conditions

Stop and ask before proceeding if any of these are true:

- The pair metadata is still missing.
- Dataset shard download or extraction would be required.
- The projected disk requirement is not approved.
- Full training would start a long GPU run.
- The validation split is missing.
- The checkpoint cannot be loaded through `AnimaSigLIPIPAdapterLoader`.

## Next Executable Step

Move from checkpoint proof to visual proof. The next path is:

1. Build a normal native SigLIP ComfyUI API workflow and UI workflow using
   `AnimaSigLIPIPAdapterLoader`, `AnimaSigLIPEncodeImage`, and
   `AnimaSigLIPIPAdapterApply`.
2. Make `/data/ai/models/ipadapter/` writable for the final SigLIP artifact, or
   choose a repo-local output plus a manual copy command.
3. Load the pilot checkpoint through `AnimaSigLIPIPAdapterLoader`.
4. Generate contact sheets against no-IP and PE-Core baselines.
5. Continue to longer training only if reference appearance improves without
   layout collapse.
