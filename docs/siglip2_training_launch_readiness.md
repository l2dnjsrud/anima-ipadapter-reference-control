# SigLIP2 Training Launch Readiness

Date: 2026-06-10

## Verdict

Do not start full SigLIP2/TimeResampler/IPCrossAttn training yet.

The native code path is ready for a dry run, but the real training run is still
blocked by paired metadata and storage/runtime approval. The PE IP-Adapter-only
line-art colorization track is stopped; line-art colorization needs spatial
conditioning in addition to reference control.

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
- No local `training_pairs*.jsonl`, pair CSV, or row-level `ref_id`/`tgt_id`/`prompt` metadata was found.

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

Create or obtain the paired metadata. The most likely path is:

1. Inspect Wenaka's original training scripts for the exact pair generation rule.
2. Generate a candidate `training_pairs_final2.jsonl` from local image ids and captions, or obtain the original file.
3. Run `training/siglip_proof.py --pairs-path ... --image-dir ... --rows-to-check 8`.
4. Only after that, implement and launch the real frozen-Anima training loop.
