# Anima IP-Adapter Reference-Control

ComfyUI custom node package for the trained Anima PE-Core IP-Adapter
reference-control candidate.

This repo is now cloneable directly into `ComfyUI/custom_nodes/`. The primary
path is a native ComfyUI graph: load the Anima DiT normally, encode a reference
image, apply the PE IP-Adapter as a `MODEL` patch, then sample with standard
ComfyUI sampler/VAE nodes. The older one-shot runner node is kept only for CLI
reproduction of the shipped evaluation sheet.

## Status

- Candidate checkpoint: `checkpoints/anima_ip_adapter_quality_20260610.safetensors`
- Recommended inference scale: `--ip_scale 1.0`
- Recommended sampler for reproducing the shipped evaluation sheet: `er_sde`
- Evaluation status: PASS
- Generated evaluation images: 40
- Nonblank check: PASS
- Best-scale mean PE uplift over no-IP: `0.0973858163`
- Best-scale improved rate: `0.875`

The model is a layout/style reference-control candidate measured with the same
PE-Core family used by the local IP-Adapter path. It is not yet a verified
character-identity recovery model.

## ComfyUI Install

Run this on the ComfyUI machine:

```bash
cd /data/ai/comfyui02/custom_nodes
sudo git clone https://github.com/l2dnjsrud/anima-ipadapter-reference-control.git
cd anima-ipadapter-reference-control
sudo git lfs pull
```

For an existing clone:

```bash
cd /data/ai/comfyui02/custom_nodes/anima-ipadapter-reference-control
sudo git pull
sudo git lfs pull
```

Put the trained adapter where ComfyUI model selector nodes can see it:

```text
/data/ai/models/ipadapter/anima_ip_adapter_quality_20260610.safetensors
```

Restart ComfyUI after cloning or pulling. On the local machine used for this
package, the runner defaults assume:

```text
Anima source: /home/wktwin/anima-lora-training-bundle/anima_lora
Anima python: /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python
Comfy models: /data/ai/models
```

If those paths differ, set `ANIMA_LORA_ROOT`, `ANIMA_LORA_PYTHON`, or
`ANIMA_COMFY_MODELS_ROOT` before starting ComfyUI.

## ComfyUI Workflow

Import one of these JSON workflows in ComfyUI:

```text
workflows/anima_ipadapter_pe_native_reference.json
workflows/anima_ipadapter_siglip_native_reference.json
workflows/anima_ipadapter_reference_generate.json
workflows/anima_ipadapter_contactsheet_ref03_ersde.json
```

Use `anima_ipadapter_pe_native_reference.json` first. It is the normal ComfyUI
workflow shape:

```text
LoadImage
  -> AnimaPEEncodeImage
AnimaPEIPAdapterLoader + UNETLoader
  -> AnimaPEIPAdapterApply
  -> CFGGuider / BasicScheduler / SamplerCustomAdvanced
  -> VAEDecode
  -> SaveImage
```

The PE loader uses the `ipadapter_name` model selector. Select:

```text
anima_ip_adapter_quality_20260610.safetensors
```

`anima_ipadapter_siglip_native_reference.json` is the native SigLIP2 pilot
workflow. It uses:

```text
LoadImage
  -> AnimaSigLIPEncodeImage
AnimaSigLIPIPAdapterLoader + UNETLoader
  -> AnimaSigLIPIPAdapterApply
  -> CFGGuider / BasicScheduler / SamplerCustomAdvanced
  -> VAEDecode
  -> SaveImage
```

For that workflow, the loader selector must list:

```text
anima_siglip_ip_adapter_pilot_20260610.safetensors
```

As of the 2026-06-11 recovery run, the SigLIP apply node patches the live Anima
DiT through a PE-style sampling wrapper instead of relying on the old
`attn2_patch`-only path. The zero-effect bug is fixed: `weight=0` matches no-IP
pixels and `weight>0` changes generated images. This still does not make the
SigLIP checkpoints finished reference-control models. Prompt-aligned generations
can look good, but the stricter identity test still fails after `color64`,
`self64`, and `self512` continuation runs. See:

```text
eval/siglip_native_workflow_eval_20260611/report.md
eval/siglip_runtime_quality_20260611_c007_self512_identity/report.md
```

A stricter one-image overfit gate then passed:

```text
eval/siglip_runtime_quality_20260611_c008_ref03_overfit1024_identity/report.md
eval/siglip_runtime_quality_20260611_c008_ref03_overfit1024_identity/contact_sheet.jpg
```

That run recovered the ref03 monk identity without direct identity prompt words,
so the SigLIP path is not a hard runtime dead end. It is still an overfit proof,
not a generalized ready-to-trust reference-control checkpoint.

The legacy workflows remain available:

```text
workflows/anima_ipadapter_reference_generate.json
workflows/anima_ipadapter_contactsheet_ref03_ersde.json
```

`anima_ipadapter_reference_generate.json` calls the Anima CLI in a subprocess.
`anima_ipadapter_contactsheet_ref03_ersde.json` reproduces the verified
contact-sheet `ref03 / seed 20260610 / ip_scale 1.0` case. On the local
ComfyUI02 install it expects `codex_contact_ref03.png` in the input directory.

For contact-sheet-like outputs, keep the node sampler at `er_sde`. The Anima
CLI defaults to `euler`, but the packaged evaluation sheet was generated with
`er_sde`.

The evaluation markdown and contact sheet are not ComfyUI workflows; they remain
under `eval/` only as evidence.

## Layout

```text
__init__.py
nodes.py
runner.py
native_pe.py
native_pe_models.py
native_pe_patch.py
native_pe_runtime.py
native_siglip.py
siglip_checkpoint.py
siglip_model.py
checkpoints/
  anima_ip_adapter_quality_20260610.safetensors
workflows/
  anima_ipadapter_pe_native_reference.json
  anima_ipadapter_reference_generate.json
docs/
  siglip_training.md
eval/reference_eval_quality_20260610_c003/
  report.md
  summary.json
  scores.csv
  manifest.json
  run_eval.sh
  contact_sheet.jpg
  images/
tools/
  reference_eval.py
tests/
  test_ip_adapter_reference_eval.py
logs/
  anima_ip_adapter_quality_20260610.progress.jsonl
evidence/
  G001-C001-quality-preflight.cli.txt
  G001-C002-strong-train.tmux.txt
  G001-C003-quality-eval.cli.txt
  final_quality_gate.json
  ulw-goals.json
  ulw-ledger.jsonl
```

## Quick Use

Run from a prepared `anima_lora` checkout with the Anima base model, Qwen text
encoder, VAE, PE-Core encoder, and project virtualenv available.

```bash
./.venv/bin/python inference.py \
  --dit models/diffusion_models/anima-base-v1.0.safetensors \
  --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
  --vae models/vae/qwen_image_vae.safetensors \
  --attn_mode flash \
  --prompt "masterpiece, best quality, score_7, safe. 1girl, solo, cafe, holding a coffee cup." \
  --negative_prompt "low quality, blurry, bad anatomy, text, watermark" \
  --seed 20260610 \
  --infer_steps 20 \
  --guidance_scale 3.5 \
  --flow_shift 3.0 \
  --sampler er_sde \
  --image_size 960 1120 \
  --save_path output/tests/ipadapter_reference \
  --ip_adapter_weight /home/wktwin/anima-ipadapter-reference-control/checkpoints/anima_ip_adapter_quality_20260610.safetensors \
  --ip_image /path/to/reference.png \
  --ip_scale 1.0
```

## Evaluation

The main evaluation report is:

```text
eval/reference_eval_quality_20260610_c003/report.md
```

The contact sheet for visual inspection is:

```text
eval/reference_eval_quality_20260610_c003/contact_sheet.jpg
```

The original command manifest and exact generation script are preserved in:

```text
eval/reference_eval_quality_20260610_c003/manifest.json
eval/reference_eval_quality_20260610_c003/run_eval.sh
```

`tools/reference_eval.py` is a snapshot of the evaluation harness from the
source `anima_lora` checkout. It depends on that project layout and is included
for reproducibility and future reruns.

## Line-Art Colorization Decision

PE IP-Adapter-only line-art colorization is not the right next target. The
adapter works as reference-control, but line-art colorization needs spatial
conditioning to preserve panel layout and local line structure. The recorded
decision and ComfyUI evidence are in:

```text
docs/line_colorization_decision.md
eval/line_color_dataset_pair_easycontrol_ip_20260610/report.md
```

Continue high-quality reference-control work through the separate
SigLIP2/TimeResampler/IPCrossAttn training stage. Use EasyControl/ControlNet-like
conditioning for line-art colorization.

## SigLIP2 Training Readiness

The next training stage is documented in:

```text
docs/siglip2_training_launch_readiness.md
docs/ipadapter_reference_research.md
```

Current state: the native SigLIP2/TimeResampler/IPCrossAttn code path,
synthetic trainability proof, real frozen-Anima smoke training, continuation
loading, and native ComfyUI API execution all pass. The available SigLIP
checkpoints are still research artifacts, not high-quality reference-control
models. The best current result is prompt-aligned style/composition influence;
blind identity transfer is not solved by the local adjacent-panel or
self-reconstruction pilots.

For local color-panel tests, generate Wenaka-style pairs from the current color
winner:

```bash
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python tools/generate_pair_manifest.py \
  /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  --output training/manifests/local_color_pairs_pilot_20260610.jsonl
```

The recorded local manifest has 1,537 pair rows with a deterministic 1,460/77
train/validation split.

Then validate the row shape without starting real Anima training:

```bash
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_proof.py \
  --pairs-path training/manifests/local_color_pairs_pilot_20260610.jsonl \
  --image-dir /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  --rows-to-check 8
```

For the bounded real smoke:

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

Smoke output:

```text
checkpoints/anima_siglip_ip_adapter_smoke_20260610.safetensors
```

## Node Behavior

The recommended PE path has three nodes:

- `AnimaPEIPAdapterLoader`: loads a PE-Core checkpoint from the ComfyUI
  `ipadapter` model selector.
- `AnimaPEEncodeImage`: encodes a ComfyUI `IMAGE` reference with PE-Core.
- `AnimaPEIPAdapterApply`: clones the `MODEL`, patches Anima cross-attention
  during sampling, and restores the patch after each call.

`Anima IP-Adapter Generate` is the legacy runner node. It accepts a ComfyUI
`IMAGE` reference, writes it to a temporary PNG, runs `inference.py` with
`--ip_adapter_weight`, `--ip_image`, and `--ip_scale`, then loads the newest PNG
from the selected output subdirectory and returns it as a ComfyUI image.

This is intentionally not wired through `comfyui_ipadapter_plus`; that custom
node targets standard SD/SDXL IP-Adapter checkpoints, while this checkpoint is
for the Anima DiT PE-Core IP-Adapter path.

## SigLIP2 Branch

The Wenaka-style SigLIP2/TimeResampler/IPCrossAttn branch is implemented as
native scaffolding in `native_siglip.py`, `native_siglip_runtime.py`,
`siglip_model.py`, and `siglip_checkpoint.py`. It rejects the PE-Core checkpoint
clearly and uses the same ComfyUI `ipadapter_name` selector style. The current
pilot checkpoint is `checkpoints/anima_siglip_ip_adapter_pilot_20260610.safetensors`.
Continuation checkpoints from 2026-06-11 are local experiment artifacts and are
ignored by git unless deliberately promoted. `docs/siglip_training.md` records
the smoke evidence, runtime fix, continuation runs, and why identity-control
requires a better training set and validation gate.

## Training Summary

- Data: 796 train images, 20 validation images
- Training: 8 epochs, 6368 steps
- Encoder: PE-Core, 1024-dimensional features
- Checkpoint metadata: `ss_network_spec=ip_adapter`, `ss_encoder=pe`
- IP tensors: 28 gate tensors, 56 key/value tensors

Full logs and LazyCodex/ULW evidence are in `logs/` and `evidence/`.

## Git LFS

The `.safetensors` checkpoint is stored with Git LFS because it is larger than
GitHub's normal file limit.
