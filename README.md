# Anima IP-Adapter Reference-Control

ComfyUI custom node package for the trained Anima PE-Core IP-Adapter
reference-control candidate.

This repo is now cloneable directly into `ComfyUI/custom_nodes/`. The node is a
runner node: it calls the verified `anima_lora/inference.py` path in a separate
Python process, then returns the generated image back to ComfyUI.

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
workflows/anima_ipadapter_reference_generate.json
workflows/anima_ipadapter_contactsheet_ref03_ersde.json
```

`anima_ipadapter_reference_generate.json` is the generic reference-control UI.
`anima_ipadapter_contactsheet_ref03_ersde.json` reproduces the verified
contact-sheet `ref03 / seed 20260610 / ip_scale 1.0` case. On the local
ComfyUI02 install it expects `codex_contact_ref03.png` in the input directory.

Both workflows are:

```text
Load Image -> Anima IP-Adapter Generate -> Save Image
```

The `Anima IP-Adapter Generate` node uses ComfyUI-style model selectors:

```text
ipadapter_name
dit_name
text_encoder_name
vae_name
```

Select `anima_ip_adapter_quality_20260610.safetensors` for `ipadapter_name`.
The runner resolves the selected names through ComfyUI `folder_paths` instead
of exposing raw checkpoint paths as text widgets.

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
checkpoints/
  anima_ip_adapter_quality_20260610.safetensors
workflows/
  anima_ipadapter_reference_generate.json
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

## Node Behavior

`Anima IP-Adapter Generate` accepts a ComfyUI `IMAGE` reference, writes it to a
temporary PNG, runs `inference.py` with `--ip_adapter_weight`, `--ip_image`, and
`--ip_scale`, then loads the newest PNG from the selected output subdirectory
and returns it as a ComfyUI image.

This is intentionally not wired through `comfyui_ipadapter_plus`; that custom
node targets standard SD/SDXL IP-Adapter checkpoints, while this checkpoint is
for the Anima DiT PE-Core IP-Adapter path.

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
