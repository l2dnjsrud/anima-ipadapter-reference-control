# Anima IP-Adapter Reference-Control Candidate

This repository packages the trained PE-Core IP-Adapter candidate and its
evaluation evidence separately from the main `anima_lora` working tree.

## Status

- Candidate checkpoint: `checkpoints/anima_ip_adapter_quality_20260610.safetensors`
- Recommended inference scale: `--ip_scale 1.0`
- Evaluation status: PASS
- Generated evaluation images: 40
- Nonblank check: PASS
- Best-scale mean PE uplift over no-IP: `0.0973858163`
- Best-scale improved rate: `0.875`

The model is a layout/style reference-control candidate measured with the same
PE-Core family used by the local IP-Adapter path. It is not yet a verified
character-identity recovery model.

## Layout

```text
checkpoints/
  anima_ip_adapter_quality_20260610.safetensors
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
