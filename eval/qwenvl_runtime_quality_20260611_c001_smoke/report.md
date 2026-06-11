# QwenVL Training Smoke 2026-06-11

## Smoke Checkpoint

- Command: `training/qwenvl_smoke_cli.py`
- Output checkpoint:
  `checkpoints/anima_qwenvl_ip_adapter_smoke_0002_20260611.safetensors`
- Steps: `2`
- Rows loaded: `2`
- Resolution: `128`
- First/final loss: `0.29517` / `0.30803`
- Mean loss: `0.30160`
- Finite loss: `true`
- Checkpoint loadable through QwenVL loader: `true`
- PE checkpoint rejected by QwenVL loader: `true`

## Identity128 Candidate

- Command: `training/qwenvl_smoke_cli.py`
- Output checkpoint:
  `checkpoints/anima_qwenvl_ip_adapter_identity128_0064_20260611.safetensors`
- Init checkpoint: none
- Steps: `64`
- Rows loaded: `16`
- Resolution: `256`
- First/final loss: `0.22585` / `0.10778`
- Mean loss: `0.20032`
- Finite loss: `true`
- Trainable parameters: `307648156`
- Frozen base parameters: `4947838963`
- Checkpoint loadable through QwenVL loader: `true`
- PE checkpoint rejected by QwenVL loader: `true`

## Decision

`training_smoke_pass_quality_unproven`

The QwenVL training/runtime path can load the public
`Qwen/Qwen3-VL-Embedding-2B` encoder, train a native Anima adapter, save a
QwenVL-marked checkpoint, and load it through the QwenVL checkpoint loader. This
does not prove image quality; visual contact-sheet evaluation is recorded in
the c001/c002/c003 QwenVL runtime-quality reports.
