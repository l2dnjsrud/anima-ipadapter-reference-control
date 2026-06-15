# QwenVL Mixed Continuation Training c055

작성일: 2026-06-12

## 목적

c054 생성 gate에서 c053 checkpoint는 노승, 녹색 괴물처럼 특수 trait 일부를 더 잘 반영했지만, 이전 `single_character_retrieval` checkpoint보다 aggregate PE/QwenVL metric이 낮았다. c055는 이 metric regression을 줄이기 위해 c052 positive identity rows만 쓰지 않고, 기존 clean32 train self-pair와 c052 reviewed positive identity pair를 섞어 짧게 이어 학습한 bounded experiment다.

이 실험의 목표는 생성 품질 통과가 아니라 다음 c056 generation gate로 넘길 수 있는 학습/checkpoint 후보를 만드는 것이다.

## 입력 데이터

- clean train source: `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- positive source: `training/manifests/c052_positive_identity_pairs_20260612.jsonl`
- mixed manifest: `training/manifests/c055_qwenvl_mixed_clean32_c052_positive_20260612.jsonl`
- manifest summary: `training/manifests/c055_qwenvl_mixed_clean32_c052_positive_20260612.summary.json`
- clean train rows: `32`
- c052 positive rows: `58`
- total rows: `90`
- heldout rows used: `0`

## 실행 명령

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/qwenvl_contrastive_cli.py \
  --manifest-path training/manifests/c055_qwenvl_mixed_clean32_c052_positive_20260612.jsonl \
  --image-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  --init-checkpoint-path checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors \
  --output-path checkpoints/anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors \
  --steps 64 \
  --resolution 256 \
  --device cuda:0 \
  --max-rows 90 \
  --lr 3e-6 \
  --seed 20260655 \
  --contrastive-weight 0.45 \
  --contrastive-margin 0.05 \
  --retrieval-weight 0.15 \
  --retrieval-margin 0.2 \
  > eval/qwenvl_c055_mixed_training_20260612/train_stdout.txt 2>&1
```

## 결과

| 항목 | 값 |
|---|---:|
| steps | `64` |
| rows_loaded | `90` |
| first_loss | `0.2682350278` |
| final_loss | `0.1592893749` |
| mean_loss | `0.2055846578` |
| mean_base_loss | `0.1528775918` |
| mean_contrastive_loss | `0.0499786948` |
| mean_retrieval_loss | `0.2014443586` |
| finite_loss | `true` |
| trainable_parameters | `308,176,540` |
| frozen_base_parameters | `4,947,838,963` |
| checkpoint_loadable | `true` |
| PE checkpoint rejected by QwenVL loader | `true` |

출력 checkpoint:

```text
checkpoints/anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors
```

파일 크기는 약 `1.2G`다. 레포 정책상 `checkpoints/*qwenvl*.safetensors`는 git에서 제외되므로 checkpoint는 로컬 artifact로 유지한다.

## 판단

결정: `qwenvl_c055_mixed_training_smoke_passed_generation_gate_pending`

c055는 학습 surface, loss finite, checkpoint loadability, family guard를 통과했다. c053보다 final loss는 낮지만, 이 수치만으로 reference-control 품질을 주장하지 않는다. 실제 품질 판단은 다음 c056 ComfyUI generation/contact-sheet gate에서 `qwen_prev_retrieval_w14`, `qwen_c052_w14`, `qwen_c055_w1/w14`를 함께 비교해야 한다.

