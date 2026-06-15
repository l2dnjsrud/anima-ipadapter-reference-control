# QwenVL c060 Failure-Focused Continuation Training

작성일: 2026-06-12

## 목적

c058/c059에서 가장 좋은 runtime 후보는 `blend_prev14_c05504`였지만, final quality pass는 아니었다. 반복 실패는 exact pose/crop, speech bubble, hand/fan prop, non-human/special silhouette였다. c060은 새 외부 데이터 없이 clean32 train rows를 실패 속성 기준으로 upweight하고, c052 reviewed positive identity rows를 유지해 이전 retrieval checkpoint에서 bounded continuation한 실험이다.

## 입력 데이터

- clean train source: `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- positive source: `training/manifests/c052_positive_identity_pairs_20260612.jsonl`
- c058 gate summary: `eval/qwenvl_c055_larger_blend_gate_20260612_c058/summary.json`
- c060 manifest: `training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.jsonl`
- c060 manifest summary: `training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.summary.json`
- clean32 rows: `32`
- failure repeated rows: `64`
- c052 positive rows: `58`
- total rows: `154`
- heldout rows used: `0`

## 실행 명령

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/qwenvl_contrastive_cli.py \
  --manifest-path training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.jsonl \
  --image-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  --init-checkpoint-path checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors \
  --output-path checkpoints/anima_qwenvl_ip_adapter_c060_failure_focused_retrieval_0096_20260612.safetensors \
  --steps 96 \
  --resolution 256 \
  --device cuda:0 \
  --max-rows 154 \
  --lr 2e-6 \
  --seed 20260660 \
  --contrastive-weight 0.40 \
  --contrastive-margin 0.05 \
  --retrieval-weight 0.25 \
  --retrieval-margin 0.2 \
  > eval/qwenvl_c060_failure_focused_training_20260612/train_stdout.txt 2>&1
```

## 결과

| 항목 | 값 |
|---|---:|
| steps | `96` |
| rows_loaded | `154` |
| first_loss | `0.1931381524` |
| final_loss | `0.2131887674` |
| mean_loss | `0.2391075663` |
| mean_base_loss | `0.1691654883` |
| mean_contrastive_loss | `0.0500044253` |
| mean_retrieval_loss | `0.1997612324` |
| finite_loss | `true` |
| trainable_parameters | `308,176,540` |
| frozen_base_parameters | `4,947,838,963` |
| checkpoint_loadable | `true` |
| PE checkpoint rejected by QwenVL loader | `true` |

출력 checkpoint:

```text
checkpoints/anima_qwenvl_ip_adapter_c060_failure_focused_retrieval_0096_20260612.safetensors
```

## 판단

결정: `qwenvl_c060_training_gate_passed_generation_gate_pending`

c060은 데이터/학습/checkpoint gate를 통과했다. 다만 final loss가 first loss보다 낮지는 않으므로, loss 수치만으로 품질 개선을 주장하지 않는다. 실제 판단은 다음 ComfyUI API generation gate에서 `no_ip`, `prev_w14`, `blend_prev14_c05504`, `c060_w14`를 같은 clean32+heldout8 기준으로 비교해 결정한다.
