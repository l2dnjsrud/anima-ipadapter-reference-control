# QwenVL c079 Synthetic-Positive Training

작성일: 2026-06-13

## 목적

c075가 c074 real direct-green target positive 10장만으로는 `blend_species_face`를 넘지 못했기 때문에, c078 synthetic target positive 23장과 c077 guard/proxy 36장을 추가해 QwenVL calibrator-only adapter가 녹색/비인간 reference-control 신호를 더 잘 학습하는지 확인한다.

## 입력 데이터

- manifest: `training/manifests/c079_synthetic_positive_direct_green_20260612.jsonl`
- manifest summary: `training/manifests/c079_synthetic_positive_direct_green_20260612.summary.json`
- image root: `.tmp/c079_synthetic_positive_direct_green_root`
- c074 real target positives: `10`
- c078 synthetic target positives: `23`
- c077 guard/proxy rows: `36`
- target positive training rows: `132`
- guard/proxy training rows: `36`
- source training rows: `80`
- heldout rows used: `0`

Raw external/synthetic images are materialized only under `.tmp/` and are not committed.

## 실행 명령

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/qwenvl_contrastive_cli.py \
  --manifest-path training/manifests/c079_synthetic_positive_direct_green_20260612.jsonl \
  --image-root .tmp/c079_synthetic_positive_direct_green_root \
  --init-checkpoint-path checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors \
  --output-path checkpoints/anima_qwenvl_ip_adapter_c079_synthetic_positive_calibrator_b128_0128_20260612.safetensors \
  --steps 128 \
  --resolution 256 \
  --device cuda:0 \
  --max-rows 248 \
  --lr 1e-5 \
  --seed 20260679 \
  --contrastive-weight 0.35 \
  --contrastive-margin 0.05 \
  --retrieval-weight 0.2 \
  --retrieval-margin 0.2 \
  --calibrator-bottleneck-dim 128 \
  --train-calibrator-only \
  --instruction "Represent this reference for strict visual identity retrieval in a manhwa panel. Prioritize non-human species, monster or demon traits, facial structure, profile silhouette, beard and headwear, skin tone, glowing eyes, hand props, fan or weapon cues, speech bubble context, pose crop, costume palette, and line/color style." \
  > eval/qwenvl_c079_synthetic_positive_training_20260612/train_stdout.txt 2>&1
```

## 결과

| 항목 | 값 |
|---|---:|
| steps | `128` |
| rows_loaded | `248` |
| first_loss | `0.2246620059` |
| final_loss | `0.2041460872` |
| mean_loss | `0.2719467274` |
| mean_base_loss | `0.2155197981` |
| mean_contrastive_loss | `0.0486642895` |
| mean_retrieval_loss | `0.1969721443` |
| finite_loss | `true` |
| trainable_parameters | `528,384` |
| frozen_base_parameters | `4,947,838,963` |
| checkpoint.loadable | `true` |
| checkpoint.pe_checkpoint_rejected | `true` |

출력 checkpoint:

```text
checkpoints/anima_qwenvl_ip_adapter_c079_synthetic_positive_calibrator_b128_0128_20260612.safetensors
```

## 복구 기록

첫 실행은 `/data/ai/models/ipadapter`가 root 소유라 `Permission denied`로 저장에 실패했다.

- 실패 로그: `eval/qwenvl_c079_synthetic_positive_training_20260612/train_stdout_permission_failure.txt`
- 처리: 같은 명령을 재실행하되 출력 checkpoint를 repo의 ignored `checkpoints/` 아래로 변경했다.

## 판단

결정: `qwenvl_c079_synthetic_positive_training_passed_generation_gate_pending`

c079는 heldout-safe manifest, finite loss, calibrator-only freeze, checkpoint loadability를 통과했다. 실제 품질은 다음 `eval/qwenvl_c079_synthetic_positive_gate_20260612/` ComfyUI API generation gate에서 `no_ip`, `blend_species_face`, `c075_tag_positive_w14`, `c079_synthetic_positive_w14`를 clean32+heldout8 및 direct-green focus로 비교해 판단한다.
