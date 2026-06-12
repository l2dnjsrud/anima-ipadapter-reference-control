# QwenVL c075 Tag-Positive Training

작성일: 2026-06-13

## 목적

c074에서 수동 검수로 확보한 direct-green/non-human target positive 10장을 실제 QwenVL adapter 학습에 넣었을 때, 기존 `blend_species_face` baseline이 놓치던 녹색 비인간/괴물형 reference-control 신호가 강화되는지 확인하기 위한 bounded 실험이다.

이 단계의 목표는 최종 품질 판정이 아니라, ComfyUI 생성 gate로 넘길 수 있는 loadable checkpoint를 만드는 것이다.

## 입력 데이터

- manifest: `training/manifests/c075_tag_positive_direct_green_20260612.jsonl`
- manifest summary: `training/manifests/c075_tag_positive_direct_green_20260612.summary.json`
- image root: `.tmp/c075_tag_positive_direct_green_root`
- target positive images: `10`
- target positive training rows: `40`
- source training rows: `80`
- total rows: `120`
- heldout rows used: `0`
- committed external raw images: `0`

외부 raw 이미지는 `.tmp/` 아래 local scratch에만 두고 커밋하지 않는다.

## 실행 명령

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/qwenvl_contrastive_cli.py \
  --manifest-path training/manifests/c075_tag_positive_direct_green_20260612.jsonl \
  --image-root .tmp/c075_tag_positive_direct_green_root \
  --init-checkpoint-path checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors \
  --output-path checkpoints/anima_qwenvl_ip_adapter_c075_tag_positive_calibrator_b128_0064_20260612.safetensors \
  --steps 64 \
  --resolution 256 \
  --device cuda:0 \
  --max-rows 120 \
  --lr 1e-5 \
  --seed 20260675 \
  --contrastive-weight 0.35 \
  --contrastive-margin 0.05 \
  --retrieval-weight 0.2 \
  --retrieval-margin 0.2 \
  --calibrator-bottleneck-dim 128 \
  --train-calibrator-only \
  --instruction "Represent this reference for strict visual identity retrieval in a manhwa panel. Prioritize non-human species, monster or demon traits, facial structure, profile silhouette, beard and headwear, skin tone, glowing eyes, hand props, fan or weapon cues, speech bubble context, pose crop, costume palette, and line/color style." \
  > eval/qwenvl_c075_tag_positive_training_20260612/train_stdout.txt 2>&1
```

## 결과

| 항목 | 값 |
|---|---:|
| steps | `64` |
| rows_loaded | `120` |
| first_loss | `0.2053343505` |
| final_loss | `0.2455184013` |
| mean_loss | `0.2621962277` |
| mean_base_loss | `0.2056205160` |
| mean_contrastive_loss | `0.0484961928` |
| mean_retrieval_loss | `0.1980102286` |
| finite_loss | `true` |
| trainable_parameters | `528,384` |
| frozen_base_parameters | `4,947,838,963` |
| checkpoint.loadable | `true` |
| checkpoint.pe_checkpoint_rejected | `true` |

출력 checkpoint:

```text
checkpoints/anima_qwenvl_ip_adapter_c075_tag_positive_calibrator_b128_0064_20260612.safetensors
```

## 복구 기록

첫 실행은 체크포인트 저장 중 루트 파티션 여유 공간 부족으로 실패했다.

- 실패 로그: `eval/qwenvl_c075_tag_positive_training_20260612/train_stdout_no_space_failure.txt`
- 실패 원인: `No space left on device`
- 처리: 깨진 partial checkpoint를 삭제하고, 재생성 가능한 ignored ComfyUI/eval raw PNG cache를 정리한 뒤 같은 명령을 재실행했다.
- 재실행 결과: checkpoint는 `safetensors`로 정상 loadable이고 QwenVL loader family guard도 통과했다.

## 판단

결정: `qwenvl_c075_tag_positive_training_passed_generation_gate_pending`

c075는 target-positive direct-green 데이터를 포함한 heldout-safe 학습, finite loss, calibrator-only freeze, checkpoint loadability를 통과했다. 실제 품질 통과 여부는 다음 `eval/qwenvl_c075_tag_positive_gate_20260612/` ComfyUI API generation gate에서 `no_ip`, current best `blend_species_face`, `c075`를 contact sheet와 PE/QwenVL metric으로 비교해야 한다.
