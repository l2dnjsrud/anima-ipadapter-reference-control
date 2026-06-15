# C107 Qwen Teacher Distillation Training

작성일: 2026-06-14

## 목적

C106에서 QwenVL teacher가 C097 hard-shape positive/explicit-negative 쌍을 평균 margin `0.21772855200937813`, AUC `1.0`으로 분리했기 때문에, 이 teacher 판단을 `neg_id` hard-negative contrastive/retrieval 학습 manifest로 전달했다.

## 입력

- manifest: `training/manifests/c107_qwen_teacher_distillation_20260613.jsonl`
- manifest summary: `training/manifests/c107_qwen_teacher_distillation_20260613.summary.json`
- image root: `.tmp/c097_siglip_hard_shape_expanded_root`
- teacher score: `eval/c106_qwen_teacher_feature_distillation_20260613/qwenvl_pair_scores.json`
- heldout rows used: `0`
- teacher source: `C106`

## 실행 명령

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/qwenvl_contrastive_cli.py \
  --manifest-path training/manifests/c107_qwen_teacher_distillation_20260613.jsonl \
  --image-root .tmp/c097_siglip_hard_shape_expanded_root \
  --init-checkpoint-path checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors \
  --output-path checkpoints/anima_qwenvl_ip_adapter_c107_qwen_teacher_calibrator_b128_0128_20260613.safetensors \
  --steps 128 --resolution 256 --device cuda:0 --max-rows 56 --lr 1e-5 --seed 20260707 \
  --contrastive-weight 0.45 --contrastive-margin 0.05 \
  --retrieval-weight 0.25 --retrieval-margin 0.2 \
  --calibrator-bottleneck-dim 128 --train-calibrator-only
```

## 결과

| 항목 | 값 |
|---|---:|
| steps | `128` |
| rows_loaded | `56` |
| first_loss | `0.9018970727920532` |
| final_loss | `0.19634446501731873` |
| mean_loss | `0.24020752124488354` |
| mean_base_loss | `0.17100206477334723` |
| mean_contrastive_loss | `0.049944027297897264` |
| mean_retrieval_loss | `0.18692258093506098` |
| finite_loss | `true` |
| trainable_parameters | `528384` |
| frozen_base_parameters | `4947838963` |
| checkpoint.loadable | `true` |
| checkpoint.pe_checkpoint_rejected | `true` |
| explicit_negative_rows | `56` |

출력 checkpoint:

```text
checkpoints/anima_qwenvl_ip_adapter_c107_qwen_teacher_calibrator_b128_0128_20260613.safetensors
```

## 판단

결정: `c107_training_passed_generation_gate_pending`

학습 자체는 통과했다. 다만 C107은 아직 실제 reference-control 품질을 증명하지 않았다. 다음 C002에서 isolated ComfyUI/API generation gate로 `no_ip`, current best `blend_species_face`, C107 후보를 clean32+heldout8에서 비교한다.
