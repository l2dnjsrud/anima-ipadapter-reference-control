# C108 Qwen teacher embedding alignment 계획

작성일: 2026-06-15 KST

## 출발점

C107은 C106 QwenVL teacher score를 hard-negative manifest로 옮겨 bounded 학습까지 성공했다. 하지만 실제 ComfyUI generation gate에서는 `no_ip`보다 개선됐을 뿐 current best `blend_species_face`를 넘지 못했다.

핵심 실패 가설은 C107이 teacher signal을 `positive vs negative score/margin`으로만 전달했고, target image가 가진 QwenVL embedding 자체를 직접 맞추지는 않았다는 점이다.

## C108 가설

C108은 reference image embedding을 adapter token으로 바꾼 뒤, pooled adapter token이 target image의 QwenVL embedding에 직접 가까워지도록 `teacher_alignment_loss`를 추가한다.

기존 C107 구성과의 차이:

- C107: `base denoising + contrastive prediction margin + ref/wrong retrieval margin`
- C108: `base denoising + contrastive prediction margin + ref/wrong retrieval margin + target-image QwenVL embedding alignment`

ComfyUI runtime checkpoint format은 바꾸지 않는다. 새 loss는 학습 중에만 사용하며, 저장 checkpoint는 기존 `IPAdapterQwenVL` / `CalibratedIPAdapterQwenVL` state_dict 그대로 저장한다.

## 사용 데이터

- source manifest: `training/manifests/c107_qwen_teacher_distillation_20260613.jsonl`
- C108 manifest: `training/manifests/c108_qwen_teacher_embedding_alignment_20260615.jsonl`
- manifest summary: `training/manifests/c108_qwen_teacher_embedding_alignment_20260615.summary.json`
- image root: `.tmp/c097_siglip_hard_shape_expanded_root`
- rows: `56`
- explicit negative rows: `56`
- heldout rows used: `0`
- missing path count: `0`
- teacher target: `target_image_qwenvl_embedding`

## 구현 파일

- `training/qwenvl_teacher_alignment_loss.py`
- `training/qwenvl_teacher_alignment_step.py`
- `training/qwenvl_teacher_alignment_rows.py`
- `training/qwenvl_teacher_alignment_summary.py`
- `training/qwenvl_teacher_alignment_smoke.py`
- `training/qwenvl_teacher_alignment_cli.py`
- `tests/test_qwenvl_teacher_alignment.py`

## 학습 명령

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
  training/qwenvl_teacher_alignment_cli.py \
  --manifest-path training/manifests/c108_qwen_teacher_embedding_alignment_20260615.jsonl \
  --image-root .tmp/c097_siglip_hard_shape_expanded_root \
  --init-checkpoint-path checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors \
  --output-path checkpoints/anima_qwenvl_ip_adapter_c108_teacher_alignment_b128_0128_20260615.safetensors \
  --steps 128 \
  --resolution 256 \
  --device cuda:0 \
  --max-rows 56 \
  --lr 1e-5 \
  --seed 20260808 \
  --contrastive-weight 0.35 \
  --contrastive-margin 0.05 \
  --retrieval-weight 0.15 \
  --retrieval-margin 0.2 \
  --teacher-weight 0.4 \
  --calibrator-bottleneck-dim 128 \
  --train-calibrator-only
```

## C001 pass gate

- manifest rows > `0`
- heldout rows used == `0`
- missing path count == `0`
- training rows_loaded > `0`
- finite_loss == `true`
- mean_teacher_loss finite
- checkpoint.loadable == `true`
- checkpoint.pe_checkpoint_rejected == `true`

## C001 실행 결과

평가 폴더:

- `eval/c108_qwen_teacher_embedding_alignment_training_20260615/`

결과:

| 항목 | 값 |
|---|---:|
| decision | `proceed_to_c108_generation_gate` |
| rows_loaded | `56` |
| explicit_negative_rows | `56` |
| first_loss | `0.6143436431884766` |
| final_loss | `0.5449631214141846` |
| mean_loss | `0.6113223081920296` |
| mean_teacher_loss | `1.003852569963783` |
| finite_loss | `true` |
| checkpoint.loadable | `true` |
| checkpoint.pe_checkpoint_rejected | `true` |

작은 2-step smoke에서도 같은 init checkpoint로 실제 모델 로드, target image QwenVL embedding 준비,
checkpoint 저장/검증이 통과했다. 다만 `--train-calibrator-only`는 init checkpoint 안에
`feature_calibrator.*` weight가 이미 있어야 한다. 이번 init checkpoint는 해당 조건을 만족하지만,
plain QwenVL adapter를 init으로 바꾸면 `training/qwenvl_smoke_checkpoint.py`의 guard에서 실패하는 것이
정상이다.

## C002 generation gate

C108 checkpoint가 만들어지면 isolated ComfyUI/API에서 다음 세 variant를 비교한다.

- `no_ip`
- current best `blend_species_face`
- `c108_qwen_teacher_alignment_w14`

필수 산출물:

- `summary.json`
- `contact_sheet_train.jpg`
- `contact_sheet_heldout.jpg`
- `pe_similarity_metrics.json`
- `qwenvl_similarity_metrics.json`
- `metric_rollup.json`
- `visual_audit.md/json`
- `report.md`

승격 기준은 C107과 같다. `no_ip` 대비 개선만으로는 부족하고, current best `blend_species_face`를 PE/QwenVL 지표와 육안 감사에서 실질적으로 넘어야 한다.
