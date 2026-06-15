# c095 SigLIP Feature-Bridge 실험 계획

## 목적

c094는 shape supervision을 추가해도 hard-shape contact sheet에서 frog, chibi,
mascot, non-human profile reference가 여전히 초록색 인간 얼굴 쪽으로 무너졌다.
따라서 c095는 같은 adapter-only continuation을 반복하지 않고, SigLIP shallow/deep
feature를 `CrossLayerEncoder`로 fuse한 뒤 `TimeResampler`로 들어가기 직전에 작은
residual bridge를 붙여 본다.

핵심 질문은 다음이다.

- frozen SigLIP feature 자체를 바꾸지 않아도 fused token 공간에 작은 bridge를 넣으면
  non-human/shape cue가 더 잘 보존되는가?
- bridge-only 학습만으로 c094보다 의미 있게 좋아지는가?
- 그래도 실패하면 adapter/bridge 수준이 아니라 SigLIP image encoder checkpoint 또는
  더 큰 데이터 확장이 필요한가?

## 구현 범위

- 새 모듈: `siglip_feature_bridge.py`
- checkpoint variant 감지 분리: `siglip_checkpoint_variants.py`
- bridge checkpoint key: `feature_bridge.norm`, `feature_bridge.down`,
  `feature_bridge.up`
- bridge 위치: `CrossLayerEncoder(shallow, deep)` 출력 뒤, `TimeResampler` 입력 전
- 학습 방식: `feature_bridge.*`만 `requires_grad=True`
- 금지: `feature_calibrator.*` 재사용, full adapter continuation, heldout07 학습

## 데이터와 시작 checkpoint

- train manifest:
  `training/manifests/c093_siglip_qwen_target_anti_collapse_20260613.jsonl`
- image root:
  `.tmp/c093_anti_collapse_root`
- init checkpoint:
  `checkpoints/anima_siglip_ip_adapter_c094_shape_supervised_0064_20260613.safetensors`
- output checkpoint:
  `checkpoints/anima_siglip_ip_adapter_c095_feature_bridge_b128_0096_20260613.safetensors`

이 manifest는 c093/c094에서 사용한 hard-shape train rows이며, `heldout07`은 평가
전용으로 남긴다.

## 학습 게이트

예정 명령:

```bash
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_shape_contrastive_cli.py \
  --manifest-path training/manifests/c093_siglip_qwen_target_anti_collapse_20260613.jsonl \
  --image-root .tmp/c093_anti_collapse_root \
  --init-checkpoint-path checkpoints/anima_siglip_ip_adapter_c094_shape_supervised_0064_20260613.safetensors \
  --output-path checkpoints/anima_siglip_ip_adapter_c095_feature_bridge_b128_0096_20260613.safetensors \
  --steps 96 \
  --max-rows 10 \
  --resolution 256 \
  --device cuda:0 \
  --lr 8e-5 \
  --seed 20260695 \
  --contrastive-weight 0.25 \
  --contrastive-margin 0.08 \
  --shape-weight 0.20 \
  --reference-shape-weight 0.35 \
  --feature-bridge-bottleneck-dim 128 \
  --train-feature-bridge-only
```

학습 통과 조건:

- loss가 finite
- rows_loaded = 10
- explicit_negative_rows = 10
- `train_feature_bridge_only = true`
- `trainable_parameter_names`가 전부 `feature_bridge.*`
- checkpoint loadable = true
- PE checkpoint rejected = true

## 생성 게이트

학습이 통과하면 isolated ComfyUI API gate를 포트 8121에서 실행한다.

비교 variant:

- `no_ip`
- `siglip_pilot_w14`
- `c089_shape_w14`
- `c091_feature_calibrator_w14`
- `c092_qwen_target_w10`
- `c092_qwen_target_w14`
- `c093_anti_collapse_w14`
- `c094_shape_supervised_w14`
- `c095_feature_bridge_w08`
- `c095_feature_bridge_w10`
- `c095_feature_bridge_w12`
- `c095_feature_bridge_w14`

평가 산출물:

- `summary.json`
- `contact_sheet_hard_shape.jpg`
- `shape_metrics.json`
- `metric_rollup.json`
- `pixel_nonblank_audit.json`
- `visual_audit.md`
- `visual_audit.json`
- `report.md`
- `cleanup_port_8121.txt`

## 판정

후보 승격은 다음 조건을 본다.

- c095 평균 uplift가 c094보다 의미 있게 높을 것
- heldout07 non-human profile이 c094보다 뚜렷하게 나아질 것
- c095 blank-like row가 없을 것
- contact sheet에서 frog/chibi/mascot/non-human rows가 초록 인간 얼굴로 붕괴하지
  않을 것

실패하면 `c095_feature_bridge_not_promoted_requires_siglip_encoder_finetune_or_data_expansion`
으로 기록하고, 다음 단계는 더 깊은 SigLIP image encoder fine-tuning 또는 데이터 확장으로
넘어간다.
