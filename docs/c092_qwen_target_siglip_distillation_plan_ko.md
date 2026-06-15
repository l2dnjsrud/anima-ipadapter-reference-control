# c092 Qwen Target SigLIP Distillation Plan

## 목적

c091에서 SigLIP feature-calibrator-only route는 c089와 거의 같은 수준에 머물렀고, QwenVL hard-shape baseline인 `c087_expanded_crop_positive_w14`를 넘지 못했다. c092는 frozen SigLIP feature 보정 반복을 중단하고, QwenVL baseline이 만든 hard-shape target 이미지를 SigLIP adapter의 실제 target image로 사용해 더 강한 supervised signal을 줄 수 있는지 확인한다.

## 가설

`c087_expanded_crop_positive_w14` 생성 결과를 target image로 materialize하면, SigLIP adapter가 PE/edge/silhouette teacher보다 QwenVL baseline의 hard-shape/non-human output manifold에 더 직접적으로 맞춰질 수 있다.

## 데이터 경계

- source summary: `eval/c091_siglip_feature_calibrator_hard_shape_gate_20260613/summary.json`
- teacher target: `c087_expanded_crop_positive_w14`
- train rows: `crop_pair00` to `crop_pair09`
- heldout excluded from training: `heldout07`
- output manifest: `training/manifests/c092_qwen_target_distillation_20260613.jsonl`
- scratch image root: `.tmp/c092_qwen_target_distillation_root`

`heldout07`은 non-human side-profile monster failure case이므로 학습에서 제외하고 generation gate에서만 본다.

## 실행 계획

1. c091 summary와 baseline candidate mapping에서 reference image와 Qwen teacher target image를 materialize한다.
2. `training/siglip_real_smoke_cli.py`로 c089 checkpoint에서 full adapter continuation을 짧게 실행한다.
3. output checkpoint를 isolated ComfyUI native SigLIP loader에 노출한다.
4. c091과 같은 hard-shape gate에서 `no_ip`, `siglip_pilot_w14`, `c089_shape_w14`, `c091_feature_calibrator_w14`, `c092_qwen_target_w10/w14`, Qwen baseline을 비교한다.
5. c092가 c089/c091을 넘고 heldout07에서 더 나은 non-human shape를 보이는지 확인한다.

## Stop Gate

- manifest total rows가 10이 아니면 중단한다.
- `heldout07`이 train manifest에 들어가면 중단한다.
- training loss가 non-finite이면 중단한다.
- checkpoint loadability 또는 PE checkpoint rejection guard가 실패하면 중단한다.
- ComfyUI generation 후 generated PNG/API/history count가 sample x variant와 다르면 중단한다.

## 품질 판정

품질 통과 기준은 `c092`가 c089/c091을 의미 있게 넘는 것이다. 최소 기준은 c089 대비 mean uplift `+0.01` 이상 개선이다. 최종 고품질 promotion은 Qwen baseline과 visual audit까지 함께 보며, 특히 heldout07 non-human side-profile shape가 개선되어야 한다.
