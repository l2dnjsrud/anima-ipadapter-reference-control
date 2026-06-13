# QwenVL c080 Paired Direct-Green Training

작성일: 2026-06-13

## 목적

c079는 c074/c078 direct-green 이미지를 `ref_id == tgt_id`로 학습해서 녹색/비인간 속성은 조금 강화했지만, 참조별 identity 다양성은 유지하지 못했다. c080은 같은 synthetic-positive 반복을 멈추고, c074 Neeko 계열 real direct-green 샘플을 `ref_id != tgt_id`인 paired supervision으로 바꿔 reference identity가 다른 target view로 전달되는지 확인한다.

## Manifest

- manifest: `training/manifests/c080_paired_direct_green_identity_20260613.jsonl`
- summary: `training/manifests/c080_paired_direct_green_identity_20260613.summary.json`
- scratch image root: `.tmp/c080_paired_direct_green_identity_root`
- c074 pair source: `10`
- c074 paired training rows: `80`
- direct self-pair rows: `0`
- c078 unpaired positive count: `23`
- c078 training rows: `0`
- guard/proxy rows: `36`
- source rows: `80`
- total rows: `196`
- heldout rows used: `0`

c078 synthetic direct-green 이미지는 target-positive 후보로는 유효하지만 같은 identity의 다른 target view가 없으므로 이번 paired 학습에는 넣지 않았다. 이 결정을 summary에 `c078_training_rows=0`으로 남겼다.

## Training

- init checkpoint: `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- output checkpoint: `checkpoints/anima_qwenvl_ip_adapter_c080_paired_direct_green_b128_0128_20260613.safetensors`
- steps: `128`
- max rows: `196`
- resolution: `256`
- contrastive weight: `0.35`
- retrieval weight: `0.2`
- calibrator bottleneck dim: `128`
- train calibrator only: `true`

## Result

- rows loaded: `196`
- first loss: `0.3953106403`
- final loss: `0.2474229336`
- mean loss: `0.2970622344`
- finite loss: `true`
- trainable parameters: `528384`
- frozen base parameters: `4947838963`
- checkpoint loadable: `true`
- PE checkpoint rejected: `true`

## Decision

결정: `qwenvl_c080_paired_direct_green_training_passed_generation_gate_pending`

학습 자체는 통과했다. 다음 판단은 ComfyUI API generation gate에서 `no_ip`, current best `blend_species_face`, c075, c079, c080을 같은 clean32+heldout8 및 direct-green focus 기준으로 비교해 결정한다.
