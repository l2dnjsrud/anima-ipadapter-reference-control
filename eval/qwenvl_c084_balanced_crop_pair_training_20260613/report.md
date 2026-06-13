# QwenVL c084 Balanced Crop-Pair Training

작성일: 2026-06-13

## 목적

c083 sheet-crop acquisition gate에서 얻은 same-identity direct-green/non-human crop pairs를 균형 샘플링해 QwenVL IP-Adapter calibrator-only 학습 신호로 사용할 수 있는지 확인했다. c080 checkpoint는 generation gate에서 미승격이었으므로, 이번 학습은 실패 template을 이어받지 않도록 `single_character_retrieval_0128` checkpoint에서 다시 시작했다.

## Manifest

- manifest: `training/manifests/c084_balanced_crop_pairs_20260613.jsonl`
- summary: `training/manifests/c084_balanced_crop_pairs_20260613.summary.json`
- scratch image root: `.tmp/c084_balanced_crop_pairs_root`
- source approved pairs: `970`
- selected rows: `80`
- approved groups: `4`
- target-positive candidates: `74`
- materialized candidate count: `57`
- group counts: `{"c082_frog_yokai_guard": 24, "c082_goblin_mage": 24, "c082_green_oni_scout": 24, "c082_jade_lizard_monk": 8}`
- max pairs per group/source-pair: `24` / `8`
- direct self-pair rows: `0`
- same-source pair rows: `0`
- heldout rows used: `0`

## Training

- init checkpoint: `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- output checkpoint: `checkpoints/anima_qwenvl_ip_adapter_c084_balanced_crop_pairs_b128_0128_20260613.safetensors`
- steps: `128`
- rows loaded: `80`
- resolution: `256`
- contrastive weight: `0.35`
- retrieval weight: `0.2`
- calibrator bottleneck dim: `128`
- train calibrator only: `True`
- instruction: c061 `species_face` instruction

## Result

- first loss: `0.10930567234754562`
- final loss: `0.20897169411182404`
- mean loss: `0.20867666404228657`
- mean base loss: `0.15149623539764434`
- mean contrastive loss: `0.05000535165891051`
- mean retrieval loss: `0.19839277071878314`
- finite loss: `True`
- trainable parameters: `528384`
- frozen base parameters: `4947838963`
- checkpoint loadable: `True`
- PE checkpoint rejected: `True`

## Decision

결정: `qwenvl_c084_balanced_crop_pair_training_passed_generation_gate_pending`

학습 자체는 통과했다. c084의 품질 판단은 아직 아니다. 다음 단계는 isolated ComfyUI API에서 `no_ip`, current best runtime `blend_species_face`, 신규 `c084_balanced_crop_pair_w14`를 clean32+heldout8 및 c083 crop-pair focus subset에서 비교하는 generation gate다.
