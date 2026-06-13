# QwenVL c085 Anchored Full-Adapter Training Summary

작성일: 2026-06-13

## 목적

c085 anchored full-adapter 학습 stdout에서 최종 training summary JSON을 추출하고, 다음 ComfyUI generation gate 진입 가능 여부를 정리했다.

## Training Summary

- init checkpoint: `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- output checkpoint: `checkpoints/anima_qwenvl_ip_adapter_c085_anchored_full_adapter_b128_0096_20260613.safetensors`
- steps: `96`
- rows_loaded: `160`
- heldout rows: `0` (`training/manifests/c085_anchored_full_adapter_20260613.summary.json`)
- final_loss: `0.12521426379680634`
- finite_loss: `true`
- trainable_parameters: `308176540`
- calibrator_bottleneck_dim: `128`
- train_calibrator_only: `false`

## 판단

결정: `ready_for_c085_comfyui_generation_gate`

c085 학습은 finite loss로 종료했고 output checkpoint는 loadable로 기록되었다. 이번 manifest에는 heldout rows가 없으므로 이 산출물은 학습 완료/로드 가능성 확인용이며, 아직 품질 통과 판정은 아니다.

다음 단계는 c085 checkpoint를 대상으로 ComfyUI generation gate를 실행해 실제 생성 이미지와 metric/visual audit으로 품질을 판단하는 것이다.
