# QwenVL c086 Generated Hard-Negative Training

작성일: 2026-06-13

## 목적

c085는 full adapter 학습으로 c084보다 일부 회복됐지만, 기존 runtime preset `blend_species_face`를 넘지 못했다. c086은 c085 ComfyUI gate에서 나온 실패 생성물을 같은 reference의 명시적 negative로 재투입해서, adapter가 "비슷하게 초록색인 결과"가 아니라 reference identity에 더 가까운 결과를 선호하도록 학습시키는 실험이다.

## Manifest

- builder: `tools/c086_generated_hard_negative_manifest.py`
- manifest: `training/manifests/c086_qwenvl_generated_hard_negative_20260613.jsonl`
- summary: `training/manifests/c086_qwenvl_generated_hard_negative_20260613.summary.json`
- training copy: `eval/qwenvl_c086_generated_hard_negative_training_20260613/manifest_stdout.json`
- source gate: `eval/qwenvl_c085_anchored_full_adapter_gate_20260613`
- scratch root: `.tmp/c086_generated_hard_negative_root`
- train negative rows: `32`
- crop negative rows: `10`
- generated negative rows: `42`
- total rows: `42`
- heldout rows used: `0`

## Training

- command surface: `training/qwenvl_contrastive_cli.py`
- init checkpoint: `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- output checkpoint: `checkpoints/anima_qwenvl_ip_adapter_c086_hard_negative_b128_0096_20260613.safetensors`
- steps: `96`
- rows loaded: `42`
- resolution: `256`
- lr: `2e-6`
- contrastive weight: `0.35`
- contrastive margin: `0.05`
- retrieval weight: `0.25`
- retrieval margin: `0.25`
- calibrator bottleneck: `128`
- train mode: full adapter + calibrator, `train_calibrator_only=false`
- explicit negative rows: `42`
- trainable parameters: `308176540`
- frozen base parameters: `4947838963`
- first loss: `0.1366712153`
- final loss: `0.1046228409`
- mean loss: `0.2471513194`
- finite loss: `true`
- checkpoint loadable: `true`
- PE checkpoint rejected: `true`

## 판단

학습 자체는 정상 통과했다. `neg_id` 기반 명시적 hard-negative row가 42개 모두 사용됐고, checkpoint는 native QwenVL IP-Adapter 형식으로 loadable하다.

다만 이 단계는 생성 품질을 증명하지 않는다. c086 checkpoint는 반드시 `eval/qwenvl_c086_generated_hard_negative_gate_20260613`의 ComfyUI API gate와 contact sheet/metric audit으로 판단한다.
