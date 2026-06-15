# QwenVL c086 Generated Hard-Negative Generation Gate

작성일: 2026-06-13

## 목적

c086은 c085 ComfyUI 실패 생성물을 명시적 negative로 넣어 학습한 checkpoint다. 이 gate의 목적은 c086이 기존 최고 runtime preset `blend_species_face`와 c085를 실제 ComfyUI 생성 결과에서 넘는지 확인하는 것이다.

## 학습 요약

- manifest: `training/manifests/c086_qwenvl_generated_hard_negative_20260613.jsonl`
- manifest summary: `training/manifests/c086_qwenvl_generated_hard_negative_20260613.summary.json`
- training report: `eval/qwenvl_c086_generated_hard_negative_training_20260613/report.md`
- init checkpoint: `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- output checkpoint: `checkpoints/anima_qwenvl_ip_adapter_c086_hard_negative_b128_0096_20260613.safetensors`
- rows loaded: `42`
- explicit negative rows: `42`
- steps: `96`
- trainable parameters: `308176540`
- final loss: `0.1046228409`
- finite loss: `true`
- heldout rows used: `0`

## Runtime

- API: isolated ComfyUI `http://127.0.0.1:8116`
- custom node: repo-local `anima-ipadapter-reference-control`
- extra model paths: `tools/comfyui_extra_model_paths.yaml`
- object info: `eval/qwenvl_c086_generated_hard_negative_gate_20260613/object_info_qwenvl_loader.json`
- loader check: c085/c086 checkpoint 모두 `AnimaQwenVLIPAdapterLoader` 선택지에 노출됨
- cleanup: 생성 및 object_info 확인 후 ComfyUI server를 종료했고 port `8116`은 닫혔다.

## 비교 대상

| variant | 구성 |
|---|---|
| `no_ip` | IP-Adapter 미적용 |
| `blend_species_face` | previous retrieval `1.4` + c055 mixed `0.4` |
| `c085_anchored_full_adapter_w14` | c085 anchored full-adapter checkpoint `1.4` |
| `c086_hard_negative_w14` | c086 generated hard-negative checkpoint `1.4` |

## 산출물

- gate summary: `eval/qwenvl_c086_generated_hard_negative_gate_20260613/summary.json`
- crop focus summary: `eval/qwenvl_c086_generated_hard_negative_gate_20260613/crop_pair_summary.json`
- train sheet: `eval/qwenvl_c086_generated_hard_negative_gate_20260613/contact_sheet_train.jpg`
- heldout sheet: `eval/qwenvl_c086_generated_hard_negative_gate_20260613/contact_sheet_heldout.jpg`
- crop focus sheet: `eval/qwenvl_c086_generated_hard_negative_gate_20260613/contact_sheet_crop_pair_focus.jpg`
- PE metric: `eval/qwenvl_c086_generated_hard_negative_gate_20260613/pe_similarity_metrics.json`
- QwenVL metric: `eval/qwenvl_c086_generated_hard_negative_gate_20260613/qwenvl_similarity_metrics.json`
- crop PE metric: `eval/qwenvl_c086_generated_hard_negative_gate_20260613/crop_pair_pe_similarity_metrics.json`
- crop QwenVL metric: `eval/qwenvl_c086_generated_hard_negative_gate_20260613/crop_pair_qwenvl_similarity_metrics.json`
- visual audit: `eval/qwenvl_c086_generated_hard_negative_gate_20260613/visual_audit.md`
- generated PNG: `200`
- clean32+heldout8 results: `160`
- crop-pair focus results: `40`
- blank image: `0`
- min pixel std: `35.8830680847`

## Metric Summary

### clean32 + heldout8

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 40 | `0.0608932152` | `0.825` |
| PE | `c085_anchored_full_adapter_w14` | 40 | `0.0308974788` | `0.725` |
| PE | `c086_hard_negative_w14` | 40 | `0.0595780790` | `0.750` |
| QwenVL | `blend_species_face` | 40 | `0.0421902567` | `0.800` |
| QwenVL | `c085_anchored_full_adapter_w14` | 40 | `0.0306017444` | `0.775` |
| QwenVL | `c086_hard_negative_w14` | 40 | `0.0388126254` | `0.900` |

### heldout8

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 8 | `0.0535339788` | `0.875` |
| PE | `c085_anchored_full_adapter_w14` | 8 | `0.0370979905` | `0.750` |
| PE | `c086_hard_negative_w14` | 8 | `0.0648405477` | `0.750` |
| QwenVL | `blend_species_face` | 8 | `0.0264708772` | `0.750` |
| QwenVL | `c085_anchored_full_adapter_w14` | 8 | `0.0192572623` | `0.750` |
| QwenVL | `c086_hard_negative_w14` | 8 | `0.0325848311` | `0.750` |

### crop-pair focus

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 10 | `-0.0916786253` | `0.400` |
| PE | `c085_anchored_full_adapter_w14` | 10 | `-0.0143372715` | `0.500` |
| PE | `c086_hard_negative_w14` | 10 | `-0.0233308047` | `0.400` |
| QwenVL | `blend_species_face` | 10 | `0.0194013953` | `0.900` |
| QwenVL | `c085_anchored_full_adapter_w14` | 10 | `0.0466495097` | `0.900` |
| QwenVL | `c086_hard_negative_w14` | 10 | `0.0361481428` | `0.800` |

## 판단

결정: `not_promoted_c086_hard_negative_partial_improvement_not_quality_pass`

c086은 학습/로드/생성 안정성 측면에서는 통과했다. 특히 heldout8 평균에서는 PE와 QwenVL 모두 c085와 `blend_species_face`를 넘었다. `heldout07`처럼 이전에 사람형 악역 template으로 무너지던 케이스에서도 초록 비인간 cue를 더 직접적으로 끌어온다.

하지만 전체 clean32+heldout8 평균에서는 `blend_species_face`보다 낮고, crop-pair focus에서는 c085보다 약해졌다. 사람이 보는 결과도 고유 체형, 작은 mascot-like silhouette, side-profile, headwear가 보존되지 않고 강한 색/종족 cue만 붙는 경향이 남아 있다.

따라서 c086 checkpoint는 high-quality reference-control checkpoint로 승격하지 않는다. 다음 루프는 generated hard-negative를 계속 반복하기보다, 더 믿을 수 있는 target-positive pair를 확보하거나 encoder-side feature adaptation으로 reference embedding 자체를 개선하는 방향으로 간다.
