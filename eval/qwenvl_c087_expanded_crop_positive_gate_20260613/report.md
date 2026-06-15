# QwenVL c087 Expanded Crop-Positive Generation Gate

작성일: 2026-06-13

## 목적

c087은 c086의 generated hard-negative 방식이 crop-focus에서 약해진 문제를 반대로 접근했다. c083에서 승인된 crop-pair target-positive 데이터를 c084/c085보다 훨씬 많이 사용하면, frog/yokai/chibi 같은 비인간 reference의 shape identity를 더 잘 붙잡을 수 있는지 검증했다.

## 학습 요약

- expanded crop manifest: `training/manifests/c087_expanded_crop_pairs_20260613.jsonl`
- anchored manifest: `training/manifests/c087_expanded_anchored_full_adapter_20260613.jsonl`
- training report: `eval/qwenvl_c087_expanded_crop_positive_training_20260613/report.md`
- init checkpoint: `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- output checkpoint: `checkpoints/anima_qwenvl_ip_adapter_c087_expanded_crop_positive_b128_0128_20260613.safetensors`
- expanded crop rows: `224`
- total anchored rows: `304`
- heldout rows used: `0`
- steps: `128`
- trainable parameters: `308176540`
- final loss: `0.1393356025`
- finite loss: `true`
- checkpoint loadable: `true`
- PE checkpoint rejected: `true`

## Runtime

- API: isolated ComfyUI `http://127.0.0.1:8116`
- custom node: repo-local `anima-ipadapter-reference-control`
- extra model paths: `tools/comfyui_extra_model_paths.yaml`
- object info: `eval/qwenvl_c087_expanded_crop_positive_gate_20260613/object_info.json`
- loader check: c085/c086/c087 checkpoint 모두 `AnimaQwenVLIPAdapterLoader` 선택지에 노출됨
- cleanup: 생성 및 object_info 확인 후 ComfyUI server를 종료했고 port `8116`은 닫혔다.

## 비교 대상

| variant | 구성 |
|---|---|
| `no_ip` | IP-Adapter 미적용 |
| `blend_species_face` | previous retrieval `1.4` + c055 mixed `0.4` |
| `c085_anchored_full_adapter_w14` | c085 checkpoint `1.4` |
| `c086_hard_negative_w14` | c086 checkpoint `1.4` |
| `c087_expanded_crop_positive_w14` | c087 checkpoint `1.4` |

## 산출물

- gate summary: `eval/qwenvl_c087_expanded_crop_positive_gate_20260613/summary.json`
- crop focus summary: `eval/qwenvl_c087_expanded_crop_positive_gate_20260613/crop_pair_summary.json`
- metric rollup: `eval/qwenvl_c087_expanded_crop_positive_gate_20260613/metric_rollup.json`
- train sheet: `eval/qwenvl_c087_expanded_crop_positive_gate_20260613/contact_sheet_train.jpg`
- heldout sheet: `eval/qwenvl_c087_expanded_crop_positive_gate_20260613/contact_sheet_heldout.jpg`
- crop focus sheet: `eval/qwenvl_c087_expanded_crop_positive_gate_20260613/contact_sheet_crop_pair_focus.jpg`
- PE metric: `eval/qwenvl_c087_expanded_crop_positive_gate_20260613/pe_similarity_metrics.json`
- QwenVL metric: `eval/qwenvl_c087_expanded_crop_positive_gate_20260613/qwenvl_similarity_metrics.json`
- crop PE metric: `eval/qwenvl_c087_expanded_crop_positive_gate_20260613/crop_pair_pe_similarity_metrics.json`
- crop QwenVL metric: `eval/qwenvl_c087_expanded_crop_positive_gate_20260613/crop_pair_qwenvl_similarity_metrics.json`
- visual audit: `eval/qwenvl_c087_expanded_crop_positive_gate_20260613/visual_audit.md`
- pixel audit: `eval/qwenvl_c087_expanded_crop_positive_gate_20260613/pixel_audit.json`
- generated PNG: `250`
- clean32+heldout8 results: `200`
- crop-pair focus results: `50`
- blank image: `0`
- min pixel std: `35.8830680847`

## Metric Summary

### clean32 + heldout8

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 40 | `0.0608932152` | `0.825` |
| PE | `c085_anchored_full_adapter_w14` | 40 | `0.0308974788` | `0.725` |
| PE | `c086_hard_negative_w14` | 40 | `0.0595780790` | `0.750` |
| PE | `c087_expanded_crop_positive_w14` | 40 | `0.0311193079` | `0.725` |
| QwenVL | `blend_species_face` | 40 | `0.0421902567` | `0.800` |
| QwenVL | `c085_anchored_full_adapter_w14` | 40 | `0.0306017444` | `0.775` |
| QwenVL | `c086_hard_negative_w14` | 40 | `0.0388126254` | `0.900` |
| QwenVL | `c087_expanded_crop_positive_w14` | 40 | `0.0310260832` | `0.800` |

### heldout8

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 8 | `0.0535339788` | `0.875` |
| PE | `c085_anchored_full_adapter_w14` | 8 | `0.0370979905` | `0.750` |
| PE | `c086_hard_negative_w14` | 8 | `0.0648405477` | `0.750` |
| PE | `c087_expanded_crop_positive_w14` | 8 | `0.0602737069` | `1.000` |
| QwenVL | `blend_species_face` | 8 | `0.0264708772` | `0.750` |
| QwenVL | `c085_anchored_full_adapter_w14` | 8 | `0.0192572623` | `0.750` |
| QwenVL | `c086_hard_negative_w14` | 8 | `0.0325848311` | `0.750` |
| QwenVL | `c087_expanded_crop_positive_w14` | 8 | `0.0139362663` | `0.750` |

### crop-pair focus

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 10 | `-0.0549685217` | `0.400` |
| PE | `c085_anchored_full_adapter_w14` | 10 | `0.0815944552` | `0.700` |
| PE | `c086_hard_negative_w14` | 10 | `0.0940596938` | `0.700` |
| PE | `c087_expanded_crop_positive_w14` | 10 | `0.0536489964` | `0.700` |
| QwenVL | `blend_species_face` | 10 | `-0.0043623924` | `0.500` |
| QwenVL | `c085_anchored_full_adapter_w14` | 10 | `0.0222077310` | `0.800` |
| QwenVL | `c086_hard_negative_w14` | 10 | `0.0023720264` | `0.500` |
| QwenVL | `c087_expanded_crop_positive_w14` | 10 | `0.0043545485` | `0.600` |

## 판단

결정: `not_promoted_c087_expanded_crop_positive_not_quality_pass`

c087은 학습, 로드, ComfyUI 생성 안정성은 통과했다. heldout8 PE 기준으로는 `blend_species_face`보다 높고 improved rate도 `1.0`이어서 일부 긍정 신호가 있다.

하지만 clean32+heldout8 전체 평균에서 c087은 PE/QwenVL 모두 blend와 c086보다 낮다. crop-focus에서도 c087은 c085/c086보다 낮고, 사람이 보는 결과에서도 frog mascot/chibi reference의 둥근 몸, 짧은 비율, 모자, side-profile silhouette을 adult green humanoid template로 바꾼다.

따라서 c087 checkpoint는 high-quality reference-control checkpoint로 승격하지 않는다. expanded target-positive crop supervision만 늘리는 접근은 한계가 분명하다. 다음 루프는 같은 full-adapter continuation 반복이 아니라 encoder-side checkpoint adaptation 또는 shape/silhouette을 직접 보상하는 feature objective로 이동한다.
