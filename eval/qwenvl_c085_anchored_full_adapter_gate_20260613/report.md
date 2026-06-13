# QwenVL c085 Anchored Full-Adapter Generation Gate

작성일: 2026-06-13

## 목적

c084 balanced crop-pair 실험은 crop focus에서 약한 신호를 보였지만, calibrator-only 범위가 너무 좁아 reference identity가 성인형 green humanoid template로 수렴했다. c085는 같은 crop-pair 신호를 유지하면서 clean32/c052/failure-anchor row를 섞고, full adapter 범위를 trainable로 열어 실제 ComfyUI generation gate에서 기존 최고 런타임 preset `blend_species_face`를 넘는지 확인했다.

## 학습 요약

- manifest: `training/manifests/c085_anchored_full_adapter_20260613.jsonl`
- manifest summary: `training/manifests/c085_anchored_full_adapter_20260613.summary.json`
- training report: `eval/qwenvl_c085_anchored_full_adapter_training_20260613/report.md`
- init checkpoint: `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- output checkpoint: `checkpoints/anima_qwenvl_ip_adapter_c085_anchored_full_adapter_b128_0096_20260613.safetensors`
- rows loaded: `160`
- steps: `96`
- trainable parameters: `308176540`
- train mode: full adapter + calibrator, `train_calibrator_only=false`
- final loss: `0.1252142638`
- finite loss: `true`
- heldout rows used: `0`

## Runtime

- API: isolated ComfyUI `http://127.0.0.1:8116`
- custom node: repo-local `anima-ipadapter-reference-control`
- extra model paths: `tools/comfyui_extra_model_paths.yaml`
- object info: `eval/qwenvl_c085_anchored_full_adapter_gate_20260613/object_info_qwenvl_loader.json`
- cleanup: 생성 후 ComfyUI server를 종료했고 port `8116`은 닫혔다.

## 비교 대상

| variant | 구성 |
|---|---|
| `no_ip` | IP-Adapter 미적용 |
| `blend_species_face` | previous retrieval `1.4` + c055 mixed `0.4` |
| `c084_balanced_crop_pair_w14` | c084 balanced crop-pair calibrator checkpoint `1.4` |
| `c085_anchored_full_adapter_w14` | c085 anchored full-adapter checkpoint `1.4` |

## 산출물

- gate summary: `eval/qwenvl_c085_anchored_full_adapter_gate_20260613/summary.json`
- crop focus summary: `eval/qwenvl_c085_anchored_full_adapter_gate_20260613/crop_pair_summary.json`
- train sheet: `eval/qwenvl_c085_anchored_full_adapter_gate_20260613/contact_sheet_train.jpg`
- heldout sheet: `eval/qwenvl_c085_anchored_full_adapter_gate_20260613/contact_sheet_heldout.jpg`
- crop focus sheet: `eval/qwenvl_c085_anchored_full_adapter_gate_20260613/contact_sheet_crop_pair_focus.jpg`
- PE metric: `eval/qwenvl_c085_anchored_full_adapter_gate_20260613/pe_similarity_metrics.json`
- QwenVL metric: `eval/qwenvl_c085_anchored_full_adapter_gate_20260613/qwenvl_similarity_metrics.json`
- crop PE metric: `eval/qwenvl_c085_anchored_full_adapter_gate_20260613/crop_pair_pe_similarity_metrics.json`
- crop QwenVL metric: `eval/qwenvl_c085_anchored_full_adapter_gate_20260613/crop_pair_qwenvl_similarity_metrics.json`
- visual audit: `eval/qwenvl_c085_anchored_full_adapter_gate_20260613/visual_audit.md`
- generated PNG: `200`
- clean32+heldout8 results: `160`
- crop-pair focus results: `40`
- blank image: `0`
- min pixel std: `35.3214362316`

## Metric Summary

### clean32 + heldout8

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 40 | `0.0608932152` | `0.825` |
| PE | `c084_balanced_crop_pair_w14` | 40 | `0.0272349045` | `0.650` |
| PE | `c085_anchored_full_adapter_w14` | 40 | `0.0308974788` | `0.725` |
| QwenVL | `blend_species_face` | 40 | `0.0421902567` | `0.800` |
| QwenVL | `c084_balanced_crop_pair_w14` | 40 | `0.0336157218` | `0.725` |
| QwenVL | `c085_anchored_full_adapter_w14` | 40 | `0.0306017444` | `0.775` |

### heldout8

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 8 | `0.0535339788` | `0.875` |
| PE | `c084_balanced_crop_pair_w14` | 8 | `-0.0063817352` | `0.625` |
| PE | `c085_anchored_full_adapter_w14` | 8 | `0.0370979905` | `0.750` |
| QwenVL | `blend_species_face` | 8 | `0.0264708772` | `0.750` |
| QwenVL | `c084_balanced_crop_pair_w14` | 8 | `0.0235493183` | `0.625` |
| QwenVL | `c085_anchored_full_adapter_w14` | 8 | `0.0192572623` | `0.750` |

### crop-pair focus

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 10 | `0.1395019785` | `0.600` |
| PE | `c084_balanced_crop_pair_w14` | 10 | `0.1503805324` | `0.700` |
| PE | `c085_anchored_full_adapter_w14` | 10 | `0.1288137928` | `0.600` |
| QwenVL | `blend_species_face` | 10 | `0.0649063945` | `1.000` |
| QwenVL | `c084_balanced_crop_pair_w14` | 10 | `0.0639641464` | `0.900` |
| QwenVL | `c085_anchored_full_adapter_w14` | 10 | `0.0652211845` | `0.900` |

## 판단

결정: `not_promoted_c085_full_adapter_weaker_than_blend`

c085는 ComfyUI에서 정상 로드되고 200개 이미지를 생성했다. blank 이미지는 없고, `AnimaQwenVLIPAdapterLoader`의 모델 선택지에도 c085 checkpoint가 정상 노출되었다.

하지만 품질 기준에서는 기존 `blend_species_face`를 넘지 못했다. clean32+heldout8 전체에서 PE/QwenVL 평균 uplift가 모두 blend보다 낮고, heldout8에서도 QwenVL 기준으로 c084보다 낮다. crop-pair focus에서는 QwenVL 평균 uplift가 blend보다 아주 근소하게 높지만, PE 기준과 improved rate가 약하고 사람이 보는 결과도 고유 체형/복장/측면 실루엣 보존이 부족하다.

시각적으로 c085는 c084 대비 일부 reference 색/피부 신호를 유지하지만, frog/yokai/chibi reference를 성인형 green humanoid로 수렴시키는 문제를 해결하지 못했다. heldout07 비인간 측면 얼굴도 여전히 사람형 악역 남캐로 무너진다.

따라서 c085 checkpoint는 high-quality reference-control checkpoint로 승격하지 않는다. 다음 루프는 full adapter continuation 반복보다, 더 강한 encoder-side supervision 또는 생성 결과를 직접 벌점으로 다루는 hard-negative/objective 쪽으로 이동해야 한다.
