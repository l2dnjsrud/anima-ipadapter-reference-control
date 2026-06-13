# QwenVL c084 Balanced Crop-Pair Generation Gate

작성일: 2026-06-13

## 목적

c084 checkpoint가 c083 sheet crop identity pair를 balanced manifest로 학습한 뒤, current best runtime preset `blend_species_face`보다 reference-control을 개선하는지 확인했다. c084는 c080 collapse를 이어받지 않기 위해 `single_character_retrieval_0128`에서 다시 시작했고, calibrator-only로 128 step 학습했다.

## Runtime

- API: isolated ComfyUI `http://127.0.0.1:8116`
- custom node: repo-local `anima-ipadapter-reference-control`
- extra model paths: `tools/comfyui_extra_model_paths.yaml`
- dependency shim: ComfyUI02 venv가 root 소유이고 `sentence_transformers`가 없어 `.tmp/comfy_py312_site`에 `sentence-transformers==5.5.1`을 target install한 뒤 PYTHONPATH에 추가했다.
- object info: `eval/qwenvl_c084_balanced_crop_pair_gate_20260613/object_info_qwenvl_loader.json`
- cleanup: 생성 후 ComfyUI server를 종료했고 port `8116`은 닫혔다.

## 비교 대상

| variant | 구성 |
|---|---|
| `no_ip` | IP-Adapter 미적용 |
| `blend_species_face` | previous retrieval `1.4` + c055 mixed `0.4` |
| `c084_balanced_crop_pair_w14` | c084 balanced crop-pair calibrator checkpoint `1.4` |

## 산출물

- training summary: `eval/qwenvl_c084_balanced_crop_pair_training_20260613/summary.json`
- gate summary: `eval/qwenvl_c084_balanced_crop_pair_gate_20260613/summary.json`
- crop focus summary: `eval/qwenvl_c084_balanced_crop_pair_gate_20260613/crop_pair_summary.json`
- train sheet: `eval/qwenvl_c084_balanced_crop_pair_gate_20260613/contact_sheet_train.jpg`
- heldout sheet: `eval/qwenvl_c084_balanced_crop_pair_gate_20260613/contact_sheet_heldout.jpg`
- crop focus sheet: `eval/qwenvl_c084_balanced_crop_pair_gate_20260613/contact_sheet_crop_pair_focus.jpg`
- PE metric: `eval/qwenvl_c084_balanced_crop_pair_gate_20260613/pe_similarity_metrics.json`
- QwenVL metric: `eval/qwenvl_c084_balanced_crop_pair_gate_20260613/qwenvl_similarity_metrics.json`
- crop PE metric: `eval/qwenvl_c084_balanced_crop_pair_gate_20260613/crop_pair_pe_similarity_metrics.json`
- crop QwenVL metric: `eval/qwenvl_c084_balanced_crop_pair_gate_20260613/crop_pair_qwenvl_similarity_metrics.json`
- visual audit: `eval/qwenvl_c084_balanced_crop_pair_gate_20260613/visual_audit.md`
- generated PNG: `150`
- clean32+heldout8 results: `120`
- crop-pair focus results: `30`
- blank image: `0`
- min pixel std: `35.8830680847`

## Metric Summary

### clean32 + heldout8

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 40 | `0.0608932152` | `0.825` |
| PE | `c084_balanced_crop_pair_w14` | 40 | `0.0272349045` | `0.650` |
| QwenVL | `blend_species_face` | 40 | `0.0421902567` | `0.800` |
| QwenVL | `c084_balanced_crop_pair_w14` | 40 | `0.0336157218` | `0.725` |

### crop-pair focus

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 10 | `-0.0046628684` | `0.600` |
| PE | `c084_balanced_crop_pair_w14` | 10 | `-0.0372054994` | `0.400` |
| QwenVL | `blend_species_face` | 10 | `0.0186477482` | `0.600` |
| QwenVL | `c084_balanced_crop_pair_w14` | 10 | `0.0302898765` | `0.700` |

## 판단

결정: `not_promoted_c084_balanced_crop_pair_weaker_than_blend`

c084는 ComfyUI에서 정상 로드되고 이미지를 생성한다. object_info에서도 `AnimaQwenVLIPAdapterLoader`가 c084 checkpoint를 선택지로 노출했다. 하지만 clean32+heldout8에서는 PE와 QwenVL 모두 기존 `blend_species_face`보다 낮다.

crop-pair focus에서는 c084가 QwenVL mean uplift 기준으로 blend보다 조금 높지만, PE 기준은 더 나쁘다. 시각적으로도 frog/yokai/chibi crop reference가 성인형 green humanoid villain으로 수렴하고, headwear, 작은 체형, costume palette, side-profile identity를 충분히 보존하지 못한다.

따라서 c084 checkpoint는 high-quality reference-control checkpoint로 승격하지 않는다. 다음 루프는 같은 crop-pair calibrator-only 반복을 중단하고, adapter-side trainable 범위 확대 또는 stronger encoder/objective 학습으로 이동해야 한다.
