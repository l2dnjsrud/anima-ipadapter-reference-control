# QwenVL c080 Paired Direct-Green Generation Gate

작성일: 2026-06-13

## 목적

c080 checkpoint가 c074 direct-green reference를 `ref_id != tgt_id` paired supervision으로 학습한 뒤, c079 synthetic-positive calibrator와 current best runtime preset `blend_species_face`보다 reference-control을 개선하는지 확인했다.

## Runtime

- API: isolated ComfyUI `http://127.0.0.1:8116`
- custom node: repo-local `anima-ipadapter-reference-control`
- extra model paths: `tools/comfyui_extra_model_paths.yaml`
- object info: `eval/qwenvl_c080_paired_direct_green_gate_20260613/object_info.json`
- cleanup: 생성 후 ComfyUI server를 종료했고 port `8116`은 닫혔다.

초기 c080 runner에서 `c079_synthetic_positive_w14` 라벨이 c080 checkpoint를 가리키는 mapping bug가 있었다. 해당 c079 산출물 50개를 삭제하고 runner를 수정한 뒤, c079 항목만 실제 c079 checkpoint로 재생성했다.

## 비교 대상

| variant | 구성 |
|---|---|
| `no_ip` | IP-Adapter 미적용 |
| `blend_species_face` | previous retrieval `1.4` + c055 mixed `0.4` |
| `c075_tag_positive_w14` | c075 tag-positive calibrator checkpoint `1.4` |
| `c079_synthetic_positive_w14` | c079 synthetic-positive calibrator checkpoint `1.4` |
| `c080_paired_direct_green_w14` | c080 paired direct-green calibrator checkpoint `1.4` |

## 산출물

- summary: `eval/qwenvl_c080_paired_direct_green_gate_20260613/summary.json`
- direct-green summary: `eval/qwenvl_c080_paired_direct_green_gate_20260613/direct_green_summary.json`
- train sheet: `eval/qwenvl_c080_paired_direct_green_gate_20260613/contact_sheet_train.jpg`
- heldout sheet: `eval/qwenvl_c080_paired_direct_green_gate_20260613/contact_sheet_heldout.jpg`
- direct-green sheet: `eval/qwenvl_c080_paired_direct_green_gate_20260613/contact_sheet_direct_green.jpg`
- visual audit: `eval/qwenvl_c080_paired_direct_green_gate_20260613/visual_audit.md`
- generated PNG: `250`
- clean32+heldout8 results: `200`
- direct-green focus results: `50`
- blank image: `0`

## Metric Summary

### clean32 + heldout8

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 40 | `0.0608932152` | `0.825` |
| PE | `c075_tag_positive_w14` | 40 | `0.0262199253` | `0.650` |
| PE | `c079_synthetic_positive_w14` | 40 | `0.0329968661` | `0.700` |
| PE | `c080_paired_direct_green_w14` | 40 | `0.0229417309` | `0.675` |
| QwenVL | `blend_species_face` | 40 | `0.0421902567` | `0.800` |
| QwenVL | `c075_tag_positive_w14` | 40 | `0.0349742755` | `0.750` |
| QwenVL | `c079_synthetic_positive_w14` | 40 | `0.0338256791` | `0.725` |
| QwenVL | `c080_paired_direct_green_w14` | 40 | `0.0341175169` | `0.750` |

### direct-green focus

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 10 | `0.0844765946` | `0.600` |
| PE | `c075_tag_positive_w14` | 10 | `0.0325176731` | `0.500` |
| PE | `c079_synthetic_positive_w14` | 10 | `0.0640649319` | `0.500` |
| PE | `c080_paired_direct_green_w14` | 10 | `0.0482934043` | `0.500` |
| QwenVL | `blend_species_face` | 10 | `-0.0102016628` | `0.500` |
| QwenVL | `c075_tag_positive_w14` | 10 | `-0.0095478654` | `0.600` |
| QwenVL | `c079_synthetic_positive_w14` | 10 | `0.0040470958` | `0.700` |
| QwenVL | `c080_paired_direct_green_w14` | 10 | `-0.0087357759` | `0.700` |

## 판단

결정: `not_promoted_c080_paired_direct_green_weaker_than_c079_and_blend`

c080은 ComfyUI에서 정상 로드되고 이미지를 생성한다. 그러나 clean32+heldout8 PE metric에서 c080은 c079와 c075보다 낮고, current best `blend_species_face`와의 격차도 크다. QwenVL clean metric에서는 c079보다 아주 조금 높지만 `0.00029` 수준이라 품질 개선으로 보기 어렵다.

direct-green focus에서도 c080은 PE 기준으로 c079보다 낮고, QwenVL 기준으로는 no_ip보다 낮다. 시각적으로도 c080은 green/non-human 방향을 내지만 reference별 색채, 장식, 성별/체형, 얼굴 identity를 안정적으로 유지하지 못하고 성인형 green humanoid villain으로 수렴한다.

따라서 c080 checkpoint는 high-quality reference-control checkpoint로 승격하지 않는다. 다음 루프는 c074 pair를 더 반복하는 방식이 아니라, 실제 paired source-target color/reference 데이터 확보 또는 encoder-side/feature objective 강화로 이동해야 한다.
