# QwenVL c075 Tag-Positive Generation Gate

작성일: 2026-06-13

## 목적

c075 checkpoint가 실제 ComfyUI native QwenVL IP-Adapter 경로에서 작동하는지, 그리고 current best `blend_species_face`보다 direct-green/non-human reference-control을 개선하는지 확인했다.

## Runtime

- API: isolated ComfyUI `http://127.0.0.1:8116`
- custom node: repo-local `anima-ipadapter-reference-control`
- extra model paths: `tools/comfyui_extra_model_paths.yaml`
- dependency fix: ComfyUI 시작 시 `.tmp/comfy_py312_deps`와 `.tmp/comfy_qwenvl_single/python_deps`를 `PYTHONPATH`에 추가했다.
- object_info: `eval/qwenvl_c075_tag_positive_gate_20260612/object_info.json`
- cleanup: 생성 후 ComfyUI server를 종료했고 port `8116`은 닫혔다.

## 비교 대상

| variant | 구성 |
|---|---|
| `no_ip` | IP-Adapter 미적용 |
| `blend_species_face` | previous retrieval `1.4` + c055 mixed `0.4` |
| `c075_tag_positive_w14` | c075 tag-positive calibrator checkpoint `1.4` |

## 산출물

- summary: `eval/qwenvl_c075_tag_positive_gate_20260612/summary.json`
- direct-green summary: `eval/qwenvl_c075_tag_positive_gate_20260612/direct_green_summary.json`
- train sheet: `eval/qwenvl_c075_tag_positive_gate_20260612/contact_sheet_train.jpg`
- heldout sheet: `eval/qwenvl_c075_tag_positive_gate_20260612/contact_sheet_heldout.jpg`
- direct-green sheet: `eval/qwenvl_c075_tag_positive_gate_20260612/contact_sheet_direct_green.jpg`
- visual audit: `eval/qwenvl_c075_tag_positive_gate_20260612/visual_audit.md`
- generated PNG: `150`
- blank image: `0`
- min pixel std: `35.8830680847168`

## Metric Summary

### clean32 + heldout8

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 40 | `0.0608932152` | `0.825` |
| PE | `c075_tag_positive_w14` | 40 | `0.0262199253` | `0.650` |
| QwenVL | `blend_species_face` | 40 | `0.0421902567` | `0.800` |
| QwenVL | `c075_tag_positive_w14` | 40 | `0.0349742755` | `0.750` |

### direct-green focus

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 10 | `0.0379917264` | `0.700` |
| PE | `c075_tag_positive_w14` | 10 | `-0.0206880599` | `0.400` |
| QwenVL | `blend_species_face` | 10 | `-0.0121086836` | `0.200` |
| QwenVL | `c075_tag_positive_w14` | 10 | `-0.0143850207` | `0.400` |

## 판단

결정: `not_promoted_c075_tag_positive_calibrator_weaker_than_blend_species_face`

c075는 ComfyUI에서 정상 로드되고 이미지를 바꾸지만, 품질 gate는 통과하지 못했다. clean32+heldout8에서는 `blend_species_face`보다 낮고, c075의 핵심 목표였던 direct-green focus에서도 PE/QwenVL 평균 uplift가 음수다.

다음 루프는 같은 calibrator-only target-positive 반복이 아니라, 더 강한 encoder-side/reference feature objective 또는 더 좋은 paired direct-green 데이터로 가야 한다.
