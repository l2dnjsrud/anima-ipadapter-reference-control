# QwenVL c079 Synthetic-Positive Generation Gate

작성일: 2026-06-13

## 목적

c079 checkpoint가 c078 synthetic direct-green target positives 23장과 c074 real target positives 10장을 반영해, current best `blend_species_face` 및 c075보다 direct-green/non-human reference-control을 개선하는지 확인했다.

## Runtime

- API: isolated ComfyUI `http://127.0.0.1:8116`
- custom node: repo-local `anima-ipadapter-reference-control`
- extra model paths: `tools/comfyui_extra_model_paths.yaml`
- object_info: `eval/qwenvl_c079_synthetic_positive_gate_20260612/object_info.json`
- cleanup: 생성 후 ComfyUI server를 종료했고 port `8116`은 닫혔다.

## 비교 대상

| variant | 구성 |
|---|---|
| `no_ip` | IP-Adapter 미적용 |
| `blend_species_face` | previous retrieval `1.4` + c055 mixed `0.4` |
| `c075_tag_positive_w14` | c075 tag-positive calibrator checkpoint `1.4` |
| `c079_synthetic_positive_w14` | c079 synthetic-positive calibrator checkpoint `1.4` |

## 산출물

- summary: `eval/qwenvl_c079_synthetic_positive_gate_20260612/summary.json`
- direct-green summary: `eval/qwenvl_c079_synthetic_positive_gate_20260612/direct_green_summary.json`
- train sheet: `eval/qwenvl_c079_synthetic_positive_gate_20260612/contact_sheet_train.jpg`
- heldout sheet: `eval/qwenvl_c079_synthetic_positive_gate_20260612/contact_sheet_heldout.jpg`
- direct-green sheet: `eval/qwenvl_c079_synthetic_positive_gate_20260612/contact_sheet_direct_green.jpg`
- visual audit: `eval/qwenvl_c079_synthetic_positive_gate_20260612/visual_audit.md`
- generated PNG: `292`
- clean32+heldout8 results: `160`
- direct-green focus results: `132`
- blank image: `0`
- min pixel std: `35.321`

## Metric Summary

### clean32 + heldout8

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 40 | `0.0608932152` | `0.825` |
| PE | `c075_tag_positive_w14` | 40 | `0.0262199253` | `0.650` |
| PE | `c079_synthetic_positive_w14` | 40 | `0.0329968661` | `0.700` |
| QwenVL | `blend_species_face` | 40 | `0.0421902567` | `0.800` |
| QwenVL | `c075_tag_positive_w14` | 40 | `0.0349742755` | `0.750` |
| QwenVL | `c079_synthetic_positive_w14` | 40 | `0.0338256791` | `0.725` |

### direct-green focus

| encoder | variant | cases | mean uplift | improved rate |
|---|---|---:|---:|---:|
| PE | `blend_species_face` | 33 | `0.3416856171` | `0.758` |
| PE | `c075_tag_positive_w14` | 33 | `0.2800143618` | `0.727` |
| PE | `c079_synthetic_positive_w14` | 33 | `0.2937260229` | `0.758` |
| QwenVL | `blend_species_face` | 33 | `0.0291833697` | `0.727` |
| QwenVL | `c075_tag_positive_w14` | 33 | `0.0386207013` | `0.727` |
| QwenVL | `c079_synthetic_positive_w14` | 33 | `0.0388706634` | `0.758` |

## 판단

결정: `not_promoted_c079_synthetic_positive_calibrator_partial_direct_green_gain`

c079는 ComfyUI에서 정상 로드되고 이미지를 생성한다. direct-green focus에서는 c075보다 QwenVL mean uplift가 아주 조금 높고, PE도 c075보다 높다. 하지만 clean32+heldout8 전체에서는 current best `blend_species_face`보다 낮고, direct-green PE에서도 `blend_species_face`가 여전히 가장 강하다.

시각적으로도 c079는 green/non-human tag를 강화하지만, reference별 identity를 충분히 유지하지 못한다. synthetic target-positive 확장은 속성 방향을 조금 보강했으나, high-quality reference-control checkpoint로 승격할 정도는 아니다.

다음 루프는 단순 synthetic-positive 반복보다 실제 paired direct-green 데이터, synthetic source-target identity pair 구성, 또는 encoder-side/reference feature objective 쪽으로 이동해야 한다.
