# c070 QwenVL / Caption Search Direct-Green Acquisition Plan

## 목표

c069는 local color dataset 전체를 pixel-green/red 기준으로 재스캔했지만 direct-green/non-human target positive를 0개만 찾았다. c070의 목표는 학습을 시작하기 전에 semantic/caption 신호로 positive를 더 찾을 수 있는지 확인하는 것이다.

이 루프에서 최소 통과 조건은 c069와 같다. direct-green/non-human target positive가 4개 이상이면 encoder-side supervised training으로 넘어갈 수 있고, 4개 미만이면 외부 데이터 또는 수동 라벨링이 필요하다.

## 입력과 경계

- 전체 color manifest: `training/manifests/local_color_self_reconstruct_20260611.jsonl`
- heldout manifest: `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- c069 reviewed labels: `eval/c069_direct_green_captioning_acquisition_20260612/reviewed_candidate_labels.jsonl`
- 이미지 루트: `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`

heldout 8장은 후보 스캔에서도 제외한다. c069에서 이미 리뷰한 후보는 visual fallback bucket에서 제외해 같은 false positive를 반복하지 않는다.

## 방법

기존 조사에서 QwenVL image-text scoring 도구는 재사용 가능하다고 확인했다. 대표 경로는 `tools/build_reference_prompt_manifest.py`의 `Qwen3VLReferenceTextScorer`, `tools/c067_attribute_teacher_core.py`의 attribute scorer다. 다만 c067/c068에서 이미 QwenVL attribute top-k를 리뷰했고 direct-green positive는 0개였다.

c070은 그 다음 가능한 경로인 sidecar caption search를 먼저 검사한다. color dataset의 `.txt` sidecar는 1571개가 있으나 실제 내용은 대부분 `mrcolor_panel_style, full color manga panel, ...` 형태의 템플릿 태그다. `green`, `monster`, `creature`, `demon`, `skin`, `red eye` 같은 semantic keyword hit는 0개다.

그래서 `tools/c070_qwenvl_caption_search.py`는 두 단계를 기록한다.

1. sidecar caption에서 direct-green/non-human keyword hit를 찾는다.
2. caption hit가 없으면 c069 후보를 제외한 visual fallback bucket을 만들어 review sheet로 남긴다.

fallback bucket은 `semantic_target_fallback`, `red_green_proxy`, `background_green_guard`다.

## 결과

c070 실제 결과:

- scanned images: 1563
- heldout rows used: 0
- missing paths: 0
- c069 reviewed ids: 33
- caption signal source: `sidecar_template_no_hits_fallback_visual_heuristics`
- caption keyword hit images: 0
- candidate rows: 36
- reviewed rows: 36
- direct-green target positive: 0
- useful non-human proxy: 12
- false positive human: 12
- false positive background/object: 12
- decision: `external_manual_data_required`

즉, 현재 local color dataset의 sidecar caption은 semantic acquisition에 사용할 수 없다. visual fallback은 proxy와 guard를 더 만들 수 있지만, 확정 positive를 만들지 못한다.

## 다음 결정

c070 이후에도 checkpoint 학습으로 넘어가지 않는다. 다음 단계는 다음 중 하나여야 한다.

- 외부/추가 데이터셋에서 green/non-human character face 후보를 수집
- 사용자가 직접 4개 이상의 direct-green/non-human target positive를 라벨링
- QwenVL embedding scorer가 아니라 실제 caption generator/VLM 질의 모델로 이미지별 semantic annotation을 생성한 뒤 재검색

positive 4개 이상이 확보되기 전까지 encoder-side direct-green supervised training은 보류한다.
