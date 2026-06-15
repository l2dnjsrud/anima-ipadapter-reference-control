# c069 Direct Green / Non-Human Captioning Acquisition Plan

## 목표

c068에서 c067 top-k 후보를 직접 리뷰했지만 `direct_green_target_positive_count=0`으로 끝났다. c069의 목표는 checkpoint를 새로 학습하지 않고, local color dataset 전체를 heldout 누수 없이 다시 훑어서 direct-green/non-human character positive를 확보할 수 있는지 확인하는 것이다.

최소 통과 조건은 `target_positive` 4개 이상이다. 이 조건을 넘지 못하면 encoder-side supervised training으로 넘어가지 않는다.

## 입력 데이터

- 전체 후보 소스: `training/manifests/local_color_self_reconstruct_20260611.jsonl`
- heldout 제외 소스: `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- 실제 이미지 루트: `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`
- 이전 top-k 기준: `eval/c067_attribute_teacher_reranker_seed_20260612/attribute_topk.json`

학습 데이터 생성이 아니므로 heldout 이미지는 후보 스캔에서도 제외한다.

## 방법

`tools/c069_direct_green_acquisition.py`가 전체 color manifest에서 heldout 8장을 제외한 1563장을 스캔한다. 각 이미지를 128x128로 축소해 green/red pixel ratio를 계산하고 다음 4개 bucket에서 상위 12개씩 후보를 만든다.

- `target_score`: 중앙부 green, strong-green, red-eye proxy를 올리고 border green을 낮춘 후보
- `background_score`: green background/object false positive guard
- `strong_green`: 강한 green pixel 후보
- `red_green_mix`: red-eye와 green이 같이 나타나는 후보

후보 manifest와 review sheet를 만든 뒤, direct-green/non-human target인지 아닌지 라벨을 분리한다.

## 검증 기준

- `candidate_manifest.jsonl`이 48 rows를 가져야 한다.
- `summary.json`에서 `heldout_rows_used=0`, `missing_paths=0`, `scanned_beyond_c067_topk=true`여야 한다.
- `reviewed_candidate_labels.jsonl`과 `annotated_review_sheet.jpg`가 생성되어야 한다.
- `direct_green_target_positive_count >= 4`일 때만 다음 encoder-side 학습으로 넘어간다.

## 결과

c069 결과는 다음과 같다.

- 스캔 이미지: 1563
- 후보 rows: 48
- 고유 후보 이미지: 33
- heldout rows used: 0
- missing paths: 0
- direct-green target positive: 0
- useful non-human proxy: 2
- green/background/object false positive: 46
- decision: `new_dataset_captioning_required`

즉, local color dataset 전체에서 green/red pixel 기반으로 후보를 확장해도 direct-green/non-human target positive는 확보되지 않았다. 일부 non-human proxy는 보이지만 확정 target positive가 아니므로 바로 encoder-side supervised training에 쓰면 background/object green을 잘못 학습할 가능성이 높다.

## 다음 결정

c069 이후에는 checkpoint 학습을 진행하지 않는다. 다음 루프는 새 데이터 획득 또는 캡션 기반 재수집이 먼저다.

가능한 다음 방향:

- 수동 라벨링으로 direct-green/non-human target positive 4개 이상 확보
- 외부/추가 color dataset에서 non-human green face/skin 후보 수집
- QwenVL captioning을 사용해 "green skin", "monster face", "non-human profile" 같은 텍스트 단서로 후보를 재검색
- direct-green positive가 4개 이상 모이면 c070에서 encoder-side supervised feature adaptation 또는 contrastive seed 학습으로 진행
