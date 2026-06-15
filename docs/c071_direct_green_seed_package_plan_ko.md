# c071 Direct-Green Manual / External Seed Package Plan

c070까지의 자동 탐색 결론은 명확하다. local color dataset 안에서 direct-green/non-human character target positive를 자동으로 4개 이상 확보하지 못했다. 따라서 c071은 checkpoint 학습이 아니라, 다음 학습을 안전하게 시작하기 위한 수동/외부 라벨 패키지 구축 루프다.

## 목표

목표는 c068, c069, c070의 proxy/guard 후보를 한곳에 모으고, 사람이 `target_positive`를 명시적으로 확정할 수 있는 annotation package를 만드는 것이다.

학습으로 넘어가는 최소 조건은 변하지 않는다.

- `target_positive`가 unique image 기준 4개 이상이어야 한다.
- clean32 heldout 8장은 annotation package에서도 제외한다.
- 자동 suggested label은 학습 positive가 아니라 review hint다.
- `target_positive`는 수동 라벨 파일에서만 확정된다.

## 입력

- c068 reviewed attributes: `eval/c068_reviewed_attribute_label_seed_20260612/reviewed_attribute_labels.jsonl`
- c069 pixel/caption acquisition review: `eval/c069_direct_green_captioning_acquisition_20260612/reviewed_candidate_labels.jsonl`
- c070 caption/search acquisition review: `eval/c070_qwenvl_direct_green_caption_search_20260612/reviewed_candidate_labels.jsonl`
- heldout exclusion: `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`

## 라벨 스키마

수동 annotation에서 허용하는 label은 다음 5개다.

- `target_positive`: direct-green/non-human target positive로 확인된 이미지
- `useful_proxy_non_human`: red eye, side profile, beard/headwear 등 proxy로 쓸 수 있지만 direct-green target은 아닌 이미지
- `guard_false_positive_human`: green/non-human 탐색에서 잘못 잡힌 human/old face/red-eye human guard
- `guard_false_positive_background_object`: green background/object/lighting guard
- `reject_unclear`: 애매하거나 훈련에 쓰면 안 되는 후보

## 산출물

`tools/c071_seed_package.py`는 다음 파일을 만든다.

- `eval/c071_direct_green_seed_package_20260612/annotation_candidates.jsonl`
- `eval/c071_direct_green_seed_package_20260612/annotation_template.csv`
- `eval/c071_direct_green_seed_package_20260612/annotated_review_sheet.jpg`
- `eval/c071_direct_green_seed_package_20260612/summary.json`

`tools/c071_import_manual_labels.py`는 사람이 채운 label 파일을 검증하고 다음을 만든다.

- `eval/c071_direct_green_seed_package_20260612/example_import/imported_manual_labels.jsonl`
- `eval/c071_direct_green_seed_package_20260612/example_import/imported_confirmed_positives.jsonl`
- `eval/c071_direct_green_seed_package_20260612/example_import/import_summary.json`

## 실제 결과

c071 package 결과:

- source row counts: c068 `48`, c069 `48`, c070 `36`
- raw candidate rows: `132`
- unique candidates: `84`
- heldout rows used: `0`
- missing paths: `0`
- suggested proxy candidates: `29`
- human guard candidates: `15`
- background/object guard candidates: `40`

example import는 자동 suggested label을 그대로 넣은 안전 예시다. 결과는 `unique_target_positive_count=0`, decision `external_manual_data_required`다.

## 다음 결정

c071은 학습 준비 패키지를 만든 것이지 학습 통과가 아니다. 다음 학습은 `manual_labels_example.csv`를 실제 사람이 검토해서 최소 4개의 unique `target_positive`를 만든 뒤, importer가 `ready_for_encoder_training`을 반환할 때만 진행한다.
