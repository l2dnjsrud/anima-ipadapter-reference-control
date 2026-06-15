# Reviewed Identity Candidates c041

작성일: 2026-06-12

## 질문

c040 character-centered filter로 남긴 14개 pair를 사람이 보기 좋은 라벨 manifest로 만들면, true same-character feature probe나 다음 학습 seed로 바로 쓸 수 있는지 확인한다.

## 산출물

- 도구: `tools/build_reviewed_identity_manifest.py`
- 테스트: `tests/test_reviewed_identity_manifest.py`
- 입력 후보: `eval/character_filtered_identity_candidates_20260612_c040/kept_candidate_pairs.jsonl`
- 수동 시각 라벨: `eval/reviewed_identity_candidates_20260612_c041/manual_visual_labels.jsonl`
- reviewed manifest: `eval/reviewed_identity_candidates_20260612_c041/reviewed_candidate_pairs.jsonl`
- usable positives: `eval/reviewed_identity_candidates_20260612_c041/usable_positive_pairs.jsonl`
- different-character negatives: `eval/reviewed_identity_candidates_20260612_c041/different_character_pairs.jsonl`
- unclear pairs: `eval/reviewed_identity_candidates_20260612_c041/unclear_pairs.jsonl`
- reviewed sheet: `eval/reviewed_identity_candidates_20260612_c041/reviewed_candidate_sheet.jpg`

## 라벨 기준

- `same_character`: 같은 캐릭터로 보이는 pair
- `different_character`: 다른 캐릭터로 보이는 pair
- `unclear`: 몸통/하반신/멀티 캐릭터/작은 배경 인물 때문에 identity target이 애매한 pair
- `positive_usable`: 같은 캐릭터로 보일 뿐 아니라 양쪽이 학습/검증 positive로 쓸 만큼 target이 명확한 pair

## 결과

- reviewed rows: 14
- `same_character`: 6
- `different_character`: 3
- `unclear`: 5
- `positive_usable`: 4

## 시각 리뷰

`reviewed_candidate_sheet.jpg`를 확인했다. c040 filter는 캐릭터가 없는 crop을 일부 줄였지만, 수동 리뷰 결과 usable positive는 4개뿐이다. 남은 문제는 다음과 같다.

- multi-character crop에서 어떤 인물을 reference target으로 볼지 애매하다.
- torso-only crop은 같은 의상처럼 보여도 identity positive로 쓰기 어렵다.
- 같은 SG page 안에서도 다른 인물 pair가 남는다.

## 결정

결정: `reviewed_seed_too_small_for_training_gate`

c041 manifest는 true same-character feature probe의 작은 seed로는 사용할 수 있다. 그러나 이 4개 usable positive만으로 adapter 학습이나 metric-head 학습을 시작하기에는 너무 작다.

## 다음 루프

1. c041의 4 usable positive와 3 different-character negative로 SigLIP layer `-6` pooled, SigLIP `mean_max_token`, QwenVL pooled, PE pooled가 같은/다른 캐릭터를 어느 정도 분리하는지 작은 sanity probe를 돌린다.
2. 동시에 mining 범위를 같은 page 14개에서 더 넓혀, face/upper-body 중심 crop 조건을 더 강하게 건다.
3. usable positive가 최소 수십 개가 되기 전까지 adapter 학습으로 넘어가지 않는다.
