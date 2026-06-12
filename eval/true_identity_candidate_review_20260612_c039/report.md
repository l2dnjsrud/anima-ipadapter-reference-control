# True Identity Candidate Review c039

작성일: 2026-06-12

## 질문

c038 strict duplicate sanity는 통과했지만, duplicate crop은 true character identity 학습용 positive가 아니다. 이번 단계는 duplicate panel을 제외하고 같은 `SG-page` 안의 서로 다른 panel 후보를 뽑아, 실제 same-character pair 라벨링 재료로 쓸 수 있는지 확인한다.

## 산출물

- 도구: `tools/build_true_identity_candidate_review.py`
- 테스트: `tests/test_true_identity_candidate_review.py`
- 후보 manifest: `eval/true_identity_candidate_review_20260612_c039/candidate_pairs.jsonl`
- 후보 sheet: `eval/true_identity_candidate_review_20260612_c039/candidate_sheet.jpg`

## 생성 규칙

- 같은 panel key의 duplicate crop은 제외한다.
- 같은 `SG-*`와 같은 page number 안에서 서로 다른 panel key 조합만 후보로 만든다.
- 1차 리뷰용으로 24 pair만 생성했다.

## 시각 리뷰

`candidate_sheet.jpg`를 직접 확인했다. 같은 page 안에서 뽑힌 후보라 scene continuity는 있지만, true same-character positive로 바로 쓰기에는 노이즈가 많다.

관찰:

- 일부 pair는 같은 캐릭터 후보로 보인다. 예: 같은 인물의 얼굴/상반신/동작으로 보이는 row가 몇 개 있다.
- 하지만 다른 인물, 배경/건물, 소품 crop, 같은 장면의 다인물 panel이 많이 섞인다.
- 후보 24개만으로도 “자동 same-page non-duplicate = same-character”라고 가정하기 어렵다.
- 학습 manifest로 바로 승격하면 identity metric이 장면/구도/배경 또는 소품 유사도에 오염될 가능성이 높다.

## 결정

결정: `same_page_candidates_need_character_filtering`

같은 `SG-page` 후보 mining은 review sheet 생성용으로는 유용하지만, 학습용 true same-character positive를 자동 생성하기에는 부족하다.

## 다음 루프

1. 후보 pair마다 양쪽 crop이 모두 캐릭터 중심인지 먼저 걸러야 한다.
2. face/upper-body/solo-character 여부를 QwenVL caption 또는 간단한 visual classifier로 판정한다.
3. 같은 캐릭터 여부는 자동 feature score만 믿지 말고, 최소한 small review set에서 사람이 볼 수 있는 label sheet를 유지한다.
4. 다음 실험은 `character_filtered_identity_candidate_mining`으로 진행한다.
