# Character-Filtered Identity Candidates c040

작성일: 2026-06-12

## 질문

c039 same-page 후보에는 배경, 소품, 다른 인물이 많이 섞였다. Qwen3-VL image-text retrieval로 양쪽 crop이 모두 캐릭터 중심인지 먼저 필터링하면 true same-character 후보 품질이 충분히 좋아지는지 확인한다.

## 산출물

- 도구: `tools/filter_character_candidate_pairs.py`
- 테스트: `tests/test_character_candidate_filter.py`
- 입력 후보: `eval/true_identity_candidate_review_20260612_c039/candidate_pairs.jsonl`
- scored 후보: `eval/character_filtered_identity_candidates_20260612_c040/scored_candidate_pairs.jsonl`
- kept 후보: `eval/character_filtered_identity_candidates_20260612_c040/kept_candidate_pairs.jsonl`
- kept sheet: `eval/character_filtered_identity_candidates_20260612_c040/kept_candidate_sheet.jpg`

## 설정

- Encoder: `Qwen/Qwen3-VL-Embedding-2B`
- Character texts: solo/upper-body/face/single-character prompts
- Non-character texts: background/object/building/group-scene prompts
- Character score: `max(character_text_scores) - max(non_character_text_scores)`
- Keep rule: both sides have character score `>= 0.15`

## 결과

- input pairs: 24
- kept pairs: 14
- threshold: 0.15

## 시각 리뷰

`kept_candidate_sheet.jpg`를 확인했다. character-centered filter는 건물/배경/소품만 있는 후보를 일부 줄였지만, true same-character positive를 자동으로 확정하기에는 부족했다.

남은 문제:

- 같은 장면의 다른 인물 pair가 남는다.
- 얼굴/상반신과 몸통 crop pair가 남아 identity label로 애매하다.
- 같은 인물로 보이는 후보도 있지만, 14개 전체가 학습용 positive로 쓰일 수준은 아니다.

## 결정

결정: `character_filter_reduces_noise_not_identity_labels`

QwenVL character-centered filtering은 candidate sheet 품질을 조금 올리는 보조 필터로 유지한다. 하지만 true same-character label을 자동 생성하는 단계로 승격하지 않는다.

## 다음 루프

1. kept 후보를 사람/시각 리뷰 가능한 label sheet로 만들고 `same_character`, `different_character`, `unclear` 라벨을 붙인다.
2. 얼굴/상반신이 양쪽 모두 보이는 pair를 우선한다.
3. 몸통/손/소품/배경 crop은 positive 후보에서 제외한다.
4. 충분한 positive label이 모이면 SigLIP layer `-6` pooled와 `mean_max_token`을 다시 평가한다.
