# C101 local positive annotation / teacher rerank plan

C100은 local-real 후보 64개를 만들었지만 reviewed local positive가 `0`개라 C101 학습을 막았다. C101은 학습이 아니라 C102 학습 가능 여부를 결정하는 보수적 라벨/teacher proposal 게이트다.

## 실행 경계

- adapter 학습 없음
- ComfyUI 생성 없음
- checkpoint 생성 없음
- C100 local-real 후보만 C102 greenlight 증거로 사용
- clean32 heldout id 제외
- label schema: `local_positive`, `local_negative`, `unclear`

## 입력 상태

- C100 decision: `c101_blocked_needs_manual_annotation_or_teacher`
- C100 candidate rows: `64`
- C100 review sheet size: `880x4384`

## C102 greenlight 기준

- reviewed local positives >= `8`
- reviewed rows = input candidate rows
- review required count = `0`
- teacher-only positive count = `0`
- heldout leakage = `0`
- missing paths = `0`
- positive label은 direct-green/non-human prior visual review 또는 수동 확인 evidence가 있어야 한다.
