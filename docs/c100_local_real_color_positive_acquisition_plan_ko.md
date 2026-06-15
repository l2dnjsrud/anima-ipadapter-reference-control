# C100 local real-color direct-green positive acquisition plan

C099에서 local real-color direct-green/non-human confirmed positive가 `0`개로 확인되었다. C100은 학습 루프가 아니라 C101 학습을 시작해도 되는지 가르는 후보 확보/검수 게이트다.

## 실행 경계

- 학습 실행 없음
- ComfyUI 생성 없음
- checkpoint 생성 없음
- clean32 heldout id 후보 제외
- c074 external source와 c097 synthetic source는 local-real greenlight 증거로 세지 않는다.

## 입력 상태

- C099 decision: `c100_blocked_needs_annotation_or_teacher`
- c066 direct-green confirmed positives: `0`
- c066 total candidates: `120`

## C101 greenlight 기준

- reviewed local positives >= `8`
- heldout leakage = `0`
- missing paths = `0`
- 후보 review sheet와 manifest가 존재해야 한다.
