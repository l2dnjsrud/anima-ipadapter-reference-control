# C099 real-color reference data gate plan

C098은 작동했지만 승격되지 않았다. C099는 같은 SigLIP hard-shape LoRA를 반복하지 않고 실제 color 데이터 sufficiency를 먼저 가른다. 즉, C097/C098 반복 금지 데이터 게이트다.

## 실행 경계

- 학습 실행 없음
- ComfyUI 실행 없음
- checkpoint 생성 없음
- heldout row 후보 제외
- 외부 direct-green과 synthetic hard-shape는 real local color sufficiency로 세지 않는다.

## C001 inventory 기준

- clean32 train rows: `32`
- clean32 heldout rows: `8`
- c052 positive pairs: `29`
- c066 direct-green positive count: `0`
- c097 selected synthetic hard-shape rows: `56`

## C002 candidate/readiness 기준

- `c099_candidate_manifest.jsonl`에서 source_type을 `real_local_color`, `external_real_direct_green`, `synthetic_hard_shape`로 분리한다.
- local real-color direct-green/non-human confirmed positive가 없으면 C100 학습으로 바로 가지 않는다.
- c074 external real source와 c097 synthetic hard-shape는 별도 fallback/teacher track으로만 기록한다.

## C003 publication 기준

- `docs/ipadapter_research_journal_ko.md`에 C099 결과를 기록한다.
- py_compile, focused pytest, artifact consistency, git diff check를 통과한다.
- 실험 완료 후 commit/push까지 끝낸다.
