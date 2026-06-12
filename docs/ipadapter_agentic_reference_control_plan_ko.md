# Agentic Reference-Control 개선 계획

작성일: 2026-06-12
범위: `/home/wktwin/anima-ipadapter-reference-control`

## 목적

c035 이후의 문제는 “학습을 더 오래 돌릴지”가 아니라 “무엇이 실패했는지 보고 다음 모델/데이터/프롬프트 경로를 고르는 루프가 있는지”다. 그래서 다음 반복은 InterleaveThinker의 planner/critic 운영 방식을 참고해 `agentic_reference_control_loop`를 먼저 만든다. 이 루프는 모델 자체가 아니라 평가와 다음 행동 선택을 담당한다.

## 외부 연구 반영

| 자료 | 2026-06-12 확인 내용 | 적용 방식 |
|---|---|---|
| `zhengdian1/InterleaveThinker` | planner agent가 입력 순서를 조직하고 critic agent가 생성 결과의 deviation을 찾아 instruction을 고치는 interleaved generation pipeline이다. 공식 저장소는 paper, models, training, inference 공개를 공지했다. | 참조 제어 실패를 `critic`이 구조화하고, 다음 prompt/model/data route를 고르는 루프 아이디어만 가져온다. |
| `zlab-princeton/i1` | 3B T2I 모델을 위한 open recipe, data processing, recaptioning, JAX training, PyTorch inference 구조를 공개한다. | reference pair mining, recaption, long attribute prompt 품질 개선과 향후 open T2I backbone 비교 근거로 쓴다. |

둘 다 IP-Adapter checkpoint나 Anima native K/V 주입 구조를 직접 제공하지 않는다. 따라서 현재 SigLIP/QwenVL adapter를 완성 모델로 포장하거나, 두 repo를 바로 대체 모델로 부르는 것은 금지한다.

공식 확인 출처:

- `https://github.com/zhengdian1/InterleaveThinker`, 확인 SHA `440c1b879cd4913b0382761f7bfa8297a32dc7d6`
- `https://github.com/zlab-princeton/i1`, 확인 SHA `cd6a34fd8e7fa7a0b7de36ff4602363e607f8a72`

## 다음 코드 작업

1. `reference-control audit manifest`를 만든다.
   - 입력: c035 reference suite, no-IP output, SigLIP output, PE metric, visual audit.
   - 출력: JSONL row마다 `case_id`, `reference`, `target_prompt`, `no_ip_output`, `ip_output`, `metric_delta`, `visual_flags`, `failure_tags`, `next_route`를 기록한다.
   - `next_route` 후보는 `prompt_patch`, `pair_mining`, `stronger_encoder`, `line_control`, `hold`로 제한한다.

2. `critic rules v1`을 만든다.
   - `identity/distinctive trait` 실패면 `stronger_encoder` 또는 `pair_mining`으로 보낸다.
   - palette/costume만 실패면 `prompt_patch`와 attribute vocab 확장을 먼저 시도한다.
   - line/page 구조 실패면 IP-Adapter가 아니라 `line_control`로 보낸다.
   - metric은 좋아도 visual audit가 실패하면 성공으로 기록하지 않는다.

3. 다음 학습 후보를 audit 결과로 고른다.
   - `stronger_encoder`가 다수면 anime/manhwa 특화 encoder 또는 image encoder adaptation을 준비한다.
   - `pair_mining`이 다수면 same-character positive/negative mining을 먼저 실행한다.
   - `prompt_patch`가 다수면 QwenVL attribute prompt vocabulary를 확장한다.

## 통과 기준

첫 번째 audit loop는 새 모델 학습 성공을 주장하지 않는다. 통과 기준은 다음 네 가지다.

1. c035 전체 케이스를 누락 없이 JSONL로 묶는다.
2. 각 row가 사람이 읽을 수 있는 실패 태그와 다음 route를 가진다.
3. route 분포가 다음 학습 또는 데이터 작업을 하나 이상 명확히 선택한다.
4. 기존 c035 decision인 `not_ready`와 완성 모델 아님 판단을 유지한다.

## 다음 루프 산출물

- `tools/build_reference_control_audit_manifest.py`
- `tests/test_reference_control_audit_manifest.py`
- `eval/siglip_runtime_quality_20260612_c035_suite_v1/reference_control_audit_manifest.jsonl`
- `eval/siglip_runtime_quality_20260612_c035_suite_v1/reference_control_audit_summary.md`

이 산출물이 만들어진 뒤에야 새 학습을 시작한다. 학습 시작 전에는 어떤 실패가 encoder 문제인지, pair mining 문제인지, prompt 문제인지 먼저 나누는 것이 우선이다.

## 2026-06-12 실행 결과

v1 산출물은 생성 완료됐다. c035 32 row 기준 route 분포는 다음과 같다.

| route | rows |
|---|---:|
| `prompt_patch` | 6 |
| `pair_mining` | 0 |
| `stronger_encoder` | 16 |
| `line_control` | 0 |
| `hold` | 10 |

해석: 다음 학습 후보는 `stronger_encoder`가 우선이다. prompt patch만으로 해결될 케이스보다 identity/distinctive trait 실패가 더 많기 때문이다.

## 2026-06-12 c036 metric gate 추가

audit v1 이후 QwenVL pooled embedding이 stronger-encoder 학습의 주 지표가 될 수 있는지 확인했다.

- 도구: `tools/score_auto_caption_qwenvl_metrics.py`
- 산출물: `eval/qwenvl_metric_probe_20260612_c036_c035/report.md`
- 결정: `qwenvl_pooled_metric_auxiliary_only`

QwenVL pooled metric은 c035에서 `siglip_ref_retrieval_w14`를 improved rate `0.90625`로 높게 평가했지만, identity-fail row의 평균 uplift가 identity-pass row보다 높았다. 따라서 agentic loop의 다음 route는 장기 학습이 아니라 `identity_positive_negative_feature_probe`다. 같은 캐릭터/다른 캐릭터 pair를 만들고 QwenVL, SigLIP, PE feature가 실제 identity를 분리하는지 먼저 확인한다.
