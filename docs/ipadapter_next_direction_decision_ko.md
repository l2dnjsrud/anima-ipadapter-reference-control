# IP-Adapter 다음 방향 결정

작성일: 2026-06-12
범위: `/home/wktwin/anima-ipadapter-reference-control`

## 결론

SELECTED: `agentic_reference_control_loop` -> `train_stronger_encoder`

현재 SigLIP attribute-reference recipe는 ComfyUI에서 실행 가능하고, 넓은 의미의 색감/의상/표정/무협 스타일 신호는 가져온다. 하지만 c035 32-case 단일 캐릭터 검증에서 metric gate와 identity/distinctive-trait gate를 통과하지 못했다. 따라서 지금 체크포인트를 “고퀄 reference-control 모델”로 패키징하는 것은 이르다.

반대로 `stop_siglip_branch`로 완전히 중단하기도 이르다. 한 장 overfit은 성공했고, PE-style query patch, PE-space 초기화, retrieval loss, 자동 속성 프롬프트를 거치며 실제 개선 신호가 있었다. 다음 방향은 현재 recipe를 실험용으로 보존하면서, 더 강한 이미지 인코더/정렬 학습을 준비하는 것이다.

중요: generic prompt reference-only is not solved.

2026-06-12에 확인한 `zhengdian1/InterleaveThinker`와 `zlab-princeton/i1`는 이 결론을 보강한다. InterleaveThinker는 planner/critic 방식의 외부 제어 루프 근거로 유용하고, i1은 open T2I data/training recipe 근거로 유용하다. 다만 둘 다 Anima IP-Adapter checkpoint 구조의 직접 대체재는 아니다.

공식 확인 출처:

- `https://github.com/zhengdian1/InterleaveThinker`, 확인 SHA `440c1b879cd4913b0382761f7bfa8297a32dc7d6`
- `https://github.com/zlab-princeton/i1`, 확인 SHA `cd6a34fd8e7fa7a0b7de36ff4602363e607f8a72`

## 선택지 판단

| 선택지 | 판단 | 이유 |
| --- | --- | --- |
| `package_current_recipe` | rejected | c035 best uplift `+0.0577`, improved rate `0.65625`로 metric gate 실패. identity/distinctive trait `16/32`로 visual gate 실패. |
| `agentic_reference_control_loop` | chosen first | InterleaveThinker식 planner/critic 판단을 직접 모델 구조가 아니라 평가/수정 루프로 가져온다. c035 실패 케이스마다 prompt, reference, output, visual audit를 묶어 다음 route를 결정한다. |
| `train_stronger_encoder` | chosen after audit loop | broad style/palette/costume 신호는 있으나, frozen SigLIP2 adapter-only로 identity/props/non-human trait가 부족하다. 더 강한 encoder 또는 image-encoder adaptation이 필요하다. |
| `stop_siglip_branch` | rejected | one-image overfit, c034 small-suite pass, c035 broad-style pass가 있어 구조적으로 불가능하다고 볼 근거는 부족하다. |

## 검증 요약

| 경로 | 목표 | 결과 | 현재 상태 |
| --- | --- | --- | --- |
| PE-Core baseline | 작동 가능한 reference-control baseline 확보 | ComfyUI contact-sheet에서 best scale `1.0`, mean uplift `+0.0937`, improved rate `87.5%` | baseline으로 유지 |
| SigLIP c034 | 8-case 자동 속성 프롬프트 검증 | `siglip_ref_retrieval_w14` uplift `+0.1452`, 7/8 개선. PE metric caveat 있음 | 가능성 확인용 |
| SigLIP c035 | 32-case single-character suite 검증 | best uplift `+0.0577`, improved rate `0.65625`, identity gate `16/32` | not ready |
| QwenVL adapter-only | QwenVL embedding을 adapter에 직접 연결 | 출력 변화는 있으나 generic wuxia/interior collapse | 현재 방식 보류 |
| line-art colorization | IP-Adapter 단독 선화 채색 | 색/스타일 압력은 있으나 구조 보존 실패. EasyControl 결합 필요 | 별도 spatial-control track |
| InterleaveThinker | agentic interleaved generation 연구 | planner/critic loop가 출력 편차를 찾고 지시를 수정한다 | reference-control audit loop 참고 |
| i1 | fully open T2I recipe | data processing, recaptioning, 1024 T2I 학습/추론 구조 공개 | data/backbone recipe 참고 |

## c035 근거

산출물:

- `eval/siglip_runtime_quality_20260612_c035_suite_v1/contact_sheet.jpg`
- `eval/siglip_runtime_quality_20260612_c035_suite_v1/report.md`
- `eval/siglip_runtime_quality_20260612_c035_suite_v1/pe_similarity_metrics.json`
- `eval/siglip_runtime_quality_20260612_c035_suite_v1/visual_audit.json`

수치:

| variant | mean uplift | improved rate | decision |
| --- | ---: | ---: | --- |
| `siglip_kv_init_w14` | +0.0292 | 0.65625 | fail |
| `siglip_ref_retrieval_w14` | +0.0577 | 0.65625 | fail |

시각 감사:

- palette/costume/expression/framing: `31/32`
- identity/distinctive trait: `16/32`
- non-human/special trait: `0/1`
- 주요 실패: 검은 장발 무협 캐릭터, 보라색/밤 궁전 배경, 붉은 눈, generic official/elder template으로 수렴

## 다음 학습 방향

지금 바로 장기 학습을 시작하지 않는다. 먼저 아래 prerequisite를 만족해야 한다.

1. Agentic audit loop
   - c035처럼 이미 생성된 no-IP/IP output, reference image, prompt, metric, visual audit를 하나의 manifest로 묶는다.
   - critic 출력은 누락 identity, 누락 palette, 누락 prop, non-human trait, prompt patch, model route를 구조화한다.
   - 이 결과가 `prompt_patch` 문제인지, pair mining 문제인지, encoder adaptation 문제인지 먼저 나눈다.

2. 데이터
   - color dataset 기반 단일 캐릭터 검증 suite를 더 확장한다.
   - 같은 캐릭터 그룹 또는 identity-positive/negative pair를 mining한다.
   - i1식 data processing/recaptioning recipe를 참고해 이미지-캡션 품질과 long prompt coverage를 올린다.
   - Wenaka dataset을 쓸 경우, 다운로드 전 명시 승인과 ref/tgt pairing rule이 필요하다.

3. 모델
   - frozen SigLIP2 adapter-only를 반복하지 않는다.
   - 후보는 anime/manhwa 특화 SigLIP/PE-like encoder, QwenVL feature calibrator, 또는 image encoder LoRA/adaptation이다.
   - FaceID-like 방향은 실사 InsightFace를 그대로 믿지 말고 manhwa character metric model을 먼저 검증한다.

4. 평가
   - c035와 같은 32-case 이상 single-character suite를 기본 gate로 둔다.
   - gate는 metric과 visual audit를 같이 본다.
   - 통과 기준은 최소 best uplift `>= +0.10`, improved rate `>= 0.75`, identity/distinctive trait `>= 18/32`이다.

## 실행 명령

현재 단계에서는 full training command를 실행하지 않는다. 다음에 할 일은 데이터/encoder 준비 검증이다.

새 agentic audit loop의 상세 계획은 `docs/ipadapter_agentic_reference_control_plan_ko.md`에 둔다.

```bash
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python -m pytest \
  tests/test_reference_prompting.py \
  tests/test_siglip_auto_caption_eval.py \
  tests/test_comfyui_workflows.py \
  tests/test_native_siglip.py -q
```

```bash
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python tools/validate_reference_suite.py \
  eval/siglip_reference_suite_v1_20260612/reference_suite_v1.jsonl \
  --data-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best
```

## 현재 사용 가능한 recipe

실험용으로는 다음 recipe를 유지한다.

- node family: `AnimaSigLIPIPAdapterLoader` -> `AnimaSigLIPEncodeImage` -> `AnimaSigLIPIPAdapterApply`
- 기본 checkpoint: `anima_siglip_ip_adapter_single_character_clean32_pe_retrieval_0128_20260611.safetensors`
- fallback checkpoint: `anima_siglip_ip_adapter_single_character_clean32_pe_space_init_0512_20260611.safetensors`
- runtime label: `siglip_ref_retrieval_w14`
- weight: `1.4`
- start/end: `0.0 / 0.85`
- 필수 조건: 자동 visible-attribute prompt를 같이 사용

이 recipe는 “품질 좋은 실험 경로”이지, 아직 “바로 믿고 쓰는 완성 모델”은 아니다.
