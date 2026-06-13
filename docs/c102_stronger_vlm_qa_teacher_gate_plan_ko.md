# C102 stronger VLM QA teacher gate plan

C101은 64개 local-real 후보를 모두 보수적으로 라벨링했지만 `local_positive=0`이었다. C102는 학습이 아니라 로컬 생성형 VLM QA teacher로 후보를 다시 판정하는 획득 게이트다.

## 실행 경계

- adapter 학습 없음
- ComfyUI 생성 없음
- checkpoint 생성 없음
- C100/C101 local-real 후보 64개만 C103 greenlight 증거로 사용
- 기존 C101 `local_negative`는 VLM 단독 판단으로 뒤집지 않음

## 확인한 VLM/teacher 표면

[
  {
    "surface": "repo_qwen3vl_embedding",
    "status": "embedding_only"
  },
  {
    "surface": "repo_c070_caption_search",
    "status": "sidecar_caption_heuristic_only"
  },
  {
    "surface": "local_qwen3vl_2b_thinking",
    "status": "runnable_but_verbose_thinking_output"
  },
  {
    "surface": "local_qwen3vl_8b_instruct",
    "status": "selected"
  }
]

## 선택한 teacher

- status: `local_qwen3vl_8b_instruct_runnable`
- model_path: `/data/ai/models/LLM/Qwen-VL/Qwen3-VL-8B-Instruct`
- reason: Qwen3-VL-8B-Instruct is a local HF-format generative VLM and follows compact labels better than Qwen3-VL-2B-Thinking.

## 입력 상태

- C101 decision: `c102_blocked_needs_manual_annotation_or_teacher`
- C101 reviewed local positive: `0`
- C100 candidate rows: `64`
- heldout count: `8`

## C103 greenlight 기준

- 64 candidates covered
- QA/manual confirmed local positives >= `8`
- teacher_only_positive_count = `0`
- heldout_leakage_count = `0`
- missing_path_count = `0`
