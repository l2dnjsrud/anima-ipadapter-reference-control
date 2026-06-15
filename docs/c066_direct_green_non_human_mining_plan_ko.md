# c066 Direct Green / Non-Human Mining Plan

작성일: 2026-06-12

## 목적

c065는 clean32 train split 안에 직접 `green monster` positive가 없고, 기존 QwenVL, SigLIP2, PE feature space가 `red glowing demonic eye` / `pale purple-skinned villain` proxy pair를 충분히 분리하지 못한다는 것을 확인했다. c066의 목적은 바로 encoder-side checkpoint 학습을 시작하는 것이 아니라, 실제 color dataset 안에 heldout 누수 없이 쓸 수 있는 직접 green / non-human 후보가 충분한지 확인하는 것이다.

핵심 질문은 다음이다.

- local color dataset 1,571장 안에서 clean32 heldout row를 제외하고 직접 green 후보를 찾을 수 있는가?
- sidecar caption, c061 selected attributes, c065 pair attributes 중 어느 근거가 후보 채굴에 유효한가?
- 직접 green 후보가 green character/non-human failure target인지, 단순 배경/소품/조명인지 구분할 수 있는가?
- 기존 QwenVL, SigLIP2, PE feature가 새 후보 pair를 분리한다면 encoder-side checkpoint로 넘어갈 수 있는가?

## 데이터 경계

사용 데이터:

- Image root: `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`
- Sidecar captions: image root 아래 `.txt`
- Clean32 train manifest: `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- Clean32 heldout manifest: `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- c061 attribute source: `eval/qwenvl_c061_instruction_calibration_gate_20260612/summary.json`
- c065 attribute source: `training/manifests/c065_failure_attribute_pairs_20260612.jsonl`

규칙:

- clean32 heldout manifest의 `ref_id`는 후보 manifest와 pair manifest에서 제외한다.
- heldout07 `green monster face with red glowing eye`는 실패 원인 및 판단 기준으로만 사용한다.
- sidecar caption은 후보 근거로 기록하지만, 현재 대부분 style-level caption이므로 직접 positive 근거로 과신하지 않는다.
- 이미지 green-pixel scan은 직접 green 후보를 찾기 위한 보조 신호이며, character/non-human label로 자동 승격하지 않는다.

## 후보 bucket

Positive 후보:

- `direct_green_pixel_candidate`: 이미지 자체에서 green/strong-green pixel ratio가 높은 후보. 단, 시각 확인상 배경/소품일 수 있으므로 `direct green character`와 동일시하지 않는다.
- `red_eye_proxy`: `red glowing demonic eye` 속성을 가진 train 후보.
- `pale_non_human_proxy`: `pale purple-skinned villain` 속성을 가진 train 후보.
- `fang_profile_proxy`: `sharp fangs visible` 또는 `side profile portrait` 계열 후보. 너무 넓게 잡히면 feature probe에서 별도 bucket으로 약하게 취급한다.

Negative 후보:

- `human_negative`: `human martial arts character`, `young clean-shaven warrior`, `female noble court character` 등 명시적 human 계열.
- `old_headwear_negative`: `old bearded martial arts master`, `middle-aged court official with black hat`, `black official hat`, `upper body close-up portrait` 등 heldout05 계열 control.
- `generic_caption_negative`: sidecar caption만 있고 target attribute가 없는 후보.

## 실행 순서

1. Candidate manifest 생성
   - `training/manifests/c066_direct_green_non_human_candidates_20260612.jsonl`
   - `training/manifests/c066_direct_green_non_human_candidates_20260612.summary.json`
   - 후보마다 `candidate_source`, `source_bucket`, `matched_keywords`, `selected_attributes`, `caption`, `green_ratio`, `strong_green_ratio`, `red_ratio`, `path_exists`, `heldout_excluded`를 기록한다.

2. Review sheet 생성
   - `eval/c066_direct_green_non_human_mining_20260612/green_top16_probe_sheet.jpg`
   - top green pixel 후보가 실제 green character인지 확인한다.
   - 현재 초기 확인 결과 top 후보들은 대부분 잎, 배경, 방, 찻잔, 소품으로 보이며, 직접 green non-human character positive로 자동 확정하기 어렵다.

3. Pair probe manifest 생성
   - `training/manifests/c066_direct_green_non_human_pairs_20260612.jsonl`
   - 같은 positive bucket 안의 후보끼리 positive pair를 만들고, matched negative control과 negative pair를 만든다.

4. Feature-separation probe
   - QwenVL, SigLIP2, PE로 같은 pair manifest를 score한다.
   - 출력은 `eval/c066_direct_green_non_human_mining_20260612/` 아래에 둔다.
   - bucket별 margin, AUC, midpoint accuracy를 기록한다.

## Stop Gate

다음 조건을 모두 만족하지 못하면 encoder-side checkpoint 학습을 시작하지 않는다.

- clean32 heldout을 제외한 직접 green character/non-human positive가 충분히 존재한다.
- positive/negative pair가 bucket별로 최소 2개 이상 만들어진다.
- QwenVL, SigLIP2, PE 중 하나가 target bucket에서 margin `>= 0.05`, AUC `>= 0.70`을 보인다.
- review sheet 기준 green 후보가 배경/소품/조명 위주가 아니라 reference-control 실패 속성에 해당한다.

실패하면 다음 단계는 더 긴 IP-Adapter continuation이 아니라, 직접 attribute teacher/reranker 또는 별도 annotation/captioning 단계다.
