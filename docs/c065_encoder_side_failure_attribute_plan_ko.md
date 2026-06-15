# c065 Encoder-Side Failure Attribute Plan

작성일: 2026-06-12

## 목적

c064에서 hard failure 3개를 QwenVL, SigLIP2, PE embedding 공간으로 재측정한 결과, adapter continuation이나 calibrator-only 반복만으로는 heldout07 같은 non-human green monster side-profile 붕괴를 고치기 어렵다고 판단했다. c065의 목적은 바로 큰 학습을 시작하는 것이 아니라, encoder-side checkpoint 또는 attribute-teacher 학습으로 넘어갈 수 있는 최소 전제 조건을 검증하는 것이다.

핵심 질문은 다음이다.

- color single-character train split 안에서 실패 속성을 heldout 누수 없이 positive/negative pair로 만들 수 있는가?
- 기존 QwenVL, SigLIP2, PE embedding이 이 pair를 어느 정도 분리하는가?
- 분리 신호가 약하면 더 긴 IP-Adapter 학습이 아니라 direct green/non-human positive 채굴 또는 별도 attribute teacher가 먼저 필요한가?

## c064에서 넘어온 근거

c064 판정은 `encoder_side_checkpoint_required_for_hard_failures`였다.

- QwenVL은 heldout01만 support했고 heldout05/07은 실패했다.
- SigLIP2는 heldout01/05/07 모두 실패했다.
- PE는 heldout05만 support했고 heldout01/07은 실패했다.
- heldout07 `non-human-green-monster-side-profile-red-eye`는 세 encoder 모두 `no_ip`를 1위로 두었다.

따라서 c065는 새 adapter head를 그대로 더 학습하는 실험이 아니다. 먼저 실패 속성 자체가 feature space에서 분리 가능한지 확인한다.

## 데이터 경계

사용 데이터:

- Train manifest: `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- Heldout manifest: `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- Attribute source: `eval/qwenvl_c061_instruction_calibration_gate_20260612/summary.json`
- Image root: `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`

규칙:

- c065 pair manifest에는 train split row만 사용한다.
- heldout manifest에 있는 `ref_id`는 anchor/candidate 어느 쪽에도 넣지 않는다.
- heldout07은 평가 실패 원인으로만 사용하고, 학습/probe pair에는 직접 넣지 않는다.
- 이미지 파일은 `<image_root>/<ref_id>.jpg` 존재 여부를 확인한다.

## 실패 속성 bucket

### non_human_red_pale_profile_proxy

목표는 heldout07의 non-human/green monster/profile/red eye 실패를 다루는 것이지만, 현재 clean32 train split에는 `green monster`, `green non-human`, `green-skinned demon`, `green demon` 같은 직접 green monster 양성이 없다. 또한 `sharp fangs visible`와 `side profile portrait`까지 positive 기준으로 쓰면 train32 대부분이 걸려 negative pair를 만들 수 없다. 그래서 c065에서는 다음 proxy만 pair-defining keyword로 사용한다.

- `red glowing demonic eye`
- `pale purple-skinned villain`

이 bucket은 green monster를 해결했다는 뜻이 아니다. direct green positive가 0이면 summary에 그대로 기록하고, c065 결과에서 별도 데이터 채굴 필요 여부를 결정한다.

### beard_headwear_crop

heldout05의 old bearded official/headwear/upper-body crop 계열을 다룬다.

- `old bearded martial arts master`
- `middle-aged court official with black hat`
- `black official hat`
- `black mustache official face`
- `upper body close-up portrait`

### old_face_crop

heldout01/05에 걸친 old face, beard, crop, speech-bubble/crop context 계열을 다룬다.

- `old bearded martial arts master`
- `bald old monk`
- `elder`
- `elderly`
- `wrinkled`
- `upper body close-up portrait`

## Pair construction

각 row는 기존 `tools/score_identity_pair_probe.py`와 호환되도록 다음 필드를 가진다.

- `pair_id`
- `label`: `positive` 또는 `negative`
- `anchor_id`
- `candidate_id`
- `anchor_group`
- `candidate_group`
- `attribute_bucket`
- `anchor_attributes`
- `candidate_attributes`
- `matched_keywords`
- `negative_reason`
- `source_split`

Positive pair는 같은 bucket 안의 train row끼리 만든다. Negative pair는 bucket anchor와 해당 bucket keyword가 없는 train row를 묶는다. 같은 이미지 self-pair는 만들지 않는다.

## c065 실행 순서

1. Pair manifest 생성
   - `training/manifests/c065_failure_attribute_pairs_20260612.jsonl`
   - `training/manifests/c065_failure_attribute_pairs_20260612.summary.json`

2. Feature-separation prerequisite
   - QwenVL, SigLIP2, PE로 같은 pair manifest를 score한다.
   - 출력은 `eval/c065_encoder_side_failure_attribute_20260612/` 아래에 둔다.
   - bucket별 positive mean, negative mean, separation margin, pairwise AUC를 기록한다.

3. 결정
   - 어떤 encoder라도 bucket별 margin `>= 0.05`와 AUC `>= 0.70`을 안정적으로 보이면 그 encoder를 teacher 후보로 본다.
   - non-human proxy가 약하고 direct green positive가 0이면, 바로 checkpoint 학습으로 가지 않고 color dataset에서 green/non-human direct positive를 추가 채굴한다.
   - beard/headwear/old-face만 강하면 해당 속성은 attribute teacher 보조 loss 후보로 남기되, heldout07 해결과는 분리해서 판단한다.

## 다음 quality gate

c065 prerequisite가 통과해도 최종 품질 pass는 아니다. 실제 checkpoint 또는 attribute teacher를 만든 뒤에는 clean32+heldout8 ComfyUI API generation gate로 돌아가야 한다.

최소 gate:

- no-IP
- 현재 best runtime preset `blend_species_face`
- 새 encoder-side/attribute-teacher 후보

판단 기준:

- heldout07 non-human side-profile에서 no-IP보다 확실히 좋아야 한다.
- heldout05 beard/headwear/crop을 유지해야 한다.
- 전체 clean32+heldout8에서 기존 best preset보다 시각 audit과 PE/QwenVL 보조 metric 모두에서 악화되지 않아야 한다.
