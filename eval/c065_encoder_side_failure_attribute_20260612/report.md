# c065 Encoder-Side Failure Attribute Probe

## 결론

판정: `existing_encoder_feature_separation_not_viable_for_c065_checkpoint`

c065는 새 이미지를 생성하거나 adapter를 추가 학습하지 않았다. c064에서 확인된 hard failure를 해결하려면 encoder-side 또는 attribute-teacher 단계가 필요하다는 가설을 검증하기 위해, clean32 train split만으로 실패 속성 positive/negative pair를 만들고 QwenVL, SigLIP2, PE embedding 공간에서 분리 가능성을 측정했다.

결과적으로 세 encoder 모두 threshold `margin >= 0.05` 및 `AUC >= 0.70`을 넘지 못했다. 특히 heldout07의 핵심인 non-human/green monster 계열 proxy는 QwenVL, SigLIP2, PE 모두 margin이 0 근처 또는 음수였다.

따라서 바로 encoder-side checkpoint를 학습하기에는 데이터와 teacher 신호가 부족하다. 다음 단계는 color dataset에서 direct green/non-human positive를 추가 채굴하거나, cosine embedding 대신 explicit attribute teacher/reranker를 먼저 만드는 쪽이다.

## 입력 데이터

- Plan: `docs/c065_encoder_side_failure_attribute_plan_ko.md`
- Pair manifest: `training/manifests/c065_failure_attribute_pairs_20260612.jsonl`
- Manifest summary: `training/manifests/c065_failure_attribute_pairs_20260612.summary.json`
- Image root: `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`

Manifest 요약:

| 항목 | 값 |
| --- | ---: |
| total pairs | 126 |
| positive pairs | 63 |
| negative pairs | 63 |
| heldout rows used | 0 |
| direct green monster positive count | 0 |
| missing pair paths | 0 |

Bucket 구성:

| bucket | source rows | positive | negative |
| --- | ---: | ---: | ---: |
| non_human_red_pale_profile_proxy | 21 | 21 | 21 |
| beard_headwear_crop | 22 | 22 | 22 |
| old_face_crop | 20 | 20 | 20 |

`direct_green_monster_positive_count=0`이 가장 중요한 데이터 결손이다. c065의 non-human bucket은 red eye/pale villain proxy일 뿐, heldout07의 green monster를 직접 학습하는 positive가 아니다.

## 전체 Encoder 결과

| encoder | margin | AUC | midpoint acc | decision |
| --- | ---: | ---: | ---: | --- |
| QwenVL | -0.005276 | 0.460569 | 0.412698 | fail |
| SigLIP2 | 0.009885 | 0.578231 | 0.595238 | fail |
| PE | -0.026462 | 0.412195 | 0.428571 | fail |

SigLIP2가 전체 평균으로는 가장 낫지만 margin `0.0099`라 teacher로 쓰기에는 너무 약하다.

## Bucket별 결과

### non_human_red_pale_profile_proxy

| encoder | margin | AUC | midpoint acc |
| --- | ---: | ---: | ---: |
| QwenVL | -0.021500 | 0.414966 | 0.404762 |
| SigLIP2 | -0.001178 | 0.503401 | 0.500000 |
| PE | -0.001421 | 0.489796 | 0.476190 |

핵심 실패 bucket이다. 세 encoder 모두 분리하지 못했다. 이 상태에서 adapter나 shallow calibrator를 학습하면 green/non-human identity를 복구하기보다 기존 human/villain template 쪽으로 다시 수렴할 가능성이 높다.

### beard_headwear_crop

| encoder | margin | AUC | midpoint acc |
| --- | ---: | ---: | ---: |
| QwenVL | 0.003458 | 0.477273 | 0.431818 |
| SigLIP2 | 0.014780 | 0.588843 | 0.590909 |
| PE | -0.046308 | 0.361570 | 0.409091 |

heldout05와 관련된 beard/headwear/crop bucket도 안정적이지 않다. c064에서 PE는 heldout05 generated-output comparison에는 도움을 줬지만, train-only attribute pair에서는 오히려 negative가 더 가깝게 나온다.

### old_face_crop

| encoder | margin | AUC | midpoint acc |
| --- | ---: | ---: | ---: |
| QwenVL | 0.002152 | 0.510000 | 0.500000 |
| SigLIP2 | 0.016117 | 0.622500 | 0.600000 |
| PE | -0.030925 | 0.385000 | 0.400000 |

old-face/crop에서는 SigLIP2가 약한 양의 신호를 보였지만 threshold와는 거리가 멀다.

## 판단

c065는 다음을 확인했다.

1. heldout 누수 없는 train-only pair manifest는 만들 수 있다.
2. clean32 안에는 direct green monster positive가 없다.
3. red/pale non-human proxy조차 기존 QwenVL/SigLIP2/PE embedding이 잘 분리하지 못한다.
4. 따라서 지금 가진 encoder 공간에 단순 contrastive objective를 얹는 것만으로는 heldout07 같은 실패를 해결할 가능성이 낮다.

## 다음 결정

다음 루프는 두 갈래 중 하나여야 한다.

1. `direct_green_non_human_mining`
   - color dataset 전체에서 green monster, non-human demon, red eye, side-profile을 직접 가진 후보를 채굴한다.
   - QwenVL caption/attribute prompt와 이미지 검색을 조합해서 positive/negative를 다시 만든다.
   - c065와 같은 pair-separation gate를 먼저 다시 통과시킨다.

2. `attribute_teacher_reranker`
   - cosine embedding 하나로 teacher를 쓰지 않는다.
   - non-human species, green skin, red eye, beard/headwear, crop/profile을 multi-label classifier 또는 reranker로 학습한다.
   - 그 결과를 adapter/encoder-side 학습 loss에 보조 신호로 넣는다.

현재 증거 기준 우선순위는 direct green/non-human 데이터 채굴이다. direct positive가 없는 상태에서 checkpoint를 학습하면 실패 원인을 학습할 재료가 부족하다.
