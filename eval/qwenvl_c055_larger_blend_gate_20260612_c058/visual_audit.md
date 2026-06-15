# c058 Visual Audit

평가 대상은 `blend_prev14_c05504` runtime recipe다. 기준 비교는 `no_ip`, 이전 QwenVL retrieval checkpoint `prev_w14`, 그리고 이전 retrieval checkpoint 뒤에 c055 mixed checkpoint를 약하게 추가 적용한 `blend_prev14_c05504`다.

## 요약 판단

`blend_prev14_c05504`는 40개 샘플 전체 평균에서는 현재 가장 강한 runtime recipe다. PE metric과 QwenVL metric 모두에서 `prev_w14`보다 평균 cosine/uplift가 높고, 특히 `no_ip` 대비 개선률은 QwenVL 기준 0.800으로 올라갔다.

하지만 contact sheet 시각 감사 기준으로는 final high-quality reference-control pass가 아니다. 얼굴/복식/색감 같은 broad character cue는 이전보다 자주 반영되지만, 정확한 pose, page layout, speech bubble, fan/hand prop, 특수한 괴물 머리 silhouette 같은 구조 디테일은 아직 불안정하다. 따라서 c058의 결정은 “current best runtime candidate”이지 “바로 믿고 쓰는 완성 모델”이 아니다.

## Heldout Audit

| sample | visual judgment |
| --- | --- |
| `heldout00` | reference는 창백한 장발 악역의 측후면/부채 느낌이다. `no_ip`는 보라색 얼굴과 부채가 섞여 틀어졌고, `prev_w14`/`blend`는 장발과 어두운 복식은 반영하지만 창백한 얼굴 방향과 부채/측후면 구도는 정확하지 않다. `blend`가 silhouette은 약간 낫지만 pass는 아니다. |
| `heldout01` | 노년 남성/분노/말풍선 reference다. `prev_w14`와 `blend` 모두 angry male cue는 반영하지만 노년성, 수염 밀도, 말풍선 위치는 약하다. `blend`가 뚜렷한 우위라고 보기 어렵다. |
| `heldout02` | bald old monk reference다. `no_ip` 대비 `prev_w14`와 `blend`가 bald/old/beard cue를 더 잘 반영한다. `blend`는 수염/두상/복식이 조금 더 강하지만, 정확한 스타일과 crop은 아직 다르다. |
| `heldout03` | 붉은 머리 여성이 앉은 장면이다. `prev_w14`와 `blend` 모두 red-haired seated woman cue를 잡으며, `blend`가 조금 더 정돈되어 보인다. 다만 손동작과 말풍선/패널 구성은 보존하지 못한다. |
| `heldout04` | 검은 머리 close-up과 speech bubble reference다. `prev_w14`와 `blend`는 black-haired side-face cue를 반영하지만 speech bubble과 crop/detail fidelity는 낮다. `blend`가 복식/선명도에서 약간 유리하다. |
| `heldout05` | 관모를 쓴 shouting official/reference profile이다. `prev_w14`와 `blend` 모두 black hat official과 shouting cue를 반영한다. `blend`가 얼굴과 pose는 더 깨끗하지만 speech bubble과 profile composition은 유지하지 못한다. |
| `heldout06` | court official headwear/mustache reference다. `blend`는 중앙 얼굴, 관모, robe cue가 `prev_w14`보다 강하지만 표정과 crop은 아직 부정확하다. |
| `heldout07` | 녹색 괴물 머리의 측면 close-up reference다. `no_ip`가 오히려 green demon close-up cue를 보존하고, `prev_w14`/`blend`는 full-body dark demon/assassin 쪽으로 이동한다. metric상 개선이 있어도 구조적으로는 실패 샘플이다. |

## 결론

- Metric: `blend_prev14_c05504`가 c058 평균 기준 best.
- Visual: 일부 identity/color/costume cue는 개선됐지만 high-quality reference-control pass는 아님.
- Failure class: exact pose, panel/crop layout, speech bubble, distinctive props, non-human silhouette.
- Next: runtime weight 추가 sweep보다 `blend_prev14_c05504`를 단일 checkpoint로 distill하거나, c058 failure class를 포함한 continuation/encoder adaptation을 진행해야 한다.
