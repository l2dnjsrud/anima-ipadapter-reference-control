# c059 Visual Audit

평가 대상은 c058의 최선 runtime recipe였던 `blend_prev14_c05504`를 단일 QwenVL adapter checkpoint로 근사할 수 있는지다. 비교 열은 `no_ip`, 이전 retrieval checkpoint `prev_w14`, runtime blend `blend_prev14_c05504`, 그리고 병합 checkpoint `merge_a025_w14`, `merge_a040_w14`다.

## 요약 판단

병합 checkpoint는 ComfyUI에서 정상 로드되고 실제 생성에도 적용된다. `merge_a040_w14`는 QwenVL metric 평균에서 `blend_prev14_c05504`와 거의 동률이며, 일부 사람 캐릭터 샘플에서는 얼굴/복식/색감 cue가 꽤 잘 들어온다.

하지만 최종 reference-control 품질로는 부족하다. PE metric 기준으로는 두 병합 후보 모두 runtime blend보다 낮고, `merge_a040_w14`의 PE improved rate는 `0.475`까지 떨어진다. 시각적으로도 c058의 실패 클래스인 정확한 pose/crop, speech bubble, hand/fan prop, non-human silhouette 문제가 해결되지 않는다.

따라서 c059의 결론은 “단일 checkpoint merge는 실행 가능한 진단 실험이지만, runtime blend를 대체할 완성 해법은 아님”이다.

## Metric Notes

| metric | best relevant result |
| --- | --- |
| PE mean uplift | `blend_prev14_c05504` = `+0.049596`; `merge_a040_w14` = `+0.025837` |
| PE improved rate | `prev_w14` = `0.750`; `blend_prev14_c05504` = `0.725`; `merge_a040_w14` = `0.475` |
| QwenVL mean uplift | `merge_a040_w14` = `+0.041614`; `blend_prev14_c05504` = `+0.041589` |
| QwenVL improved rate | `blend_prev14_c05504`, `merge_a025_w14`, `merge_a040_w14` all `0.800` |

QwenVL metric만 보면 `merge_a040_w14`가 좋아 보이지만, PE metric과 contact sheet를 같이 보면 과신하면 안 된다.

## Heldout Audit

| sample | visual judgment |
| --- | --- |
| `heldout00` | 장발 악역/측후면 cue는 모든 IP variant가 `no_ip`보다 안정적이다. `merge_a040`은 색감과 얼굴 방향은 괜찮지만 fan/측후면 구도는 여전히 약하다. |
| `heldout01` | 분노한 남성 cue는 잘 들어오지만 노년성, 말풍선, 얼굴 윤곽은 보존되지 않는다. `merge_a040`은 profile 쪽으로 가며 원본 노년 남성보다는 젊은 무협 인물에 가깝다. |
| `heldout02` | bald old monk cue는 병합 후보가 비교적 잘 잡는다. `merge_a025`/`merge_a040` 모두 두상과 수염은 좋지만, 원본의 정면 close-up과 피부/표정 fidelity는 아직 다르다. |
| `heldout03` | 붉은 여성/좌식/궁중 분위기는 대체로 반영된다. 다만 speech bubble과 자세/손 동작은 유지되지 않고, 병합 후보가 runtime blend보다 명확히 낫다고 보기 어렵다. |
| `heldout04` | 검은 머리 side-face와 손 gesture는 반영된다. `merge_a040`은 붉은 눈과 profile을 강하게 가져오지만 말풍선, crop, 배경 구조는 실패한다. |
| `heldout05` | 관모와 shouting official cue는 `prev_w14`/blend/merge 모두 들어온다. `merge_a040`은 hat silhouette이 더 강하지만 얼굴이 찌그러지고 profile/speech bubble은 보존하지 못한다. |
| `heldout06` | 관모/수염 cue는 잡지만 원본의 겁먹은 court official 표정과 세로 close-up이 안정적으로 유지되지 않는다. 병합 후보는 더 강한 악역 템플릿으로 밀린다. |
| `heldout07` | 가장 중요한 실패 케이스다. 원본은 초록 괴물 머리의 측면 close-up인데, 모든 IP variant가 full-body dark demon/assassin 쪽으로 무너진다. `merge_a025`/`merge_a040`도 이 구조 실패를 해결하지 못한다. |

## 결론

- Loadability: pass.
- API generation: pass.
- Metric: QwenVL-only로는 `merge_a040_w14`가 runtime blend와 동률급.
- Visual/PE: pass 아님.
- Decision: `single_checkpoint_merge_not_quality_pass_runtime_blend_remains_best`.
- Next: 단순 checkpoint interpolation은 중단하고, failure-focused continuation 또는 encoder/feature adaptation으로 넘어가야 한다.
