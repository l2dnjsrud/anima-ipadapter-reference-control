# c075 Visual Audit

결정: `not_promoted`

## 본 것

`contact_sheet_direct_green.jpg`에서 c075는 녹색 피부, 꼬리, 긴 귀 같은 큰 trait는 어느 정도 유지한다. 하지만 reference의 핵심인 네온/귀여운 채색 감각, 머리 장식과 뿔 실루엣, 원본의 캐릭터 얼굴 인상은 대부분 사라지고 무협풍 성인 humanoid로 수렴한다.

`contact_sheet_heldout.jpg`에서도 c075는 current best `blend_species_face`와 비슷하거나 더 약하다. 특히 heldout07의 괴물 side-profile reference는 여전히 사람형 dark villain으로 변하고, 녹색 피부와 비인간 얼굴 구조를 제대로 유지하지 못한다.

## 숫자와 시각 판단

- clean32+heldout8 PE mean uplift: blend `0.0608932152`, c075 `0.0262199253`
- clean32+heldout8 QwenVL mean uplift: blend `0.0421902567`, c075 `0.0349742755`
- direct-green PE mean uplift: blend `0.0379917264`, c075 `-0.0206880599`
- direct-green QwenVL mean uplift: blend `-0.0121086836`, c075 `-0.0143850207`

heldout01은 QwenVL 점수에서 c075가 blend보다 약간 좋아 보이지만, aggregate와 시각 identity는 충분하지 않다. heldout05는 PE/visual 기준에서 blend가 더 강하고, heldout07은 두 방식 모두 실패한다.

## 결론

c075는 runtime pass다. 노드, 모델 selector, API 생성, nonblank output은 모두 통과했다.

하지만 high-quality reference-control checkpoint로는 승격하지 않는다. 현재 학습 신호는 green/non-human tag를 생성 prompt 쪽으로 밀어 넣는 데는 도움이 되지만, reference image의 구체적 시각 identity를 adapter embedding으로 안정적으로 전달하지 못한다.

다음 결정: `do_not_continue_calibrator_only_on_same_signal`

다음 실험은 더 강한 encoder-side adaptation, reference feature supervised objective, 또는 실제 paired direct-green 데이터 구축 쪽으로 이동해야 한다.
