# c086 Generated Hard-Negative Visual Audit

작성일: 2026-06-13

## 판단

결정: `not_promoted_c086_hard_negative_partial_improvement_not_quality_pass`

c086은 c085 실패 생성물을 explicit negative로 사용한 hard-negative continuation이다. ComfyUI native QwenVL IP-Adapter 노드에서 정상 로드되고 200개 이미지를 생성했으며, blank image는 없었다. 하지만 high-quality reference-control checkpoint로 승격하기에는 아직 부족하다.

## 시각 확인

- `contact_sheet_heldout.jpg`: c086은 c085보다 reference 색, 측면 실루엣, 일부 피부/얼굴 cue를 더 강하게 반영한다.
- heldout00: c086은 reference의 긴 머리와 측면 구도를 유지하지만, 기존 `blend_species_face`와 비교해 우위가 뚜렷하지 않다.
- heldout01: reference의 노년 얼굴/말풍선 context는 여전히 약하고, c086도 젊은 분노 얼굴로 이동한다.
- heldout05/heldout06: 관모/수염/관복 신호는 보존되지만 표정과 얼굴 구조가 과장되어 reference identity가 안정적이라고 보기 어렵다.
- heldout07: c086은 c085보다 초록 비인간 reference를 더 직접적으로 끌어오지만, 최종 이미지는 사람형 캐릭터 옆에 초록 얼굴 cue를 붙이는 방식에 가깝다.
- `contact_sheet_crop_pair_focus.jpg`: c086은 색과 비인간 cue를 강하게 밀지만, 작은 체형, mascot silhouette, 단순 headwear, side-profile 정보를 안정적으로 유지하지 못한다.

## 수치 확인

- clean32+heldout8 PE: `blend_species_face` uplift `0.0608932152`, c085 `0.0308974788`, c086 `0.0595780790`
- clean32+heldout8 QwenVL: `blend_species_face` uplift `0.0421902567`, c085 `0.0306017444`, c086 `0.0388126254`
- heldout8 PE: `blend_species_face` uplift `0.0535339788`, c085 `0.0370979905`, c086 `0.0648405477`
- heldout8 QwenVL: `blend_species_face` uplift `0.0264708772`, c085 `0.0192572623`, c086 `0.0325848311`
- crop-pair focus PE: `blend_species_face` uplift `-0.0916786253`, c085 `-0.0143372715`, c086 `-0.0233308047`
- crop-pair focus QwenVL: `blend_species_face` uplift `0.0194013953`, c085 `0.0466495097`, c086 `0.0361481428`

## 결론

c086은 clean32+heldout8에서 c085보다 명확히 개선됐고, heldout8 평균은 PE/QwenVL 모두 `blend_species_face`를 넘었다. 이 점은 generated hard-negative objective가 실패 유형을 일부 교정한다는 증거다.

그러나 전체 clean32+heldout8 평균에서는 `blend_species_face`를 안정적으로 넘지 못했고, crop-pair focus에서는 c085보다 약해졌다. 시각적으로도 reference의 고유 체형과 구조를 유지하기보다는 강한 색/종족 cue를 붙이는 경향이 남아 있다.

따라서 c086은 `runtime pass / partial improvement / quality fail`로 기록한다. 다음 루프는 generated hard-negative를 더 반복하기보다, target-positive pair를 더 믿을 수 있게 만들거나 encoder-side feature adaptation으로 reference identity embedding 자체를 개선해야 한다.
