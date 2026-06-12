# c079 Visual Audit

결정: `not_promoted_c079_synthetic_positive_calibrator_partial_direct_green_gain`

## 본 것

`contact_sheet_direct_green.jpg` 기준으로 c079는 c075보다 direct-green target 신호가 조금 더 안정적이다. 특히 여러 행에서 녹색 피부, 뿔/귀 실루엣, 어두운 망토형 복식이 유지된다. 다만 reference의 다양성은 충분히 따라오지 못한다. 작은 체형, 밝은 색감, 귀여운 표정, 원본마다 다른 장식과 비인간 얼굴 구조가 대부분 성인형 녹색 humanoid 악역으로 수렴한다.

`contact_sheet_heldout.jpg`에서는 c079가 c075와 거의 같은 구도와 얼굴을 낸다. heldout02, heldout05, heldout06처럼 명확한 복식/수염/관모가 있는 경우에는 current best `blend_species_face`와 비슷하게 보이지만, 전체적으로 더 낫다고 보기 어렵다. 가장 중요한 heldout07 비인간 side-profile monster는 여전히 사람형 dark villain으로 바뀌고, reference의 턱/주둥이/측면 실루엣은 유지되지 않는다.

## 숫자와 시각 판단

- clean32+heldout8 PE mean uplift: blend `0.0608932152`, c075 `0.0262199253`, c079 `0.0329968661`
- clean32+heldout8 QwenVL mean uplift: blend `0.0421902567`, c075 `0.0349742755`, c079 `0.0338256791`
- direct-green PE mean uplift: blend `0.3416856171`, c075 `0.2800143618`, c079 `0.2937260229`
- direct-green QwenVL mean uplift: blend `0.0291833697`, c075 `0.0386207013`, c079 `0.0388706634`

c079는 direct-green QwenVL metric에서 c075보다 아주 작게 높고, PE direct-green에서도 c075보다 높다. 하지만 clean32+heldout aggregate에서는 `blend_species_face`보다 낮고, direct-green PE에서도 `blend_species_face`가 여전히 가장 강하다.

## 결론

c079는 runtime pass다. ComfyUI native QwenVL 노드에서 모델 selector로 로드됐고, clean32+heldout8 160장과 direct-green focus 132장을 생성했으며 blank output은 0장이다.

품질은 아직 high-quality reference-control pass가 아니다. 이번 synthetic-positive 확장은 “녹색/비인간” 속성 신호를 조금 보강했지만, reference image의 구체적 identity와 다양성을 안정적으로 전달하지 못한다.

다음 결정: `do_not_promote_c079_continue_with_more_precise_pair_or_encoder_objective`

다음 실험은 synthetic target-positive를 더 늘리는 단순 반복보다, 실제 paired direct-green 이미지, synthetic source-target pair의 명시적 identity supervision, 또는 QwenVL/SigLIP feature space에서 reference identity를 분리하는 encoder-side objective가 필요하다.
