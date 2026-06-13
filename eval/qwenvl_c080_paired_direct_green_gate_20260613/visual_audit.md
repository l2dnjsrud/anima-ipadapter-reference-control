# c080 Visual Audit

결정: `not_promoted_c080_paired_direct_green_weaker_than_c079_and_blend`

## 본 것

`contact_sheet_direct_green.jpg` 기준으로 c080은 c079와 거의 같은 방향으로 수렴한다. 녹색 피부, 긴 머리, 뿔/귀, 어두운 무협풍 복식 같은 큰 속성은 일부 유지된다. 하지만 reference column의 핵심인 밝은 색감, 여성형 얼굴, 작은 체형, 과장된 장식, 귀여운 표정, 캐릭터별 다른 실루엣은 대부분 사라진다.

특히 direct_green00-09 모두에서 c080은 source reference의 다양한 캐릭터성을 그대로 전달하기보다 성인형 green humanoid villain 템플릿으로 모인다. c079와 비교해도 눈에 띄는 개선은 없고, 일부 행에서는 c079가 더 선명하게 reference와 가까운 형태를 유지한다.

`contact_sheet_heldout.jpg`에서는 c080이 c075/c079와 거의 같은 이미지를 만든다. reference가 이미 무협풍 남녀 인물일 때는 큰 붕괴 없이 비슷한 결과를 내지만, 이것은 c080의 paired direct-green 학습이 일반 reference-control을 개선했다는 뜻은 아니다. 핵심 실패인 비인간 side-profile, monster face, 특수 실루엣은 여전히 인간형 dark villain으로 흡수된다.

## 숫자와 시각 판단

- clean32+heldout8 PE mean uplift: blend `0.0608932152`, c075 `0.0262199253`, c079 `0.0329968661`, c080 `0.0229417309`
- clean32+heldout8 QwenVL mean uplift: blend `0.0421902567`, c075 `0.0349742755`, c079 `0.0338256791`, c080 `0.0341175169`
- direct-green PE mean uplift: blend `0.0844765946`, c075 `0.0325176731`, c079 `0.0640649319`, c080 `0.0482934043`
- direct-green QwenVL mean uplift: blend `-0.0102016628`, c075 `-0.0095478654`, c079 `0.0040470958`, c080 `-0.0087357759`

숫자와 시각 판단이 같은 방향이다. c080은 QwenVL clean aggregate에서 c079와 사실상 동률에 가깝지만, direct-green 집중 평가에서는 c079보다 낮고 `blend_species_face`보다 약하다.

## 결론

c080은 runtime pass다. 모델 selector에서 로드됐고, clean32+heldout8 200장과 direct-green focus 50장을 생성했으며 blank output은 0장이다.

품질은 pass가 아니다. paired direct-green identity supervision은 이 작은 c074 pair set만으로는 참조별 identity를 분리해 전달하지 못했다. 다음 실험은 같은 c074/c080 구조 반복이 아니라 더 많은 실제 paired data, stronger synthetic pair generation with identity preservation, 또는 QwenVL/SigLIP encoder-side objective로 넘어가야 한다.
