# c084 Balanced Crop-Pair Visual Audit

작성일: 2026-06-13

## 판단

결정: `not_promoted_c084_balanced_crop_pair_weaker_than_blend`

c084는 ComfyUI QwenVL IP-Adapter 노드에서 정상 로드되고 150개 생성 결과를 만들었다. blank 이미지는 없고 API 실행도 통과했다. 그러나 고퀄 reference-control checkpoint로는 승격하지 않는다.

## 시각 확인

- `contact_sheet_train.jpg`: c084는 green/non-human 방향을 유지하지만, 기존 `blend_species_face`보다 얼굴 구조와 의상 디테일이 뚜렷하게 좋아지지 않는다.
- `contact_sheet_heldout.jpg`: c084와 blend의 차이가 작고, heldout reference의 고유한 silhouette/장식/나이/체형을 안정적으로 가져오지 못한다.
- `contact_sheet_crop_pair_focus.jpg`: QwenVL metric은 c084가 조금 높지만, frog/chibi/yokai reference가 성인형 green humanoid로 변형된다. reference crop의 headwear, 작은 체형, 의상 색, side-profile 정보가 충분히 유지되지 않는다.

## 수치 확인

- clean32+heldout8 PE: `blend_species_face` uplift `0.0608932152`, c084 uplift `0.0272349045`
- clean32+heldout8 QwenVL: `blend_species_face` uplift `0.0421902567`, c084 uplift `0.0336157218`
- crop-pair focus PE: `blend_species_face` uplift `-0.0046628684`, c084 uplift `-0.0372054994`
- crop-pair focus QwenVL: `blend_species_face` uplift `0.0186477482`, c084 uplift `0.0302898765`

## 결론

c084는 crop focus에서 QwenVL embedding상 약간의 신호를 얻었지만, 사람이 보는 reference-control 품질은 부족하다. 특히 작은 비인간 캐릭터, frog/yokai 형태, headwear, 복장 silhouette을 보존하지 못한다.

다음 루프는 같은 crop-pair calibrator-only 반복이 아니라, adapter-side trainable 범위를 넓히거나 stronger encoder/objective를 적용하는 방향으로 이동한다.
