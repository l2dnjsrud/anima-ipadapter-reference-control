# c085 Anchored Full-Adapter Visual Audit

작성일: 2026-06-13

## 판단

결정: `not_promoted_c085_full_adapter_weaker_than_blend`

c085는 QwenVL native ComfyUI IP-Adapter 노드에서 정상 로드되고, clean32+heldout8 및 crop-pair focus에서 총 200개 이미지를 만들었다. 생성 안정성은 통과했지만, 고퀄 reference-control checkpoint로는 승격하지 않는다.

## 시각 확인

- `contact_sheet_heldout.jpg`: c085는 `no_ip`보다는 reference 색/분위기를 끌어오지만, 대부분의 hard case에서 `blend_species_face`보다 명확히 낫지 않다.
- heldout01: reference는 노년 얼굴과 말풍선이 중요한데, c085는 젊은 분노 얼굴로 바뀌고 노년 얼굴 구조를 유지하지 못한다.
- heldout05/heldout06: 수염/관모/관복 신호는 일부 오지만, 기존 blend 대비 얼굴 identity가 더 안정적이라고 보기 어렵다.
- heldout07: reference는 비인간 초록 측면 얼굴인데, c085는 여전히 사람형 장발 악역 남캐로 수렴한다.
- `contact_sheet_crop_pair_focus.jpg`: c085는 초록 피부와 강한 얼굴 대비를 유지하지만, frog/yokai/chibi의 작은 체형, headwear, 단순한 mascot silhouette, side-profile 정보가 무협 adult humanoid template로 바뀐다.

## 수치 확인

- clean32+heldout8 PE: `blend_species_face` uplift `0.0608932152`, c084 `0.0272349045`, c085 `0.0308974788`
- clean32+heldout8 QwenVL: `blend_species_face` uplift `0.0421902567`, c084 `0.0336157218`, c085 `0.0306017444`
- heldout8 PE: `blend_species_face` uplift `0.0535339788`, c084 `-0.0063817352`, c085 `0.0370979905`
- heldout8 QwenVL: `blend_species_face` uplift `0.0264708772`, c084 `0.0235493183`, c085 `0.0192572623`
- crop-pair focus PE: `blend_species_face` uplift `0.1395019785`, c084 `0.1503805324`, c085 `0.1288137928`
- crop-pair focus QwenVL: `blend_species_face` uplift `0.0649063945`, c084 `0.0639641464`, c085 `0.0652211845`

## 결론

c085는 adapter-side trainable 범위를 넓히면 c084보다 일부 PE/heldout 신호가 회복된다는 점을 보여줬다. 그러나 QwenVL heldout 평균과 실제 시각 품질은 기대보다 낮다. 특히 reference의 체형, 측면 silhouette, 나이, mascot-like 단순 형태를 보존하지 못한다.

다음 루프에서는 full adapter continuation을 반복하기보다, hard-negative 생성 결과와 reference crop을 직접 대조하는 objective 또는 encoder-side feature adaptation을 설계해야 한다.
