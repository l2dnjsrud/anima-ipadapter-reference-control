# c087 Visual Audit

작성일: 2026-06-13

## 판정

결정: `not_promoted_c087_expanded_crop_positive_not_quality_pass`

c087은 ComfyUI에서 정상 로드되고 250장 모두 nonblank로 생성되었다. 하지만 reference-control 품질 기준에서는 승격하지 않는다.

## Heldout Sheet

`contact_sheet_heldout.jpg` 기준으로 c087은 색감과 무협풍 인물 안정성은 유지한다. 그러나 고유 얼굴 구조, 소품, speech-bubble context, side-profile silhouette을 안정적으로 보존하지 못한다.

- `heldout00`: 검은 의상과 긴 머리 cue는 유지하지만 reference의 측면 얼굴과 장면 crop은 약하다.
- `heldout01`: old-face reference가 젊은 shouting male template로 바뀐다. speech bubble과 주름/나이 cue가 충분히 잠기지 않는다.
- `heldout02`: bald old monk/beard cue는 어느 정도 반영되지만 c086보다 뚜렷한 개선은 아니다.
- `heldout05`: bearded official identity가 유지되기보다 일반 angry official template로 흐른다.
- `heldout07`: 가장 중요한 비인간 side-profile monster reference에서 c087도 인간형 악역 + 초록 배경/피부 cue로 무너진다. c086보다 명확히 낫지 않다.

## Crop-Focus Sheet

`contact_sheet_crop_pair_focus.jpg`에서는 문제가 더 분명하다. reference는 frog/yokai/chibi mascot 계열의 작은 체형, 둥근 몸, 큰 눈, 모자, 짧은 비율이 핵심인데 c087은 대부분 adult green humanoid head/body template로 변환한다.

c087은 확장 target-positive crop 224개를 학습했지만, 그 신호가 silhouette/비율/모자/mascot body를 잠그는 데 충분하지 않았다. 색상과 비인간 cue는 배웠지만, shape-level identity preservation은 부족하다.

## Metric Interpretation

clean32+heldout8 전체 평균:

- PE: c087 `0.0311193079`, blend `0.0608932152`, c086 `0.0595780790`
- QwenVL: c087 `0.0310260832`, blend `0.0421902567`, c086 `0.0388126254`

heldout8:

- PE: c087 `0.0602737069`, blend `0.0535339788`, c086 `0.0648405477`
- QwenVL: c087 `0.0139362663`, blend `0.0264708772`, c086 `0.0325848311`

crop-focus:

- PE: c087 `0.0536489964`, c085 `0.0815944552`, c086 `0.0940596938`
- QwenVL: c087 `0.0043545485`, c085 `0.0222077310`, c086 `0.0023720264`

PE heldout만 보면 c087은 blend보다 높고 improved rate도 `1.0`이다. 하지만 QwenVL heldout에서는 c087이 가장 낮고, 전체 clean 평균과 crop-focus에서도 기존 baseline을 넘지 못한다. 시각 결과도 metric의 부정적 판단과 일치한다.

## Next Decision

expanded positive crop supervision만 늘리는 방식은 reference-control gate를 통과하지 못했다. 다음 루프는 같은 full-adapter continuation을 반복하기보다, encoder-side checkpoint adaptation 또는 shape/silhouette을 직접 보상하는 feature objective로 이동해야 한다.
