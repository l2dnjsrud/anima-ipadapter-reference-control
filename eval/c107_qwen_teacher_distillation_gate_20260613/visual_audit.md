# C107 Visual Audit

작성일: 2026-06-14 KST

## 확인 산출물

- `contact_sheet_train.jpg`
- `contact_sheet_heldout.jpg`
- 비교 열: `reference` / `no_ip` / `blend_species_face` / `c107_qwen_teacher_w14`

## 육안 판단

C107은 `no_ip` 대비 reference-control이 켜진 것은 맞다. 긴 흑발, 검붉은 의상, 보라색 기운, 붉은 눈, 측면/상반신 포즈 같은 큰 속성은 여러 샘플에서 reference 방향으로 이동했다.

하지만 기존 best인 `blend_species_face`보다 안정적으로 좋지는 않다. 대부분의 행에서 C107은 blend 결과와 매우 유사하거나 약간 다른 변형으로 보이며, 얼굴형/나이/수염/대머리/소품/말풍선 맥락 같은 세부 identity는 blend보다 더 강하게 고정되지 않는다.

Heldout 8개만 보면 QwenVL cosine은 C107이 아주 미세하게 높지만, PE cosine과 전체 육안 안정성은 blend가 우세하다. 특히 `heldout02`, `heldout05`, `heldout06` 같은 나이/수염/관모/대머리 중심 샘플에서 C107은 일부 특징을 잡지만 reference의 정확한 인상까지 승격할 정도는 아니다.

## 결정

결정: `c107_generation_gate_not_promoted`

C107은 학습 가능한 Qwen teacher 신호라는 증거로 남긴다. 그러나 standalone checkpoint로 배포하거나 current best로 교체할 수준은 아니다. 다음 루프는 C107 방식의 teacher score를 단순 hard-negative manifest로만 쓰는 대신, teacher embedding/target 자체를 직접 맞추거나 기존 best blend와 조합 가능한 방식으로 넘어간다.
