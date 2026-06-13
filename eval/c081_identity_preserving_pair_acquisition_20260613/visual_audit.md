# c081 Identity-Preserving Pair Acquisition Visual Audit

결정: `more_identity_pairs_required`

## 본 것

c081은 6개 direct-green/non-human identity group마다 4개 view를 생성했다. 결과는 24장 모두 생성됐고 blank image는 없었다. 다만 대부분의 이미지는 한 장 안에 여러 포즈가 들어간 character sheet 또는 turnaround sheet 형태로 나왔다.

이 결과는 reference 이미지로는 유용한 proxy가 될 수 있지만, IP-Adapter paired training의 target image로 바로 쓰기에는 위험하다. target image가 multi-pose sheet이면 adapter가 단일 캐릭터 reference-control이 아니라 character sheet layout을 학습할 수 있다.

## 그룹별 판단

- `c081_green_oni_scout`: identity와 팔레트는 비교적 안정적이지만 front/profile/action 모두 여러 포즈가 한 장에 들어간다.
- `c081_jade_lizard_monk`: lizard/tail/robe identity는 좋지만 여러 turn-around가 섞인다.
- `c081_goblin_mage`: 가장 안정적인 identity group이지만 모든 view가 multi-pose sheet다.
- `c081_frog_yokai_guard`: 일부 single-looking image가 있으나 three-quarter/action은 반복 캐릭터 sheet이고, group 전체를 paired target으로 승인하기 어렵다.
- `c081_plant_dryad`: green plant identity는 좋지만 sheet layout이 강하다.
- `c081_serpent_dancer`: view별 스타일과 identity가 흔들리며, 일부는 인간형 일러스트로 바뀐다.

## 결론

manual label은 24장 모두 `useful_proxy_non_human`으로 남겼고 `target_positive`로 승격하지 않았다. 따라서 approved identity group은 `0`, approved pair rows는 `0`, direct self pair rows는 `0`이다.

다음 루프는 같은 identity prompt를 유지하되 `character design sheet`, `turnaround`, `multiple poses`, `reference sheet`를 negative에 넣고, prompt에도 `exactly one character, one pose, single image, no sheet`를 강하게 넣어야 한다.
