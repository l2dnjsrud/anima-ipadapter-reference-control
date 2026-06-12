# Reviewed Seed Feature Probe c042

작성일: 2026-06-12

## 질문

c041 reviewed identity seed는 작지만 사람이 라벨을 확인한 pair다. 이 seed에서 SigLIP/QwenVL/PE feature가 same-character positive와 different-character negative를 분리하는지 확인한다.

## 입력

- reviewed seed: `eval/reviewed_identity_candidates_20260612_c041/reviewed_candidate_pairs.jsonl`
- pair probe manifest: `eval/reviewed_seed_feature_probe_20260612_c042/pair_probe_manifest.jsonl`
- positive rows: 4 (`positive_usable=true`)
- negative rows: 3 (`review_label=different_character`)
- excluded rows: same-but-noisy 2, unclear 5

## 산출물

- 변환 도구: `tools/build_reviewed_pair_probe_manifest.py`
- 변환 테스트: `tests/test_reviewed_pair_probe_manifest.py`
- SigLIP pooled: `eval/reviewed_seed_feature_probe_20260612_c042/siglip_pooled_report.md`
- SigLIP layer -6 token: `eval/reviewed_seed_feature_probe_20260612_c042/siglip_layer_m6_token_report.md`
- QwenVL pooled: `eval/reviewed_seed_feature_probe_20260612_c042/qwenvl_pooled_report.md`
- PE pooled: `eval/reviewed_seed_feature_probe_20260612_c042/pe_pooled_report.md`

## 결과

| feature | positive mean | negative mean | margin | AUC | midpoint acc | decision |
|---|---:|---:|---:|---:|---:|---|
| SigLIP pooled | 0.920388 | 0.917526 | 0.002862 | 0.416667 | 0.571429 | fail |
| QwenVL pooled | 0.872421 | 0.848406 | 0.024015 | 0.666667 | 0.714286 | fail |
| PE pooled | 0.856988 | 0.914968 | -0.057980 | 0.416667 | 0.285714 | fail |
| SigLIP layer -6 pooled | 0.514571 | 0.880698 | -0.366127 | 0.083333 | 0.142857 | fail |
| SigLIP layer -6 mean_max_token | 0.776755 | 0.733530 | 0.043225 | 0.916667 | 0.714286 | underpowered |
| SigLIP layer -6 topk_token | 0.994644 | 0.994293 | 0.000351 | 0.750000 | 0.714286 | fail |

기존 pass 기준은 margin `>= 0.05` and AUC `>= 0.70`이다. 이 기준을 만족한 feature는 없다.

## 해석

- QwenVL pooled는 pooled feature 중 가장 나은 방향성을 보였지만 margin과 AUC가 모두 부족하다.
- SigLIP layer -6 `mean_max_token`은 AUC `0.916667`로 흥미로운 후보지만 margin이 `0.043225`로 기준보다 낮고, 표본이 4 positive / 3 negative라 과신하면 안 된다.
- PE pooled는 이 seed에서 negative가 positive보다 더 높아 identity gate 후보로 부적합하다.
- SigLIP pooled는 거의 분리하지 못했다.

## 결정

결정: `reviewed_seed_feature_gate_not_passed`

c041 seed는 feature sanity probe로는 유용했지만, 이 결과만으로 IP-Adapter 학습이나 metric-head 학습을 시작하지 않는다. 다만 SigLIP layer -6 `mean_max_token`은 다음 더 큰 reviewed identity set에서 다시 확인할 우선 후보로 유지한다.

## 다음 루프

1. same-page 14개 후보를 넘어 mining 범위를 넓힌다.
2. QwenVL character-centered filter에 face/upper-body 조건을 더 강하게 넣는다.
3. target이 작은 배경 인물이거나 torso-only인 pair를 positive 후보에서 제외한다.
4. usable positive가 최소 수십 개가 되면 SigLIP layer -6 `mean_max_token`, QwenVL pooled, SigLIP pooled, PE pooled를 다시 비교한다.
5. 그때도 raw feature가 gate를 통과하지 못하면 metric head/calibrator 학습으로 넘어간다.
