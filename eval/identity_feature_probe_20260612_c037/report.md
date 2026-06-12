# Identity Feature Probe c037

작성일: 2026-06-12

## 질문

기존 pooled image feature를 stronger Anima IP-Adapter 학습 신호로 쓰기 전에, 약한 identity-positive pair와 identity-negative pair를 분리할 수 있는지 확인한다.

## 설정

- Dataset root: `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`
- Manifest: `eval/identity_feature_probe_20260612_c037/identity_pair_probe_manifest.jsonl`
- Pair construction: 같은 `SG-*` 폴더 pair는 positive proxy, 다음 `SG-*` 폴더 pair는 negative proxy로 둔다.
- Pair count: 128 total, 64 positive, 64 negative
- Caveat: 이 probe는 검증된 same-character identity label이 아니라 약한 proxy gate다.

## 결과

| encoder | positive mean | negative mean | margin | pairwise AUC | midpoint accuracy | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `Qwen/Qwen3-VL-Embedding-2B` | 0.789310 | 0.756705 | +0.032605 | 0.591309 | 0.570313 | `feature_not_sufficiently_separated` |
| `google/siglip2-base-patch16-512` | 0.893205 | 0.880010 | +0.013195 | 0.575928 | 0.562500 | `feature_not_sufficiently_separated` |
| `pe` | 0.855997 | 0.840406 | +0.015591 | 0.580566 | 0.578125 | `feature_not_sufficiently_separated` |

통과 기준은 margin `>= 0.05` 그리고 pairwise AUC `>= 0.70`이다. 세 encoder 모두 실패했다.

## 해석

QwenVL은 세 pooled feature probe 중 margin이 가장 좋지만, AUC가 `0.591309`로 scorer의 최소 기준 `0.70`보다 크게 낮다. SigLIP2와 PE pooled feature도 positive/negative 평균 차이가 작고 AUC가 0.60 미만이다.

이 결과는 c036 결론을 강화한다. generic VL pooled similarity는 broad style, 색감, 구도 유사도 관찰에는 쓸 수 있지만, FaceID-like identity metric이나 primary reference-control training signal로 쓰기에는 부족하다.

## 결정

결정: `pooled_identity_feature_not_ready`

QwenVL, SigLIP2, PE pooled-image cosine에 주로 의존하는 장기 adapter 학습은 시작하지 않는다.

## 다음 루프

다음 루프는 더 엄격한 identity 데이터와 feature를 먼저 만든다.

1. same-SG proxy가 아니라 true same-character positive pair를 mining하거나 라벨링한다.
2. 같은 작품/장면/스타일 안에서 hard negative를 만든다.
3. pooled embedding 하나가 아니라 token-level, region-level, layer-level feature를 probe한다.
4. 필요하면 QwenVL/SigLIP/PE feature 위에 작은 metric head 또는 calibrator를 먼저 학습한다.
5. 이 gate를 통과한 뒤에 IP-Adapter K/V 주입 또는 reference-control adapter 학습으로 연결한다.

## 산출물

- `tools/build_identity_pair_probe_manifest.py`
- `tools/image_feature_embedders.py`
- `tools/score_identity_pair_probe.py`
- `tests/test_identity_feature_probe.py`
- `eval/identity_feature_probe_20260612_c037/identity_pair_probe_manifest.jsonl`
- `eval/identity_feature_probe_20260612_c037/pe_identity_pair_probe.json`
- `eval/identity_feature_probe_20260612_c037/pe_identity_pair_probe.md`
- `eval/identity_feature_probe_20260612_c037/qwenvl_identity_pair_probe.json`
- `eval/identity_feature_probe_20260612_c037/qwenvl_identity_pair_probe.md`
- `eval/identity_feature_probe_20260612_c037/siglip_identity_pair_probe.json`
- `eval/identity_feature_probe_20260612_c037/siglip_identity_pair_probe.md`
