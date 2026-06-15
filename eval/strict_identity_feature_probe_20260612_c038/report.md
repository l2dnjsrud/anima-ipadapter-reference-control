# Strict Panel Feature Probe c038

작성일: 2026-06-12

## 질문

c037에서 pooled feature가 same-SG weak identity proxy를 충분히 분리하지 못했다. 그렇다면 encoder feature 자체가 무력한 것인지, 아니면 c037 proxy label이 너무 약한 것인지 확인한다.

## 설정

- Dataset root: `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`
- Manifest: `eval/strict_identity_feature_probe_20260612_c038/strict_panel_pair_probe_manifest.jsonl`
- Positive: 같은 panel key의 v4/v5 duplicate crop
- Negative: 같은 `SG-*` 폴더 안의 다른 panel key
- Pair count: 128 total, 64 positive, 64 negative

이 probe는 진짜 캐릭터 identity benchmark가 아니라 feature pipeline sanity control이다. 통과하더라도 “캐릭터 정체성 제어가 해결됐다”는 뜻은 아니다.

## Pooled Feature 결과

| encoder | positive mean | negative mean | margin | pairwise AUC | midpoint accuracy | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `Qwen/Qwen3-VL-Embedding-2B` | 0.999170 | 0.793074 | +0.206096 | 1.000000 | 0.968750 | `feature_separates_proxy_pairs` |
| `google/siglip2-base-patch16-512` | 0.999557 | 0.893771 | +0.105785 | 1.000000 | 0.968750 | `feature_separates_proxy_pairs` |
| `pe` | 0.998897 | 0.858507 | +0.140389 | 0.999756 | 0.914063 | `feature_separates_proxy_pairs` |

## SigLIP Token Feature 결과

`google/siglip2-base-patch16-512`, top-k `64` 기준:

| layer | metric | positive mean | negative mean | margin | pairwise AUC | midpoint accuracy | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `-1` | `pooled` | 0.999446 | 0.865676 | +0.133770 | 1.000000 | 0.968750 | `token_feature_separates_pairs` |
| `-1` | `mean_max_token` | 0.994342 | 0.677364 | +0.316978 | 1.000000 | 1.000000 | `token_feature_separates_pairs` |
| `-1` | `topk_token` | 0.999956 | 0.998010 | +0.001946 | 0.991699 | 0.968750 | `token_feature_not_sufficiently_separated` |
| `-6` | `pooled` | 0.998710 | 0.524800 | +0.473910 | 0.999756 | 0.867188 | `token_feature_separates_pairs` |
| `-6` | `mean_max_token` | 0.995232 | 0.715541 | +0.279691 | 1.000000 | 1.000000 | `token_feature_separates_pairs` |
| `-6` | `topk_token` | 0.999892 | 0.994004 | +0.005889 | 0.997314 | 0.929688 | `token_feature_not_sufficiently_separated` |

## 해석

c038은 세 encoder 모두 near-duplicate panel pair를 강하게 분리한다. 따라서 c037 실패는 feature extraction pipeline 자체가 망가져서가 아니다. c037의 same-SG positive proxy가 캐릭터 identity label로 너무 약했고, pooled feature가 broad visual similarity와 duplicate detection에는 민감하지만 캐릭터 정체성을 자동으로 보장하지 않는다는 해석이 더 맞다.

SigLIP token metric 중 layer `-6` pooled와 layer `-1`/`-6` `mean_max_token`은 strict duplicate control에서 큰 margin을 냈다. 하지만 이것도 duplicate crop detection 결과이지, true same-character generalization 결과가 아니다.

## 결정

결정: `strict_duplicate_feature_sanity_pass_identity_unsolved`

다음 학습은 pooled cosine을 그대로 쓰는 장기 IP-Adapter 학습이 아니라, true same-character pair mining/labeling과 token/layer feature probe로 진행한다.

## 다음 루프

1. duplicate crop은 positive에서 제외하고, 같은 캐릭터의 다른 컷/다른 포즈만 positive로 쓰는 manifest를 만든다.
2. 같은 `SG-*`, 같은 장면, 비슷한 의상/배경 안의 다른 캐릭터를 hard negative로 둔다.
3. SigLIP `mean_max_token` 같은 token-level metric을 후보로 유지하되, true identity manifest에서 다시 검증한다.
4. 통과하면 작은 metric head/calibrator를 먼저 학습하고, 그 다음 IP-Adapter K/V 학습으로 연결한다.
