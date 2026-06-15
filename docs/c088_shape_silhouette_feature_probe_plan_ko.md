# c088 Shape/Silhouette Feature Probe 계획

## 목적

c087은 expanded crop-positive supervision을 늘렸지만 crop-focus에서 frog/chibi/non-human silhouette을 잠그지 못했다. 색감과 초록 피부 cue는 따라가지만, 둥근 몸체, 짧은 비율, profile 형태가 adult green humanoid template로 무너졌다.

c088의 목적은 새 adapter continuation을 바로 돌리기 전에, 기존 feature들이 이 형태 붕괴를 수치로 분리할 수 있는지 확인하는 것이다. 분리 신호가 있으면 c089에서 supervised shape objective로 옮기고, 없으면 adapter head 반복을 멈추고 encoder-side checkpoint adaptation을 우선한다.

## 입력

- Source gate: `eval/qwenvl_c087_expanded_crop_positive_gate_20260613`
- Manifest: `eval/c088_shape_silhouette_feature_probe_20260613/probe_manifest.jsonl`
- Summary: `eval/c088_shape_silhouette_feature_probe_20260613/summary.json`
- Rows: `11`
  - crop-focus frog/yokai guard: `10`
  - heldout07 non-human side-profile: `1`
- Heldout training rows used: `0`

비교 variant:

- `no_ip`
- `blend_species_face`
- `c085_anchored_full_adapter_w14`
- `c086_hard_negative_w14`
- `c087_expanded_crop_positive_w14`

## Scoring

1. QwenVL embedding cosine
   - 현재 QwenVL native workflow와 같은 `Qwen/Qwen3-VL-Embedding-2B` 계열 feature를 사용한다.
2. SigLIP2 embedding cosine
   - `google/siglip2-base-patch16-512` pooled vision feature를 사용한다.
3. PE embedding cosine
   - 기존 PE metric과 비교 가능한 PE pooled feature를 사용한다.
4. Edge/projection/silhouette metric
   - reference와 candidate를 같은 해상도로 맞춘 뒤 edge map cosine, edge projection cosine, foreground/silhouette IoU를 합성한다.

## 판정 기준

- 케이스 단위:
  - best variant가 `no_ip`가 아니어야 한다.
  - no-IP 대비 uplift가 충분히 커야 한다.
  - 1위와 2위 margin이 있어야 한다.
- 전체 판정:
  - embedding 세트와 edge/silhouette 점수가 같은 방향으로 reference-like 후보를 고르면 `shape/silhouette supervised objective viable`
  - 세 encoder 또는 shape metric이 no-IP/generic output을 더 좋게 보면 `encoder-side checkpoint adaptation required`

## Stop Gate

c088은 새 이미지를 생성하지 않는다. 기존 c087 generation artifact만 재사용한다. 따라서 출력이 이상하면 학습으로 넘어가지 않고, manifest/path/metric 산출물만 기록한 뒤 원인을 문서화한다.
