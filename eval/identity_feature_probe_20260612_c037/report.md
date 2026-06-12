# Identity Feature Probe c037

작성일: 2026-06-12

## 목적

c035 audit에서 `stronger_encoder` route가 16/32로 가장 큰 실패 축이었다. 장기 학습으로 들어가기 전에 PE, Qwen3-VL, SigLIP2 pooled image feature가 약한 identity-positive/negative pair를 실제로 분리하는지 확인했다.

이 probe는 완성 identity benchmark가 아니다. 현재 데이터셋의 같은 SG 폴더를 positive proxy로, 다음 SG 폴더를 negative proxy로 쓰는 약한 선별 게이트다. 여기서도 분리하지 못하면 해당 pooled feature를 primary identity loss나 pass/fail gate로 올리지 않는다.

## 입력

- Dataset root: `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`
- Manifest: `eval/identity_feature_probe_20260612_c037/identity_pair_probe_manifest.jsonl`
- Groups: `269`
- Usable groups: `235`
- Pairs: `128`
- Positive pairs: `64`
- Negative pairs: `64`

## 결과

| encoder | positive mean | negative mean | margin | pairwise AUC | midpoint accuracy | decision |
|---|---:|---:|---:|---:|---:|---|
| PE | 0.855997 | 0.840406 | 0.015591 | 0.580566 | 0.578125 | `feature_not_sufficiently_separated` |
| Qwen/Qwen3-VL-Embedding-2B | 0.789310 | 0.756705 | 0.032605 | 0.591309 | 0.570312 | `feature_not_sufficiently_separated` |
| google/siglip2-base-patch16-512 | 0.893205 | 0.880010 | 0.013195 | 0.575928 | 0.562500 | `feature_not_sufficiently_separated` |

통과 기준은 margin `>= 0.05` 그리고 pairwise AUC `>= 0.70`이다. 세 encoder 모두 실패했다.

## 판단

결정: `pooled_identity_feature_not_ready`

QwenVL pooled feature가 PE/SigLIP2보다 margin은 높지만 AUC가 0.60 미만이라 identity encoder 후보로 승격할 수 없다. PE와 SigLIP2도 positive/negative 평균 차이가 너무 작다. 따라서 다음 루프에서는 pooled cosine을 직접 reward/loss로 쓰지 않는다.

다음 모델 작업은 아래 중 하나로 좁힌다.

1. 같은 캐릭터로 더 엄격하게 검증된 positive pair를 먼저 만든다.
2. pooled vector 대신 visual-token feature, shallow/deep layer feature, cross-attention token matching을 따로 probe한다.
3. QwenVL/SigLIP/PE 위에 작은 metric head 또는 calibrator를 학습해 identity-positive/negative separation 자체를 먼저 개선한다.
4. 그 뒤에야 IP-Adapter K/V 주입 또는 reference-control adapter 학습으로 연결한다.

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

## 실행 명령

```bash
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python tools/build_identity_pair_probe_manifest.py \
  /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  eval/identity_feature_probe_20260612_c037/identity_pair_probe_manifest.jsonl \
  --pairs-per-label 64
```

```bash
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python tools/score_identity_pair_probe.py \
  eval/identity_feature_probe_20260612_c037/identity_pair_probe_manifest.jsonl \
  eval/identity_feature_probe_20260612_c037/pe_identity_pair_probe.json \
  --data-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  --encoder pe \
  --device cuda \
  --report-path eval/identity_feature_probe_20260612_c037/pe_identity_pair_probe.md
```

```bash
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python tools/score_identity_pair_probe.py \
  eval/identity_feature_probe_20260612_c037/identity_pair_probe_manifest.jsonl \
  eval/identity_feature_probe_20260612_c037/qwenvl_identity_pair_probe.json \
  --data-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  --encoder qwenvl \
  --report-path eval/identity_feature_probe_20260612_c037/qwenvl_identity_pair_probe.md
```

```bash
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python tools/score_identity_pair_probe.py \
  eval/identity_feature_probe_20260612_c037/identity_pair_probe_manifest.jsonl \
  eval/identity_feature_probe_20260612_c037/siglip_identity_pair_probe.json \
  --data-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  --encoder siglip \
  --device cuda \
  --report-path eval/identity_feature_probe_20260612_c037/siglip_identity_pair_probe.md
```
