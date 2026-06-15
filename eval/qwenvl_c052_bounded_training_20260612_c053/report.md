# QwenVL c052 Bounded Training c053

작성일: 2026-06-12

## 목적

c052에서 QwenVL pooled identity gate가 diverse reviewed seed에서도 안정적으로 통과했기 때문에, 그 positive seed만 사용해 bounded QwenVL adapter continuation을 한 번 실행했다. 이 실험은 생성 품질 통과가 아니라 `학습 surface`, `manifest`, `checkpoint compatibility`가 다음 generation gate로 넘어갈 수 있는지 확인하는 단계다.

## 입력 데이터

- source: `eval/qwenvl_combined_diverse_feature_probe_20260612_c052/usable_positive_pairs.jsonl`
- training manifest: `training/manifests/c052_positive_identity_pairs_20260612.jsonl`
- manifest summary: `training/manifests/c052_positive_identity_pairs_20260612.summary.json`
- positive pairs: `29`
- training rows: `58`
- direction: `bidirectional_anchor_candidate`
- prompt: `mrcolor_panel_style, full color manga panel, clean webtoon coloring, manhwa panel art, character panel, close-up panel, action panel, single panel`

## 실행 명령

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/qwenvl_contrastive_cli.py \
  --manifest-path training/manifests/c052_positive_identity_pairs_20260612.jsonl \
  --image-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  --init-checkpoint-path checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors \
  --output-path checkpoints/anima_qwenvl_ip_adapter_c052_identity_retrieval_0064_20260612.safetensors \
  --steps 64 \
  --resolution 256 \
  --device cuda:0 \
  --max-rows 58 \
  --lr 5e-6 \
  --seed 20260653 \
  --contrastive-weight 0.35 \
  --contrastive-margin 0.05 \
  --retrieval-weight 0.35 \
  --retrieval-margin 0.2 \
  > eval/qwenvl_c052_bounded_training_20260612_c053/train_stdout.txt 2>&1
```

## 결과

| 항목 | 값 |
|---|---:|
| steps | `64` |
| rows_loaded | `58` |
| first_loss | `0.2666643262` |
| final_loss | `0.2033316642` |
| mean_loss | `0.2616678744` |
| mean_base_loss | `0.1721751245` |
| mean_contrastive_loss | `0.0499850863` |
| mean_retrieval_loss | `0.2057084858` |
| finite_loss | `true` |
| trainable_parameters | `308,176,540` |
| frozen_base_parameters | `4,947,838,963` |
| checkpoint_loadable | `true` |
| PE checkpoint rejected by QwenVL loader | `true` |

출력 checkpoint:

```text
checkpoints/anima_qwenvl_ip_adapter_c052_identity_retrieval_0064_20260612.safetensors
```

파일 크기는 `1,232,727,316` bytes다. 단, 레포의 `.gitignore`가 `checkpoints/*qwenvl*.safetensors`를 제외하므로 이 checkpoint는 커밋하지 않는 local artifact로 둔다.

## Import fix

직접 CLI 실행 시 외부 site-package의 `training` 패키지가 로컬 `training/` 디렉터리를 가릴 수 있어, 로컬 패키지를 명시하는 `training/__init__.py`와 회귀 테스트 `tests/test_training_package_imports.py`를 추가했다. 이 수정 덕분에 `PYTHONPATH=.` 조건에서 `training.qwenvl_contrastive_smoke`가 레포 내부 파일로 해석된다.

## 판단

결정: `qwenvl_c052_bounded_training_smoke_passed_generation_gate_pending`

c053은 bounded training과 checkpoint compatibility는 통과했다. 하지만 이것만으로 reference-control 품질을 주장할 수 없다. 다음 단계는 이 checkpoint를 ComfyUI에서 선택 가능하게 노출하고, c035-style single-character generation/contact-sheet gate에서 no-IP baseline과 비교하는 것이다.
