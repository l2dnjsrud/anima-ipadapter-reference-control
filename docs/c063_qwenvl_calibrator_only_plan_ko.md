# c063 QwenVL Calibrator-Only 계획

## 목적

c062는 `calibrator_bottleneck_dim=128`을 붙였지만 실제로는 QwenVL adapter 전체가 trainable로 열려 `308,176,540` parameters가 학습되었다. 결과적으로 checkpoint는 active/loadable이었지만 `blend_species_face` runtime preset보다 약했고, `heldout07` 비인간 side-profile 붕괴를 고치지 못했다.

c063의 목적은 같은 broad adapter continuation을 반복하지 않고, QwenVL embedding을 IP-Adapter resampler에 넣기 직전의 feature calibration만 조정하는 것이다. 즉 `feature_calibrator.*`만 trainable로 열어 작은 feature-adaptation이 reference identity cue를 더 안정적으로 보강할 수 있는지 확인한다.

## 가설

QwenVL embedding 자체는 reference의 색감과 일부 costume cue를 어느 정도 담고 있지만, 현재 adapter가 Anima DiT cross-attention으로 넘기는 방향이 고퀄 identity control에 맞지 않는다. 작은 residual calibrator만 학습하면 기존 adapter의 생성 안정성을 덜 건드리면서 비인간 profile, 노인 얼굴 구조, 수염/관모, prop, crop/speech-bubble context 같은 실패 cue를 조금 더 잘 보존할 수 있다.

## 구현 표면

- `training/qwenvl_smoke_checkpoint.py`
  - `train_calibrator_only=True`일 때 모든 adapter parameter를 freeze하고 `feature_calibrator.*`만 trainable로 설정한다.
- `training/qwenvl_contrastive_smoke.py`
  - optimizer는 `requires_grad=True` parameter만 받는다.
  - summary에 `train_calibrator_only`를 기록한다.
- `training/qwenvl_contrastive_cli.py`
  - CLI option `--train-calibrator-only`를 노출한다.

ComfyUI runtime은 별도 변경하지 않는다. 기존 `QwenVL` checkpoint loader가 calibrated checkpoint를 감지하고 `CalibratedIPAdapterQwenVL`로 load할 수 있기 때문이다.

## 데이터

학습 manifest는 `training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.jsonl`을 그대로 사용한다.

- rows: `154`
- heldout rows: 사용 금지
- 목적: c060/c061/c062에서 반복 실패한 identity cue를 포함한 failure-focused train set 재사용

## 학습 명령 초안

```sh
CUDA_VISIBLE_DEVICES=0 /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/qwenvl_contrastive_cli.py \
  --manifest-path training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.jsonl \
  --image-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  --init-checkpoint-path checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors \
  --output-path checkpoints/anima_qwenvl_ip_adapter_c063_calibrator_only_b128_0128_20260612.safetensors \
  --steps 128 \
  --resolution 256 \
  --device cuda:0 \
  --max-rows 154 \
  --lr 8e-5 \
  --seed 20260663 \
  --contrastive-weight 0.35 \
  --contrastive-margin 0.05 \
  --retrieval-weight 0.20 \
  --retrieval-margin 0.2 \
  --calibrator-bottleneck-dim 128 \
  --train-calibrator-only \
  --instruction "Represent this reference for strict visual identity retrieval in a manhwa panel. Prioritize non-human species, monster or demon traits, facial structure, profile silhouette, beard and headwear, skin tone, glowing eyes, hand props, fan or weapon cues, speech bubble context, pose crop, costume palette, and line/color style."
```

## 학습 stop gate

- `finite_loss=true`
- `rows_loaded=154`
- `train_calibrator_only=true`
- `trainable_parameters`가 calibrator-only 규모여야 한다. 예상치는 약 `528,384` parameters다.
- checkpoint가 loadable이어야 한다.
- PE checkpoint reject guard가 true여야 한다.

## 생성 gate

학습이 통과하면 clean32 train `32`장과 heldout `8`장을 같은 API gate로 평가한다.

비교 column:

- `no_ip`
- `blend_species_face`: 현재 최상 runtime preset, previous retrieval `1.4` + c055 `0.4`
- `c063_calibrator_only_w14`: c063 checkpoint `1.4`

판단 기준:

- PE/QwenVL heldout uplift가 `blend_species_face`에 근접하거나 초과하는지 본다.
- 특히 `heldout01`, `heldout05`, `heldout07`을 시각 감사한다.
- `heldout07` 비인간 side-profile이 계속 human dark-villain template으로 붕괴하면 high-quality gate는 실패다.

## 예상 결정

c063이 실패하면 단순 adapter/calibrator checkpoint 학습은 더 진행하지 않고, 별도 단계로 QwenVL/SigLIP encoder-side adaptation 또는 failure-attribute supervised embedding checkpoint를 설계해야 한다.
