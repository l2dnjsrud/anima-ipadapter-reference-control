# c089 Shape/Silhouette Distillation Pilot Plan

## 배경

c088 probe는 edge/projection/silhouette metric과 PE embedding에는 hard-shape reference 신호가 일부 있지만, QwenVL/SigLIP2 embedding 자체만으로는 frog/yokai/chibi/non-human profile 실패를 안정적으로 분리하지 못한다는 결론을 냈다. 따라서 c089에서는 broad adapter continuation을 반복하지 않고, 이미 존재하는 PE teacher distillation 경로를 hard-shape trainable subset에 적용한다.

## 선택한 가설

선택 가설은 `SigLIP PE-teacher shape distillation`이다.

- `training/siglip_teacher_smoke.py`는 PE adapter teacher prediction을 같은 noisy latent에서 student SigLIP adapter가 모방하도록 학습할 수 있다.
- `training/siglip_teacher_step.py`는 PE token retrieval과 PE token alignment loss를 이미 지원한다.
- `training/siglip_teacher_cli.py`는 loadable SigLIP checkpoint를 만들 수 있는 command surface다.
- c089에서는 새 대형 architecture를 만들지 않고, c087/c088에서 실패한 hard-shape crop group만 뽑아 PE teacher signal이 실제로 trainable pilot loss로 내려가는지 확인한다.

QwenVL 경로는 현재 PE teacher prediction/token을 직접 받는 학습 step이 없다. 그래서 c089에서 QwenVL 전체 구조를 새로 만들기보다, 이미 구현된 SigLIP teacher 경로로 “teacher signal이 adapter checkpoint에 들어갈 수 있는가”를 먼저 검증한다. 이 pilot이 긍정이면 다음 루프에서 QwenVL 쪽에도 동일 objective를 이식한다.

## 입력 데이터

- source manifest: `training/manifests/c087_expanded_crop_pairs_20260613.jsonl`
- source image root: `.tmp/c087_expanded_crop_pairs_root`
- c089 manifest: `training/manifests/c089_shape_silhouette_distillation_20260613.jsonl`
- c089 summary: `training/manifests/c089_shape_silhouette_distillation_20260613.summary.json`
- c089 scratch root: `.tmp/c089_shape_silhouette_distillation_root`
- selected rows: `64`
- heldout training rows used: `0`
- group balance:
  - `c082_frog_yokai_guard`: `16`
  - `c082_goblin_mage`: `16`
  - `c082_green_oni_scout`: `16`
  - `c082_jade_lizard_monk`: `16`

## Teacher Signals

- `pe_teacher_prediction`: frozen PE adapter teacher denoiser prediction
- `pe_token_retrieval`: SigLIP student tokens retrieve matching PE tokens over wrong PE tokens
- `edge_projection_silhouette_probe`: c088에서 decision 근거가 된 explicit shape metric; c089 manifest/report에는 teacher source label로 남기고, pilot 결과 해석에 사용한다

## 학습 Command Surface

예상 pilot command:

```sh
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_teacher_cli.py \
  --manifest-path training/manifests/c089_shape_silhouette_distillation_20260613.jsonl \
  --image-root .tmp/c089_shape_silhouette_distillation_root \
  --output-path checkpoints/anima_siglip_ip_adapter_c089_shape_pe_teacher_0032_20260613.safetensors \
  --steps 32 \
  --max-rows 32 \
  --resolution 256 \
  --device cuda:0 \
  --teacher-weight 0.7 \
  --pe-retrieval-weight 0.35 \
  --pe-retrieval-margin 0.2 \
  --pe-token-weight 0.15 \
  --pe-token-block-stride 4 \
  --contrastive-weight 0.15 \
  --contrastive-margin 0.05 \
  --pe-kv-init
```

## Stop Gates

- manifest path validation fails: stop and fix data.
- heldout_training_rows_used > 0: stop and do not train.
- training loss is non-finite: stop and report objective failure.
- checkpoint is not loadable or PE checkpoint is not rejected by SigLIP loader: stop and do not promote.
- if pilot produces finite loss/loadable checkpoint, run c089 metrics/report first; generation gate is allowed only if metric report does not show immediate collapse.

## Expected Gate Variants

If c089 proceeds to generation in a later loop, compare:

- `no_ip`
- current best `blend_species_face`
- best QwenVL hard-negative baseline `c086_hard_negative_w14`
- c087 expanded crop-positive
- c089 SigLIP PE-teacher pilot, likely at scales `1.0`, `1.4`

## C001 Decision

c089 C001 decision is `ready_for_c089_siglip_pe_teacher_pilot`. The pilot is bounded, uses no heldout training rows, and exercises an existing loadable checkpoint path instead of adding a large new encoder architecture.
