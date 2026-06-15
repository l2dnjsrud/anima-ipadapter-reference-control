# c087 Expanded Crop-Positive Full-Adapter Training

## Goal

c086 generated hard-negative model improved some heldout cases but regressed crop-focus identity. c087 tests the opposite pressure: use substantially more approved target-positive crop-pair supervision from c083, then train a full QwenVL adapter continuation.

## Data

- Expanded crop-pair manifest: `training/manifests/c087_expanded_crop_pairs_20260613.jsonl`
- Anchored training manifest: `training/manifests/c087_expanded_anchored_full_adapter_20260613.jsonl`
- Image root: `.tmp/c087_expanded_crop_positive_root`
- Expanded crop rows: 224
- Clean anchor rows: 32
- c052 positive anchor rows: 16
- Failure anchor rows: 32
- Total rows: 304
- Heldout rows used: 0

The target was around 320 crop rows, but the approved c083 source-pair structure yielded 224 selected rows after per-group and per-source-pair balancing. This is still 2.8x the 80 crop rows used by c084/c085.

## Training

```bash
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/qwenvl_contrastive_cli.py \
  --manifest-path training/manifests/c087_expanded_anchored_full_adapter_20260613.jsonl \
  --image-root .tmp/c087_expanded_crop_positive_root \
  --init-checkpoint-path checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors \
  --output-path checkpoints/anima_qwenvl_ip_adapter_c087_expanded_crop_positive_b128_0128_20260613.safetensors \
  --steps 128 \
  --resolution 256 \
  --device cuda:0 \
  --max-rows 304 \
  --lr 2e-6 \
  --contrastive-weight 0.25 \
  --contrastive-margin 0.05 \
  --retrieval-weight 0.15 \
  --retrieval-margin 0.25 \
  --calibrator-bottleneck-dim 128
```

## Result

- Steps: 128
- Rows loaded: 304
- First loss: 0.1127584055
- Final loss: 0.1393356025
- Mean loss: 0.2089717955
- Finite loss: true
- Trainable parameters: 308,176,540
- Checkpoint: `checkpoints/anima_qwenvl_ip_adapter_c087_expanded_crop_positive_b128_0128_20260613.safetensors`
- Checkpoint loadable: true
- PE checkpoint rejected: true

## Verification

```bash
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python -m py_compile \
  tools/c087_expanded_crop_positive_manifest.py \
  tools/c085_anchored_full_adapter_manifest.py \
  tests/test_c087_expanded_crop_positive_manifest.py \
  tests/test_c085_anchored_full_adapter_manifest.py
```

```bash
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python -m pytest \
  tests/test_c087_expanded_crop_positive_manifest.py \
  tests/test_c085_anchored_full_adapter_manifest.py \
  tests/test_c084_balanced_crop_pair_manifest.py -q
```

Result: 7 passed.

## Decision

Training is valid and ready for the c087 ComfyUI generation gate. Promotion is not decided at this stage; it depends on clean32+heldout8 and crop-focus generation metrics plus visual audit against blend, c085, and c086.
