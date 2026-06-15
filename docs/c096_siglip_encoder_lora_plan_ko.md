# c096 SigLIP Encoder LoRA 실험 계획

## 목적

c095 feature bridge는 ComfyUI에서 정상 작동했지만 hard-shape reference-control 품질은
c094와 거의 같거나 약했다. 특히 `heldout07`의 non-human side-profile은 여전히 인간형
얼굴로 무너졌다. 따라서 c096은 adapter 뒤쪽 보정을 반복하지 않고, SigLIP vision encoder
자체의 마지막 attention projection에 작은 LoRA를 붙여 reference feature 공간을 움직일 수
있는지 확인한다.

## 가설

SigLIP2 base encoder가 만화/무협/비인간 캐릭터의 실루엣 신호를 충분히 잘 분리하지 못해
adapter가 collapse attractor로 끌린다면, 마지막 attention projection의 low-rank LoRA를
positive teacher target과 explicit negative 사이의 feature margin으로 짧게 학습했을 때
reference feature가 더 유용해질 수 있다.

## 구현 범위

- `siglip_encoder_lora.py`
  - `SiglipVisionModel` 내부 `vision_model.encoder.layers.*.self_attn.{q_proj,v_proj,out_proj}`에
    LoRA wrapper를 적용한다.
  - LoRA만 safetensors로 저장/로드한다.
- `native_siglip.py`
  - `AnimaSigLIPEncodeImage`에 `encoder_lora_name` selector를 추가한다.
  - 기본값은 `none`이며, 기존 workflow/API는 동일하게 동작한다.
- `tools/siglip_auto_caption_types.py`, `tools/siglip_auto_caption_eval.py`
  - `Variant.encoder_lora`를 추가해 같은 IP-Adapter checkpoint에서 기본 encoder와
    LoRA encoder를 나란히 비교할 수 있게 한다.
- `training/siglip_encoder_lora_contrastive.py`
  - frozen SigLIP base + trainable LoRA만 학습한다.
  - loss는 reference-anchor가 positive teacher target에 가까워지고 explicit negative에서
    멀어지는 margin loss다.

## 데이터 경계

- train manifest:
  `training/manifests/c093_siglip_qwen_target_anti_collapse_20260613.jsonl`
- image root:
  `.tmp/c093_anti_collapse_root`
- rows:
  `crop_pair00` - `crop_pair09`
- explicit negatives:
  `c092_qwen_target_w14_negative`
- heldout:
  `heldout07`은 학습에서 제외하고 generation gate에서만 본다.

## 학습 명령

```bash
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
  training/siglip_encoder_lora_contrastive.py \
  --manifest-path training/manifests/c093_siglip_qwen_target_anti_collapse_20260613.jsonl \
  --image-root .tmp/c093_anti_collapse_root \
  --output-path checkpoints/anima_siglip_encoder_lora_c096_rank8_0096_20260613.safetensors \
  --steps 96 \
  --max-rows 10 \
  --device cuda:0 \
  --lr 1e-4 \
  --seed 20260696 \
  --rank 8 \
  --alpha 8.0 \
  --margin 0.08 \
  --layer-count 2
```

## 통과 조건

- `finite_loss = true`
- `rows_loaded = 10`
- `explicit_negative_rows = 10`
- `heldout_rows = []`
- trainable parameter 이름이 모두 `lora_` 경로다.
- LoRA checkpoint가 `verify_siglip_lora` 및 fresh SigLIP model load에 통과한다.
- checkpoint 파일은 local-only ignored artifact로 남긴다.

## 생성 게이트

학습이 통과하면 isolated ComfyUI에서 다음 비교군을 생성한다.

- `no_ip`
- `c094_shape_supervised_w14`
- `c095_feature_bridge_w14`
- `c096_lora_c094_w08`
- `c096_lora_c094_w10`
- `c096_lora_c094_w12`
- `c096_lora_c094_w14`

`c096_lora_*`는 IP-Adapter checkpoint로 c094를 쓰고, `AnimaSigLIPEncodeImage`의
`encoder_lora_name`에 c096 LoRA checkpoint를 선택한다.

## 판정

c096은 다음을 만족해야 다음 단계 후보가 된다.

- c096 평균 uplift가 c094와 c095보다 의미 있게 높다.
- `heldout07`에서 non-human side-profile cue가 c094/c095보다 뚜렷하게 좋아진다.
- c096 blank-like row가 없다.
- contact sheet에서 frog/chibi/mascot/non-human rows가 같은 초록 인간 얼굴로 수렴하지 않는다.

실패하면 `c096_encoder_lora_not_promoted_requires_data_expansion_or_deeper_encoder_training`으로
기록하고, 다음 루프는 더 많은 color/reference-control paired data 또는 더 깊은 encoder
fine-tuning으로 넘어간다.

## 결과

학습 gate는 통과했다.

- steps: `96`
- rows loaded: `10`
- explicit negative rows: `10`
- heldout rows: `[]`
- first loss: `0.0183230527`
- final loss: `0.0074154139`
- mean positive similarity: `0.8003282845`
- mean negative similarity: `0.7492882957`
- checkpoint loadable: `true`

첫 학습 실행은 LoRA wrapper가 base linear의 device/dtype을 따라가지 않아 CUDA device mismatch로 실패했다.
`LoRALinear`가 `base.weight.device`와 `base.weight.dtype`에 맞춰 `lora_down/lora_up`을 만들도록
수정한 뒤 학습은 정상 종료했다.

ComfyUI generation gate에서는 한 번 더 구조 차이 버그가 드러났다. 학습/저장 쪽 LoRA key는
`vision_model.encoder...` 형태였지만, ComfyUI runtime은 `SiglipVisionModel`을 직접 로드해서
실제 모듈 경로가 `encoder...`로 시작했다. `apply_saved_siglip_lora`에서 저장 key는 유지하고
runtime module name만 현재 모델 구조에 맞게 정규화하도록 수정했다.

최종 생성 gate:

- output: `eval/c096_siglip_encoder_lora_generation_gate_20260613/`
- generated images: `77`
- contact sheet: `contact_sheet_hard_shape.jpg`
- C096 blank-like rows: `[]`
- cleanup receipt: `cleanup_port_8122.txt`

수치:

| variant | mean uplift | improved rate |
|---|---:|---:|
| `c094_shape_supervised_w14` | `0.0878832954` | `0.9090909091` |
| `c095_feature_bridge_w14` | `0.0865223347` | `0.9090909091` |
| `c096_lora_c094_w08` | `0.0576397215` | `0.7272727273` |
| `c096_lora_c094_w10` | `0.0786039762` | `0.8181818182` |
| `c096_lora_c094_w12` | `0.0830830928` | `0.9090909091` |
| `c096_lora_c094_w14` | `0.0880849553` | `0.9090909091` |
| `c087_expanded_crop_positive_w14` | `0.1089544056` | `0.9090909091` |

최종 decision은 `c096_encoder_lora_not_promoted_requires_data_expansion_or_deeper_encoder_training`이다.
C096 w14는 C094/C095와 거의 동률이지만 Qwen baseline을 넘지 못했고, `heldout07`에서도
C094/C095보다 낮았다. Contact sheet에서도 frog/chibi/mascot/non-human reference가 여전히
green adult humanoid face template으로 수렴한다. 따라서 이 LoRA는 runtime 기능 증명으로는
성공했지만 고품질 reference-control 후보로 승격하지 않는다.
