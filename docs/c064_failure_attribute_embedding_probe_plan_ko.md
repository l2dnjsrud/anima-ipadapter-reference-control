# c064 Failure-Attribute Embedding Probe 계획

## 목적

c063까지 확인한 사실은 단순 adapter continuation, runtime blend, instruction calibration, calibrator-only 학습이 모두 reference-control 품질 gate를 통과하지 못했다는 것이다. 특히 heldout hard case에서는 색감이나 일부 의상 cue보다 더 중요한 구조적 속성이 무너졌다.

c064의 목적은 새 학습을 바로 시작하기 전에, 현재 후보 feature space 자체가 실패 속성을 분리할 수 있는지 확인하는 것이다. QwenVL, SigLIP2, PE embedding에서 reference와 생성 결과의 거리/순위가 hard attribute를 반영하지 못한다면 adapter head만 더 학습하는 것은 반복 실패일 가능성이 높다.

## 입력 아티팩트

기준 gate는 c063 clean32+heldout8 ComfyUI API 결과를 사용한다.

- 기준 summary: `eval/qwenvl_c063_calibrator_only_gate_20260612/summary.json`
- 비교 결과:
  - `no_ip`
  - `blend_species_face`: 현재 최상 runtime preset, previous retrieval `1.4` + c055 `0.4`
  - `c063_calibrator_only_w14`: c063 calibrator-only checkpoint `1.4`
- source reference root: `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`

## hard failure rows

- `heldout01`: side profile/young warrior prompt로 보이지만 c063 시각 감사에서 old-face/crop/speech-bubble context 같은 reference 고유 cue가 약하게 반영된 케이스다.
- `heldout05`: old bearded official, black official hat, upper-body crop, beard/headwear identity가 핵심인데 c063이 PE/시각 기준에서 더 악화된 케이스다.
- `heldout07`: green monster face, red glowing eye, non-human side-profile이 핵심인데 blend와 c063 모두 human dark-villain template으로 붕괴한 케이스다.

입력 manifest:

- `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/probe_manifest.jsonl`

각 row는 reference path와 `no_ip`, `blend_species_face`, `c063_calibrator_only_w14` 생성 이미지를 모두 포함한다. 이 manifest는 학습용이 아니라 오프라인 probe 입력이며, heldout 이미지를 train에 섞지 않는다.

## feature encoders

- QwenVL: `Qwen/Qwen3-VL-Embedding-2B`, 현재 QwenVL native workflow와 같은 image embedding 계열
- SigLIP2: `google/siglip2-base-patch16-512`, 이전 SigLIP 실험의 strongest open encoder 후보
- PE: local PE encoder, 기존 PE-Core 기준과 비교 가능한 pooled feature

## scoring

각 encoder마다 다음 값을 기록한다.

- reference와 `no_ip`, `blend_species_face`, `c063_calibrator_only_w14` 사이 cosine
- `uplift = cosine(candidate, reference) - cosine(no_ip, reference)`
- sample별 candidate rank
- `blend_species_face` 대비 c063 delta
- hard failure별 binary decision:
  - `encoder_space_supports_supervised_signal`: reference similarity가 no_ip보다 좋아지고, 생성 후보 사이 rank가 시각 감사와 대체로 일치하는 경우
  - `encoder_space_not_enough`: reference와 실패 template 사이의 순위/거리 차이가 작거나, 시각적으로 실패한 후보를 embedding이 높게 평가하는 경우

## stop gate

Probe는 오프라인 CLI 실험으로 끝나야 한다.

- `summary.json`, `report.md`, encoder별 metric JSON이 모두 생성되어야 한다.
- `heldout01`, `heldout05`, `heldout07` 각각에 대해 QwenVL/SigLIP/PE 판정을 기록해야 한다.
- ComfyUI runtime은 사용하지 않는다.
- 생성 이미지가 이미 존재하는 c063 artifact를 사용하므로 새 PNG 생성은 하지 않는다.

## 다음 결정

- 세 encoder 중 하나라도 hard failure를 안정적으로 분리하면, c065는 그 encoder feature를 teacher로 쓰는 작은 supervised embedding/calibrator objective를 설계한다.
- 세 encoder 모두 시각 실패를 제대로 분리하지 못하면, c065는 adapter head 반복이 아니라 encoder-side checkpoint 학습/미세조정 단계로 넘어간다.
