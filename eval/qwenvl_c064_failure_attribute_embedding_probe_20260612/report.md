# c064 Failure-Attribute Embedding Probe

## 결론

판정: `encoder_side_checkpoint_required_for_hard_failures`

c064는 새 이미지를 생성하지 않고, c063 gate에서 이미 생성된 heldout hard case 3개를 QwenVL, SigLIP2, PE embedding 공간에서 다시 측정했다. 결과적으로 기존 off-the-shelf encoder 공간만으로는 hard failure를 안정적으로 분리하지 못했다.

- QwenVL: `heldout01`만 support, `heldout05/07` 실패
- SigLIP2: `heldout01/05/07` 모두 실패
- PE: `heldout05`만 support, `heldout01/07` 실패
- 가장 중요한 `heldout07` non-human green side-profile은 세 encoder 모두 `no_ip`를 1위로 두었고 IP 결과는 negative uplift였다.

따라서 다음 루프는 adapter continuation/calibrator-only 반복이 아니라, 실패 속성을 직접 학습하는 encoder-side checkpoint 또는 attribute teacher 단계로 넘어가야 한다.

## 입력

- Plan: `docs/c064_failure_attribute_embedding_probe_plan_ko.md`
- Manifest: `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/probe_manifest.jsonl`
- Source gate: `eval/qwenvl_c063_calibrator_only_gate_20260612/summary.json`
- Source reference root: `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`

비교 variant:

- `no_ip`
- `blend_species_face`
- `c063_calibrator_only_w14`

## Encoder별 결과

### QwenVL

- Metrics: `qwenvl_probe_metrics.json`
- Report: `qwenvl_probe_report.md`
- Summary: 3 cases 중 1 case support

| sample | attribute | best | best uplift | top margin | decision |
| --- | --- | --- | ---: | ---: | --- |
| heldout01 | old-face/speech-bubble-context-side-profile | c063_calibrator_only_w14 | 0.096285 | 0.022098 | support |
| heldout05 | old-bearded-official-black-hat-upper-body-crop | blend_species_face | 0.016990 | 0.016577 | not enough |
| heldout07 | non-human-green-monster-side-profile-red-eye | no_ip | 0.000000 | 0.051999 | not enough |

QwenVL은 c063 때와 같은 패턴이다. heldout01은 수치상 c063을 좋게 보지만, visual audit에서는 old-face/crop/speech-bubble context가 충분히 복원되지 않았다. 즉 QwenVL score가 좋아져도 생성 품질 판단과 완전히 일치하지 않는다.

### SigLIP2

- Metrics: `siglip_probe_metrics.json`
- Report: `siglip_probe_report.md`
- Summary: 3 cases 중 0 case support

| sample | attribute | best | best uplift | top margin | decision |
| --- | --- | --- | ---: | ---: | --- |
| heldout01 | old-face/speech-bubble-context-side-profile | no_ip | 0.000000 | 0.028017 | not enough |
| heldout05 | old-bearded-official-black-hat-upper-body-crop | no_ip | 0.000000 | 0.010737 | not enough |
| heldout07 | non-human-green-monster-side-profile-red-eye | no_ip | 0.000000 | 0.014182 | not enough |

SigLIP2 pooled embedding은 이 hard set에서 IP 적용 결과보다 no-IP baseline을 더 reference에 가깝다고 판단했다. 지금 형태 그대로는 supervised adapter/calibrator teacher로 쓰기 어렵다.

### PE

- Metrics: `pe_probe_metrics.json`
- Report: `pe_probe_report.md`
- Summary: 3 cases 중 1 case support

| sample | attribute | best | best uplift | top margin | decision |
| --- | --- | --- | ---: | ---: | --- |
| heldout01 | old-face/speech-bubble-context-side-profile | blend_species_face | 0.043998 | 0.020951 | not enough |
| heldout05 | old-bearded-official-black-hat-upper-body-crop | blend_species_face | 0.093415 | 0.093415 | support |
| heldout07 | non-human-green-monster-side-profile-red-eye | no_ip | 0.000000 | 0.095589 | not enough |

PE는 heldout05의 bearded official/headwear/crop 계열은 잘 분리하지만, heldout07 non-human profile에서는 IP 적용 결과를 오히려 reference에서 멀어진 것으로 본다.

## Hard Failure별 판정

### heldout01

QwenVL은 c063을 1위로 두지만 PE는 blend를 1위, SigLIP2는 no-IP를 1위로 둔다. encoder 간 판단이 불일치하고 visual audit과도 완전히 맞지 않으므로, 이 케이스 하나만 보고 QwenVL shallow calibrator를 더 미는 것은 위험하다.

### heldout05

PE는 blend가 확실히 좋다고 판단하지만 QwenVL은 uplift가 0.017로 약하고 SigLIP2는 no-IP를 1위로 둔다. bearded official/headwear/crop은 PE 쪽 teacher 신호가 일부 있지만, QwenVL adapter에 바로 distill하기에는 encoder 간 일관성이 부족하다.

### heldout07

세 encoder 모두 no-IP를 1위로 둔다. blend와 c063 모두 negative uplift다.

이 케이스는 현재 reference-control 실패의 핵심이다. non-human species, green skin, side-profile silhouette, red glowing eye 같은 속성이 기존 embedding space에서 생성 결과와 reference를 제대로 묶지 못한다. adapter head만 추가 학습해서 해결하기 어렵다는 쪽으로 판단한다.

## 다음 실험 분기

c065는 다음 중 하나로 가야 한다.

1. Encoder-side failure-attribute checkpoint
   - color dataset에서 single-character crop을 사용한다.
   - positive는 same/ref attribute, negative는 human-template collapse/다른 species/다른 headwear/다른 crop으로 구성한다.
   - 목표는 QwenVL 또는 SigLIP2 embedding 자체가 heldout07 같은 non-human 속성을 no-IP/generic human보다 높게 분리하도록 만드는 것이다.

2. Attribute teacher reranker
   - QwenVL/PE/SigLIP2를 단순 cosine teacher로 쓰지 않는다.
   - non-human, beard/headwear, crop/profile 같은 failure labels를 별도 classifier/reranker로 학습해 adapter 학습 loss에 보조 신호로 넣는다.

현재 증거 기준으로 우선순위는 `1. Encoder-side failure-attribute checkpoint`다. heldout07이 세 encoder 모두에서 실패했기 때문에 teacher 조합만으로는 핵심 붕괴를 고치기 어렵다.
