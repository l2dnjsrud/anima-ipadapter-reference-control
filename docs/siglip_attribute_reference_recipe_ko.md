# SigLIP Attribute Reference 사용 레시피

최종 정리일: 2026-06-12

## 목적

이 레시피는 현재 IP-Adapter 연구에서 가장 실용적인 SigLIP reference-control 경로를 ComfyUI에서 재현하기 위한 것이다. 핵심은 참조 이미지만 던지는 방식이 아니라, 참조 이미지에서 보이는 속성을 프롬프트에 함께 넣고 native SigLIP IP-Adapter가 색감, 의상, 얼굴 프레이밍, 표정, 만화풍 질감을 당겨오게 하는 것이다.

## 사용할 노드

이 경로는 반드시 `AnimaSigLIP*` 노드로 사용한다.

- `AnimaSigLIPIPAdapterLoader`
- `AnimaSigLIPEncodeImage`
- `AnimaSigLIPIPAdapterApply`

`AnimaPE*` 노드로 로드하지 않는다. 체크포인트 파일명에 `pe_space` 또는 `pe_retrieval`이 남아 있어도 이것은 학습 과정에서 PE teacher/anchor 신호를 썼다는 뜻이지 PE-Core 노드용 체크포인트라는 뜻이 아니다.

## 추천 workflow

ComfyUI에서 다음 파일을 import한다.

```text
workflows/anima_ipadapter_siglip_attribute_reference.json
```

이 workflow는 일반 ComfyUI 그래프 형태다.

```text
LoadImage
  -> AnimaSigLIPEncodeImage
AnimaSigLIPIPAdapterLoader + UNETLoader
  -> AnimaSigLIPIPAdapterApply
  -> CFGGuider / BasicScheduler / SamplerCustomAdvanced
  -> VAEDecode
  -> SaveImage
```

## 추천 variant 이름

사용자-facing report와 새 eval에서는 다음 이름을 사용한다.

| 노출명 | 실제 체크포인트 | 용도 |
|---|---|---|
| `siglip_ref_retrieval_w14` | `anima_siglip_ip_adapter_single_character_clean32_pe_retrieval_0128_20260611.safetensors` | 기본 추천. c034에서 mean uplift `+0.1452`, improved rate `87.5%` |
| `siglip_kv_init_w14` | `anima_siglip_ip_adapter_single_character_clean32_pe_space_init_0512_20260611.safetensors` | fallback. c034에서 mean uplift `+0.1103`, improved rate `87.5%` |

## 추천 값

| 항목 | 값 |
|---|---|
| adapter weight | `1.4` |
| start_at | `0.0` |
| end_at | `0.85` |
| sampler | `er_sde` |
| scheduler | `simple` |
| steps | `18` |
| guidance | `3.2` - `3.5` |
| size | 단일 캐릭터 기준 `768x1024` 우선 |

## 프롬프트 원칙

generic prompt만으로는 아직 reference-only identity control이 안정적으로 해결되지 않았다. 참조 이미지에서 보이는 속성을 프롬프트에 넣어야 한다.

좋은 프롬프트는 다음 범주를 포함한다.

- 나이와 수염: `old bearded martial arts master`, `white gray beard and thick eyebrows`
- 머리와 색: `long flowing black hair`, `red hair and pale makeup`
- 표정: `stern serious expression`, `angry tense expression`, `open screaming mouth`
- 의상과 색: `black martial robe with red trim`, `tan traditional robe`
- 구도: `upper body close-up portrait`, `side profile portrait`
- 소품/직책: `black official hat`, `scholar robe`
- 비인간 특징: `green monster face with red glowing eye`
- 배경/조명: `cool blue gray palace lighting`, `red gold indoor palace colors`

자동 생성은 다음 도구를 사용한다.

```bash
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python tools/build_reference_prompt_manifest.py \
  <source_manifest.jsonl> \
  /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  <output_auto_reference_prompts.jsonl>
```

생성된 prompt manifest는 다음 eval runner에 넣는다.

```bash
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python tools/siglip_auto_caption_eval.py \
  <output_auto_reference_prompts.jsonl> \
  --out-dir eval/<new_siglip_runtime_eval_dir>
```

`--out-dir`는 항상 새 디렉터리로 지정한다. 기존 c033/c034 baseline 디렉터리에 다시 쓰지 않는다.

## 현재 한계

- generic prompt + reference image만으로 identity, props, palette를 모두 복구하는 단계는 아니다.
- PE pooled-cosine은 보조 지표다. c034의 monster row처럼 시각적으로 더 가까운데 metric은 낮게 나오는 경우가 있다.
- 선화 채색은 이 레시피의 목표가 아니다. 선화 채색에는 EasyControl/ControlNet 계열 spatial control이 별도로 필요하다.
- QwenVL adapter-only checkpoint는 아직 품질 통과가 아니다. 현재 Qwen3-VL은 attribute prompt scoring 용도로 쓰는 것이 더 안정적이다.

## 판단 기준

새 평가에서는 최소한 다음을 함께 본다.

- no-IP 대비 mean uplift
- improved rate
- blank output 여부
- contact sheet row-level visual audit
- identity/distinctive trait 보존 여부

c034 기준 현재 최고 결과는 `siglip_ref_retrieval_w14`다. 다음 c035 평가는 이 레시피가 32-case single-character suite에서도 유지되는지 확인하는 것이 목표다.
