# Anima IP-Adapter 연구 일지

최종 정리일: 2026-06-12
범위: `/home/wktwin/anima-ipadapter-reference-control` 레포의 IP-Adapter 연구, 학습, ComfyUI 노드, 평가 기록만 다룬다.

## 1. 연구가 시작된 이유

처음 목표는 선화/채색 데이터셋처럼 컷 배치, 인물 스타일, 색감, 복장, 분위기를 참조 이미지에서 가져와서 Anima 계열 모델에서 안정적으로 재현하는 것이었다. 단순히 이미지를 한 장 생성하는 수준이 아니라, ComfyUI에서 실제 워크플로우로 사용할 수 있는 고품질 reference-control 모델이 필요했다.

초기 조사에서 Wenaka의 `comfyui-anima-ipadapter` 프로젝트와 `Wenaka/anima-ip-adapter-dataset`가 발견되었지만, 바로 쓸 수 있는 완성 IP-Adapter 모델이 공개되어 있지 않았다. 사용자 제공 댓글에서도 같은 문제가 지적되었다. 댓글 요지는 공개된 모델이 없고, 2B VL 모델을 쓰는 이유는 이미지 인코딩을 담당할 수 있는 작은 VL 모델이 필요하기 때문이며, SigLIP 시도는 쉽지 않았고 QwenVL 또는 애니메이션/만화 전용 이미지 인코더가 필요할 수 있다는 것이었다.

따라서 연구 목표는 다음처럼 잡혔다.

- Wenaka 프로젝트를 그대로 쓰는 것이 가능한지 확인한다.
- 기존 IP-Adapter, ComfyUI IPAdapter Plus, FaceID 계열 구조를 조사한다.
- Anima 모델에 맞는 native ComfyUI IP-Adapter 노드와 체크포인트 형식을 만든다.
- PE-Core, SigLIP, QwenVL 계열을 각각 실험해서 실제 참조 반영 품질을 판단한다.
- 최종적으로는 “믿고 쓸 수 있는 고품질 reference-control”을 목표로 하되, 선화 채색은 별도 spatial-control 문제인지도 확인한다.

## 2. 핵심 결론

현재까지의 결론은 세 가지다.

1. PE-Core 기반 IP-Adapter는 실제로 참조 영향을 준다. ComfyUI API contact-sheet 평가에서 best scale `1.0`, mean uplift `+0.0937`, improved rate `87.5%`로 통과했다.
2. PE 또는 SigLIP IP-Adapter 단독으로 선화 페이지를 고품질 채색하는 것은 목표와 다르다. IP-Adapter는 참조/스타일 압력이고, 선화 채색에는 EasyControl/ControlNet 같은 공간 구조 제어가 별도로 필요하다.
3. native SigLIP 경로는 “일반 프롬프트 + 참조만으로 모든 걸 해결”하는 수준은 아직 아니다. 2026-06-12 c034 8-case에서는 자동 속성 프롬프트와 결합했을 때 좋은 신호가 나왔지만, c035 32-case 단일 캐릭터 suite에서는 best uplift `+0.0577`, improved rate `0.65625`, identity/distinctive trait `16/32`로 목표 gate를 통과하지 못했다. 따라서 현재 SigLIP recipe는 실험용 best path이지, 바로 믿고 쓰는 완성 reference-control 모델은 아니다.

## 3. 조사한 것과 조사 이유

### Wenaka Anima IP-Adapter

조사 이유는 가장 직접적인 출발점이 Anima용 IP-Adapter였기 때문이다. 코드와 데이터셋은 있었지만 공개된 완성 모델이 없었고, 사용자가 바로 ComfyUI에서 쓰려면 노드, 체크포인트, 워크플로우가 필요했다.

확인 결과 `Wenaka/anima-ip-adapter-dataset`는 public/ungated였고 약 36.5 GiB 규모였다. 파일은 `images_00.tar`부터 `images_13.tar`까지 이미지 tar shard 중심이었다. 하지만 학습에 필요한 `ref_id`, `tgt_id`, `prompt` 쌍 메타데이터는 데이터셋 preview/file list에서 확인되지 않았다. 그래서 단순 다운로드와 캡션만으로는 충분하지 않고, 참조-타깃 pairing rule 또는 pair mining이 필요하다고 판단했다.

### Tencent IP-Adapter

조사 이유는 원조 IP-Adapter의 학습 구조를 확인하기 위해서였다.

Tencent 구조는 CLIP text encoder, VAE, UNet, CLIP vision encoder를 고정하고 image projection model과 adapter module만 학습한다. 이미지 임베딩을 text hidden에 결합하고 diffusion noise-prediction MSE로 학습한다. 이 구조는 방향성 참고에는 유용했지만, SD/CLIP 중심이라 Anima native DiT/VL 구조에는 그대로 맞지 않았다.

### ComfyUI IPAdapter Plus

조사 이유는 사용자가 원하는 ComfyUI 사용 형태가 “모델 로더 -> 적용 노드 -> 샘플러”라는 표준 그래프였기 때문이다.

결론은 UX 패턴만 참고하고 구현은 그대로 쓰지 않는 쪽으로 정리했다. IPAdapter Plus는 ComfyUI `CLIP_VISION`, SD/SDXL checkpoint family, preset 중심 구조라 Anima SigLIP/QwenVL 경로와 직접 호환되지 않는다.

### FaceID / FaceID Plus V2

조사 이유는 사용자가 “IPAdapter Plus FaceID V2 같은 모델을 우리도 만들 수 있는지” 물었기 때문이다.

결론은 가능하지만 핵심 난이도는 adapter가 아니라 identity embedding이다. InsightFace/ArcFace는 실사 얼굴에 강하지만 무협 만화 캐릭터의 헤어스타일, 의상, 얼굴 각도, 선화/채색 차이를 안정적으로 묶어주는 공간은 아니다. 고품질 캐릭터 ID 모델을 만들려면 same-character group mining, 애니/만화 특화 metric model, identity/palette/prop token supervision이 필요하다.

## 4. 사용한 데이터셋

### Local color-panel dataset

주요 실험 데이터셋은 로컬의 color-panel 계열이었다.

```text
/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best
```

`tools/generate_pair_manifest.py` 기준 측정값은 다음과 같다.

| 항목 | 값 |
|---|---:|
| 디렉터리 | 269 |
| 소스 이미지 | 1,571 |
| 생성 pair row | 1,537 |
| singleton으로 스킵된 디렉터리 | 34 |

이 데이터셋은 color reference, panel/page style, 같은 디렉터리 내 인접 이미지 pair를 만들기에 적합했다. 초반 SigLIP smoke/pilot, identity8, identity128, clean32, single-character 실험이 여기서 파생되었다.

### Local single-character subsets

다중 컷 페이지는 참조 품질 판정이 어렵다는 판단 때문에 단일 캐릭터 subset을 따로 만들었다.

- `local_color_single_character_identity4_20260611.jsonl`
- `local_color_single_character_clean32_20260611.jsonl`
- `local_color_single_character_clean32_heldout8_20260611.jsonl`

단일 캐릭터 실험은 “참조 이미지의 인물 정체성, 색상, 소품, 표정이 실제로 넘어오는지”를 눈으로 확인하기 쉬워서 이후 SigLIP/QwenVL 평가의 중심 gate가 되었다.

### Wenaka public dataset

조사와 설계에는 포함했지만, 전체 다운로드와 full training은 진행하지 않았다. 이유는 크기가 약 36.5 GiB이고, 공개 파일만으로는 학습 pair 메타데이터가 부족했기 때문이다. 추후 사용하려면 이미지 다운로드, 캡션 생성, pair mining, 소량 검증, pilot training 순서가 필요하다.

### Line/color pair

선화 채색 가능성 확인에는 선화 이미지와 대응 color reference를 사용했다.

```text
line input: /home/wktwin/anima-lora-training-bundle/image_dataset/MS-138/MS-138__MS_138-05_LLM.jpg
color ref:  /home/wktwin/anima-lora-training-bundle/.pytest_cache/image_dataset_color/101-200/SG-138/SG-138-05.jpg
```

이 실험은 IP-Adapter 단독으로 채색 데이터셋급 결과가 나오는지 보기 위한 것이었고, 결론은 “단독으로는 부족, spatial control 필요”였다.

## 5. 개발한 것과 개발 이유

### PE-Core native ComfyUI 경로

PE-Core 기반 checkpoint가 실제로 작동하는지 보기 위해 native PE 노드와 평가 흐름을 만들었다.

- `native_pe.py`
- `native_pe_models.py`
- `native_pe_patch.py`
- `native_pe_runtime.py`
- `workflows/anima_ipadapter_pe_native_reference.json`
- `tools/reference_eval.py`

목적은 “일단 작동하는 baseline”을 확보하고, 이후 SigLIP/QwenVL 결과가 PE보다 나은지 또는 못한지 비교하기 위한 기준을 세우는 것이었다.

### Native SigLIP 경로

Wenaka 코드의 SigLIP 시도와 사용자 댓글의 문제의식을 이어받아 Anima 전용 SigLIP IP-Adapter family를 만들었다.

- `siglip_model.py`
- `siglip_checkpoint.py`
- `native_siglip.py`
- `native_siglip_runtime.py`
- `siglip_feature_calibration.py`
- `native_ip_attention.py`
- `workflows/anima_ipadapter_siglip_native_reference.json`

개발 중 중요한 수정은 PE-style query patch였다. 초반 native patch는 pre-attention hidden state에서 별도 IP cross-attention stream을 더하는 형태였는데, PE-Core는 실제 Anima attention query를 계산한 뒤 그 query로 IP K/V에 attention한다. 이 차이를 맞춘 뒤 참조 영향이 더 분명해졌다.

### Native QwenVL 경로

댓글에서 QwenVL 가능성이 언급되었고, `Qwen/Qwen3-VL-Embedding-2B`가 이미지 임베딩 후보로 보였기 때문에 별도 checkpoint family로 분리했다.

- `qwenvl_model.py`
- `qwenvl_checkpoint.py`
- `native_qwenvl.py`
- `qwenvl_feature_calibration.py`
- `training/qwenvl_real_smoke.py`
- `training/qwenvl_token_retrieval.py`

QwenVL checkpoint에는 `qwenvl_family` marker를 넣어 PE/SigLIP checkpoint와 섞이지 않도록 했다. 이는 이전에 사용자가 지적한 “모델 경로/노드가 이상하게 섞이는 문제”를 막기 위한 fail-loud 설계다.

### 학습 및 평가 도구

반복 평가와 문서화를 위해 다음 도구를 만들었다.

- `tools/generate_pair_manifest.py`: 로컬 이미지 디렉터리에서 pair manifest 생성
- `tools/build_reference_prompt_manifest.py`: Qwen3-VL embedding으로 bounded attribute prompt 생성
- `tools/siglip_auto_caption_eval.py`: 자동 속성 프롬프트 기반 ComfyUI API 평가
- `tools/score_siglip_auto_caption_metrics.py`: no-IP 대비 PE cosine uplift 계산
- `training/siglip_real_smoke.py`: SigLIP real smoke/pilot 학습
- `training/siglip_reference_loss.py`: correct/wrong reference contrastive loss
- `training/pe_teacher_distillation.py`: PE teacher distillation
- `training/pe_token_retrieval.py`: PE-token retrieval loss
- `training/pe_space_siglip_adapter.py`: PE K/V projection을 SigLIP 쪽에 이식하는 초기화

## 6. 검증 방법

평가는 크게 네 단계로 진행했다.

1. 체크포인트 로드 검증: family marker와 tensor shape로 PE/SigLIP/QwenVL checkpoint가 잘못 섞이지 않게 했다.
2. ComfyUI 노드 검증: `object_info`와 API 실행으로 노드가 UI/API에 보이는지 확인했다.
3. 이미지 생성 검증: ComfyUI HTTP API로 no-IP baseline과 IP 적용 결과를 생성하고 contact sheet를 만들었다.
4. 품질 판단: PE pooled-cosine uplift, improved rate, nonblank/pixel std 같은 보조 지표와 사람 눈으로 보는 identity/palette/props/layout 평가를 함께 사용했다.

중요한 점은 PE pooled-cosine은 보조 지표라는 것이다. 특히 c034 green monster row처럼 시각적으로는 SigLIP 결과가 더 가까운데 PE cosine은 낮게 나오는 경우가 있었다. 그래서 최종 판단은 수치와 contact sheet를 같이 봤다.

## 7. 모델 버전별 연구 기록

### 7.1 PE-Core baseline

| 모델 | 목적 | 검증 목표 | 결과 |
|---|---|---|---|
| `anima_ip_adapter_quality_20260610.safetensors` | 작동 가능한 reference-control baseline | ComfyUI API에서 no-IP 대비 참조 유사도 개선 | PASS. best scale `1.0`, mean uplift `+0.0937`, improved rate `87.5%` |

이 모델은 바로 참조 영향이 확인된 baseline이다. 다만 이 결과는 PE family 내부 cosine 기반이고, character identity를 완전히 복구한다는 뜻은 아니다. layout/style reference-control 성격이 강하다.

### 7.2 선화 채색 실험

| 경로 | 목적 | 결과 |
|---|---|---|
| PE IP-Adapter img2img only | 선화 입력과 color reference만으로 채색 가능성 확인 | 색감/스타일 압력은 보였지만 원본 패널 구조가 무너짐 |
| PE IP-Adapter low denoise | 낮은 denoise로 구조 보존 가능성 확인 | 여전히 불안정 |
| AnimaEasyControlPatch + PE IP-Adapter | spatial line control과 reference control 결합 | 패널/말풍선/배치 보존은 크게 개선, 색은 적용되지만 최종 채색 품질은 아님 |

결론은 IP-Adapter 자체의 한계라기보다 역할 차이다. IP-Adapter는 참조 스타일/이미지 조건이고, 선화 채색은 입력 선의 공간 구조를 보존해야 하므로 별도 line-control/colorize checkpoint가 필요하다.

### 7.3 SigLIP 초기 smoke/pilot

| 모델 | 학습 방식 | 검증 결과 |
|---|---|---|
| `anima_siglip_ip_adapter_smoke_20260610.safetensors` | local color pair 1-step smoke | finite loss, loadable, PE checkpoint rejection 확인 |
| `anima_siglip_ip_adapter_pilot_20260610.safetensors` | local color pair 16-step pilot | checkpoint-compatible, tensor 변화 확인. 시각 품질 pass는 아님 |
| `anima_siglip_ip_adapter_color64_continue_20260611.safetensors` | color64 continuation | prompt에 identity 힌트가 있을 때 시각 강도 증가. 힌트 제거 identity test는 fail |
| `anima_siglip_ip_adapter_self64_continue_20260611.safetensors` | self reconstruction continuation | identity test fail |
| `anima_siglip_ip_adapter_self512_continue_20260611.safetensors` | longer self reconstruction | identity test fail |

초기 결론은 “노드는 작동하고 픽셀은 변하지만, reference identity를 믿고 맡길 수준은 아니다”였다.

### 7.4 SigLIP overfit과 identity scale-up

| 모델 | 목적 | 결과 |
|---|---|---|
| `anima_siglip_ip_adapter_ref03_overfit1024_20260611.safetensors` | 한 장 reference/target overfit으로 가능성 확인 | overfit pass. prompt identity 힌트 없이도 monk 얼굴, red beads, robe color 일부 회복 |
| `anima_siglip_ip_adapter_identity8_1024_20260611.safetensors` | 8-reference self reconstruction | reference influence는 보이나 identity generalization 불완전 |
| `anima_siglip_ip_adapter_identity128_1024_20260611.safetensors` | 128-reference bf16 continuation | loss는 내려가나 visual fail, scale movement 미약 |
| `anima_siglip_ip_adapter_identity128_2048_20260611.safetensors` | longer bf16 continuation | visual fail |
| `anima_siglip_ip_adapter_identity128_fp32_3072_20260611.safetensors` | fp32 adapter training 안정화 | 253/255 tensor 이동 확인, 하지만 visual quality fail |

이 단계에서 native SigLIP 자체가 불가능한 것은 아니라는 증거가 생겼다. 한 장 overfit은 됐기 때문이다. 하지만 frozen SigLIP2 + adapter-only 구조만 오래 돌리는 것은 고품질 일반화로 이어지지 않았다.

### 7.5 SigLIP contrastive/calibration/teacher 계열

| 모델 | 목적 | 결과 |
|---|---|---|
| `anima_siglip_ip_adapter_identity128_contrastive_0064_20260611.safetensors` | correct/wrong reference contrastive smoke | loadable, reference distance 개선 신호 |
| `anima_siglip_ip_adapter_identity128_contrastive_0512_20260611.safetensors` | contrastive 512-step | reference-dependent variation 증가, quality fail |
| `anima_siglip_ip_adapter_identity128_calibrated_contrastive_0064_20260611.safetensors` | feature calibrator 추가 | 일부 row 개선, quality pass 아님 |
| `anima_siglip_ip_adapter_identity128_calibrated_contrastive_0576_20260611.safetensors` | longer calibrated continuation | scene average 쪽으로 overfit |
| `anima_siglip_ip_adapter_teacher_smoke_0002_20260611.safetensors` | PE teacher distillation smoke | finite/loadable |
| `anima_siglip_ip_adapter_identity128_pe_teacher_0064_20260611.safetensors` | PE teacher distillation | reference-dependent output은 보이나 quality pass 아님 |

이 단계의 결론은 loss/teacher/calibrator가 “기계적으로 작동”하는 것과 “좋은 reference-control”은 다르다는 점이다. 결과가 계속 generic court/interior/wuxia template으로 모이는 문제가 있었다.

### 7.6 PE-style query patch와 단일 캐릭터 실험

| 모델 | 목적 | 결과 |
|---|---|---|
| `anima_siglip_ip_adapter_identity128_pe_query_patch_0064_20260611.safetensors` | PE-style query patch 후 짧은 재학습 | 참조 영향 증가, quality pass 아님 |
| `anima_siglip_ip_adapter_single_character_identity4_pe_query_patch_0256_20260611.safetensors` | 4개 단일 캐릭터 micro-train | palette/pose 일부 개선, identity pass 아님 |
| `anima_siglip_ip_adapter_single_character_clean32_pe_query_patch_0512_20260611.safetensors` | clean32 단일 캐릭터 확장 | 단일 캐릭터 gate는 유효하나 props/beard/glasses/demon face fail |
| `anima_siglip_ip_adapter_single_character_clean32_token_sep_0256_20260611.safetensors` | token separation loss | variation은 증가, fidelity 개선은 부족 |
| `anima_siglip_ip_adapter_single_character_clean32_pe_token_anchor_0256_20260611.safetensors` | PE-token anchor | 안정성은 개선, identity pass 아님 |

사용자가 지적한 것처럼 단일 캐릭터 기준 평가는 훨씬 보기 쉬웠고, 이후 주요 검증 gate가 되었다. 그러나 단일 캐릭터에서도 frozen SigLIP adapter-only 방식은 identity, 소품, 비인간 얼굴, 특수 표정을 충분히 복원하지 못했다.

### 7.7 PE-space / retrieval SigLIP

| 모델 | 목적 | 결과 |
|---|---|---|
| `anima_siglip_ip_adapter_single_character_clean32_pe_space_init_0512_20260611.safetensors` | PE adapter의 K/V projection과 gate를 SigLIP 쪽에 초기화 | sharp/clean 결과, 그러나 young black-haired wuxia male template으로 수렴 |
| `anima_siglip_ip_adapter_single_character_clean32_pe_retrieval_0128_20260611.safetensors` | matching PE token retrieval loss | clean output 유지, elder/scholar/green demon identity fail |

이 두 모델은 이후 속성 프롬프트와 결합했을 때 가장 실용적인 SigLIP runtime 후보가 되었다. 이름에 `pe`가 들어가지만 PE-Core 노드를 쓰는 것이 아니라, SigLIP checkpoint가 PE-space 초기화/teacher/retrieval 신호를 사용했다는 뜻이다.

### 7.8 QwenVL 계열

| 모델 | 목적 | 결과 |
|---|---|---|
| `anima_qwenvl_ip_adapter_smoke_0002_20260611.safetensors` | QwenVL family smoke | loadable, native node surface 확인 |
| `anima_qwenvl_ip_adapter_identity128_0064_20260611.safetensors` | QwenVL 64-step identity128 | finite loss, output 변화, quality pass 아님 |
| `anima_qwenvl_ip_adapter_identity128_contrastive_0064_20260611.safetensors` | QwenVL contrastive | output 변화, generic yellow-robed/interior collapse |
| `anima_qwenvl_ip_adapter_identity128_calibrated_contrastive_0064_20260611.safetensors` | QwenVL feature calibration | trains/loads, reference collapse 유지 |
| `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors` | QwenVL token retrieval | generic black-haired wuxia male template으로 collapse |

QwenVL은 embedding 후보로는 좋다. `Qwen/Qwen3-VL-Embedding-2B` probe에서 2048-d embedding이 생성되고 reference 분리도 확인됐다. 하지만 frozen QwenVL embedding + adapter-only 학습은 아직 quality pass를 만들지 못했다. QwenVL은 caption/attribute generation 또는 더 강한 image-feature encoder/calibrator 쪽으로 쓰는 것이 현재로서는 더 설득력 있다.

### 7.9 속성 프롬프트와 자동 prompt manifest

| 평가 | 목적 | 결과 |
|---|---|---|
| c031 QwenVL attribute prompt | generic prompt 실패를 속성 프롬프트로 재검증 | promising, primary pass는 아님 |
| c032 SigLIP attribute prompt | SigLIP + visible attribute prompt | first practical quality pass. 현재 노출명 기준 `siglip_kv_init_w14` 8/8 개선, uplift `+0.0603`; `siglip_ref_retrieval_w14` 7/8 개선, uplift `+0.0670` |
| c033 auto caption vocab1 | Qwen3-VL embedding 기반 자동 속성 prompt | useful but red-haired female/green monster under-described |
| c034 auto caption vocab2 | vocab 확장 후 자동 속성 prompt | visual pass with PE metric caveat. 현재 노출명 기준 `siglip_kv_init_w14` uplift `+0.1103`, `siglip_ref_retrieval_w14` uplift `+0.1452`, 둘 다 7/8 개선 |
| c035 reference suite v1 | 32-case single-character suite로 확장 검증 | `siglip_ref_retrieval_w14`가 best지만 uplift `+0.0577`, improved rate `0.65625`, identity/distinctive trait `16/32`로 not_ready |

이 단계가 현재 가장 중요한 성과이자 한계다. SigLIP adapter가 참조 이미지를 혼자 다 해석하게 맡기면 약하고, 자동 속성 프롬프트가 visible identity/palette/prop/expression/non-human trait를 잡아주면 broad style과 일부 디테일을 당겨온다. 하지만 c035에서 검은 장발 무협 캐릭터, 보라색/밤 궁전 배경, 붉은 눈, generic official/elder template으로 수렴하는 문제가 반복되어 고품질 reference-control로 패키징하기에는 부족하다고 판단했다.

### 7.10 2026-06-12 신규 외부 연구 반영: InterleaveThinker / i1

사용자가 제보한 두 공개 저장소를 2026-06-12 기준으로 확인했다. 확인 기준은 공식 GitHub 저장소와 README다.

| 저장소 | 확인한 내용 | 이 프로젝트에 반영할 점 | 그대로 쓰면 안 되는 점 |
|---|---|---|---|
| `zhengdian1/InterleaveThinker` | 기존 이미지 생성기에 planner/critic 기반 interleaved generation 루프를 붙이는 multi-agent pipeline이다. 2026-06-12에 paper, models, training, inference 공개가 공지됐다. | 현재 c035 실패처럼 한 번 생성하고 끝내는 방식이 아니라, 참조 이미지/프롬프트/출력의 차이를 critic이 구조화하고 다음 prompt 또는 모델 경로를 수정하는 `reference-control critic` 루프를 만든다. | IP-Adapter의 K/V 주입 구조나 checkpoint family를 직접 제공하는 연구는 아니다. 우리 adapter 구조 근거로 바로 가져오면 안 된다. |
| `zlab-princeton/i1` | 3B text-to-image 모델을 위한 fully open recipe, 데이터 처리, JAX 학습, PyTorch 추론 구조를 공개한 저장소다. | 고해상도 T2I backbone을 만들 때의 data processing, recaptioning, training recipe를 참고한다. 특히 우리 pair mining/attribute caption 품질을 높이는 자료로 본다. | reference-image conditioning이나 IP-Adapter 결합 자체가 핵심인 저장소가 아니다. reference-control adapter의 직접 대체재로 취급하지 않는다. |

확인한 공식 출처:

- `https://github.com/zhengdian1/InterleaveThinker`, 확인 SHA `440c1b879cd4913b0382761f7bfa8297a32dc7d6`
- `https://github.com/zlab-princeton/i1`, 확인 SHA `cd6a34fd8e7fa7a0b7de36ff4602363e607f8a72`

이 반영으로 다음 판단이 바뀐다. 단순히 frozen SigLIP2 adapter-only 학습을 더 오래 돌리는 대신, 먼저 `agentic_reference_control_loop`를 만든다. 이 루프는 c035 같은 평가 케이스마다 참조 이미지, 자동 속성 프롬프트, no-IP 출력, IP 출력, visual audit 결과를 묶고, 누락된 identity/palette/prop/non-human trait를 구조화한다. 그 다음 route를 `prompt_patch`, `data_pair_mining`, `stronger_encoder_training`, `line_control_track` 중 하나로 결정한다.

정리하면 InterleaveThinker는 우리에게 모델 구조가 아니라 `planner/critic 운영 방식`을 주고, i1은 reference adapter가 아니라 `데이터/recaption/backbone 학습 레시피`를 준다. 둘 다 직접 IP-Adapter 완성품으로 포장하지 않는다.

### 7.11 2026-06-12 Agentic reference-control audit manifest v1

7.10의 판단을 코드로 고정하기 위해 c035 suite를 audit manifest로 변환하는 도구와 테스트를 추가했다.

- 도구: `tools/build_reference_control_audit_manifest.py`
- 테스트: `tests/test_reference_control_audit_manifest.py`
- 산출물: `eval/siglip_runtime_quality_20260612_c035_suite_v1/reference_control_audit_manifest.jsonl`
- 요약: `eval/siglip_runtime_quality_20260612_c035_suite_v1/reference_control_audit_summary.md`

실행 결과 c035 32 row 전체가 audit row로 묶였고, route 분포는 다음과 같다.

| route | rows | 의미 |
|---|---:|---|
| `stronger_encoder` | 16 | identity/distinctive trait 또는 non-human/special trait 실패가 중심이다. |
| `prompt_patch` | 6 | visual gate는 괜찮지만 metric이 떨어지거나 prompt 보강이 먼저 필요한 케이스다. |
| `hold` | 10 | 현재 audit rule 기준으로 직접 학습 전 보류 가능한 케이스다. |
| `pair_mining` | 0 | v1 rule에는 아직 same-character pair 부족 판단을 자동 배정하지 않았다. |
| `line_control` | 0 | c035는 선화/페이지 구조 채색 실패 suite가 아니라 single-character reference suite다. |

따라서 다음 실제 모델 작업은 `stronger_encoder` 쪽이 우선이다. 단, 이것은 현재 SigLIP/QwenVL checkpoint가 완성됐다는 뜻이 아니다. audit v1은 오히려 c035 `not_ready` 판단을 더 구조화해서, 새 학습 전에 실패 원인을 route별로 나누는 역할을 한다.

### 7.12 2026-06-12 QwenVL pooled metric probe c036

`stronger_encoder`로 바로 장기 학습을 시작하기 전에, `Qwen/Qwen3-VL-Embedding-2B` pooled image embedding이 c035의 더 나은 품질 지표나 학습 신호가 될 수 있는지 확인했다.

- 도구: `tools/score_auto_caption_qwenvl_metrics.py`
- 테스트: `tests/test_score_auto_caption_qwenvl_metrics.py`
- 입력: `eval/siglip_runtime_quality_20260612_c035_suite_v1/summary.json`
- 산출물: `eval/qwenvl_metric_probe_20260612_c036_c035/qwenvl_similarity_metrics.json`
- 보고서: `eval/qwenvl_metric_probe_20260612_c036_c035/report.md`

수치상으로는 QwenVL pooled metric이 PE pooled-cosine보다 낙관적이었다.

| variant | mean uplift | improved rate |
|---|---:|---:|
| `siglip_kv_init_w14` | +0.0422 | 0.84375 |
| `siglip_ref_retrieval_w14` | +0.0446 | 0.90625 |

하지만 visual audit와 맞춰보면 문제가 있었다. identity-pass row보다 identity-fail row의 평균 uplift가 더 높았다. 즉 QwenVL pooled embedding은 broad style, 색감, 의상, 구도, 만화풍 유사도에는 반응하지만, 우리가 c035에서 실패로 본 identity/distinctive trait collapse를 충분히 벌점화하지 못했다.

결정: `qwenvl_pooled_metric_auxiliary_only`

따라서 QwenVL pooled embedding은 보조 지표로만 사용한다. 다음 stronger-encoder 루프는 pooled similarity를 바로 loss/gate로 삼지 말고, identity-positive/negative pair mining과 feature separation probe를 먼저 통과해야 한다.

### 7.13 2026-06-12 Identity feature probe c037

c036의 결론을 코드로 검증하기 위해, PE/QwenVL/SigLIP2 pooled image feature가 약한 identity-positive/negative pair를 분리하는지 확인했다.

- 도구: `tools/build_identity_pair_probe_manifest.py`
- feature wrapper: `tools/image_feature_embedders.py`
- scoring: `tools/score_identity_pair_probe.py`
- 테스트: `tests/test_identity_feature_probe.py`
- 산출물: `eval/identity_feature_probe_20260612_c037/report.md`

입력 pair는 현재 color dataset에서 같은 SG 폴더를 positive proxy, 다음 SG 폴더를 negative proxy로 만든 약한 게이트다. 같은 캐릭터가 검증된 정답셋은 아니므로, 여기서 통과해도 바로 완성 identity benchmark는 아니다. 반대로 여기서 실패하면 pooled feature를 primary identity loss나 gate로 올리면 안 된다.

| encoder | positive mean | negative mean | margin | pairwise AUC | decision |
|---|---:|---:|---:|---:|---|
| PE | 0.8560 | 0.8404 | 0.0156 | 0.5806 | `feature_not_sufficiently_separated` |
| Qwen/Qwen3-VL-Embedding-2B | 0.7893 | 0.7567 | 0.0326 | 0.5913 | `feature_not_sufficiently_separated` |
| SigLIP2 base patch16 512 | 0.8932 | 0.8800 | 0.0132 | 0.5759 | `feature_not_sufficiently_separated` |

결정: `pooled_identity_feature_not_ready`

세 encoder 모두 margin `>= 0.05`, AUC `>= 0.70` 기준을 통과하지 못했다. QwenVL pooled feature가 margin은 가장 높았지만 AUC가 0.60 미만이라 identity-positive/negative separation으로 보기 어렵다. 따라서 다음 stronger-encoder 루프는 pooled cosine을 직접 학습 신호로 쓰지 않고, 더 엄격한 same-character mining, visual-token/deep-layer feature probe, 또는 작은 metric head/calibrator를 먼저 검증한다.

### 7.14 2026-06-12 Strict panel feature sanity probe c038

c037 실패가 feature extraction 자체의 문제인지, 아니면 same-SG proxy label이 너무 약한 문제인지 확인하기 위해 strict duplicate control을 실행했다.

- manifest: `eval/strict_identity_feature_probe_20260612_c038/strict_panel_pair_probe_manifest.jsonl`
- pooled scorer: `tools/score_identity_pair_probe.py`
- SigLIP token scorer: `tools/score_siglip_token_pair_probe.py`
- token metric helper: `tools/token_pair_probe_metrics.py`
- 종합 보고서: `eval/strict_identity_feature_probe_20260612_c038/report.md`

positive는 같은 panel key의 v4/v5 duplicate crop이고, negative는 같은 `SG-*` 폴더 안의 다른 panel key다. 이것은 캐릭터 identity benchmark가 아니라 feature pipeline sanity control이다.

| encoder/metric | margin | pairwise AUC | decision |
|---|---:|---:|---|
| Qwen3-VL pooled | +0.2061 | 1.0000 | pass |
| SigLIP2 pooled | +0.1058 | 1.0000 | pass |
| PE pooled | +0.1404 | 0.9998 | pass |
| SigLIP2 `mean_max_token` | +0.3170 | 1.0000 | pass |
| SigLIP2 layer `-6` pooled | +0.4739 | 0.9998 | pass |

결정: `strict_duplicate_feature_sanity_pass_identity_unsolved`

따라서 c037 실패는 feature pipeline이 완전히 망가진 것이 아니라, same-SG proxy가 identity label로 약하고 pooled feature가 duplicate/broad visual similarity에는 강하지만 true character identity를 보장하지 못한다는 쪽으로 해석한다. 다음 루프는 duplicate crop을 제외한 true same-character positive와 같은 장면/스타일 hard negative를 만들고, SigLIP layer `-6` pooled 및 `mean_max_token`을 후보로 다시 검증하는 것이다.

### 7.15 2026-06-12 True identity candidate review c039

c038 다음으로 duplicate panel을 제외한 true same-character 후보를 만들 수 있는지 확인했다.

- 도구: `tools/build_true_identity_candidate_review.py`
- 테스트: `tests/test_true_identity_candidate_review.py`
- 후보 manifest: `eval/true_identity_candidate_review_20260612_c039/candidate_pairs.jsonl`
- 후보 sheet: `eval/true_identity_candidate_review_20260612_c039/candidate_sheet.jpg`
- 보고서: `eval/true_identity_candidate_review_20260612_c039/report.md`

규칙은 같은 `SG-page` 안에서 서로 다른 panel key 조합을 뽑는 것이다. 시각 리뷰 결과, scene continuity는 있지만 true same-character positive로 바로 쓰기에는 노이즈가 많았다. 다른 인물, 배경/건물, 소품, 다인물 panel이 섞여 있었다.

결정: `same_page_candidates_need_character_filtering`

따라서 같은 `SG-page` 후보 mining은 label sheet 생성용으로는 유용하지만, 학습 manifest로 바로 승격하면 identity metric이 장면/구도/소품 유사도에 오염될 수 있다. 다음 루프는 후보 양쪽이 모두 캐릭터 중심인지 먼저 거르는 `character_filtered_identity_candidate_mining`이다.

### 7.16 2026-06-12 Character-filtered identity candidates c040

c039 후보에 Qwen3-VL image-text retrieval 기반 character-centered filter를 적용했다.

- 도구: `tools/filter_character_candidate_pairs.py`
- 테스트: `tests/test_character_candidate_filter.py`
- 입력: `eval/true_identity_candidate_review_20260612_c039/candidate_pairs.jsonl`
- scored 후보: `eval/character_filtered_identity_candidates_20260612_c040/scored_candidate_pairs.jsonl`
- kept 후보: `eval/character_filtered_identity_candidates_20260612_c040/kept_candidate_pairs.jsonl`
- kept sheet: `eval/character_filtered_identity_candidates_20260612_c040/kept_candidate_sheet.jpg`
- 보고서: `eval/character_filtered_identity_candidates_20260612_c040/report.md`

설정은 `max(character text score) - max(non-character text score)`가 양쪽 모두 `>= 0.15`일 때 keep으로 두었다. 24개 후보 중 14개가 남았다.

결정: `character_filter_reduces_noise_not_identity_labels`

필터는 배경/건물/소품 노이즈를 줄이는 보조 필터로는 쓸 수 있다. 그러나 남은 14개에도 다른 인물, 몸통 crop, 애매한 pair가 있어 true same-character label을 자동 생성하는 단계로 승격하지 않는다. 다음 루프는 kept 후보에 `same_character`, `different_character`, `unclear` 라벨을 붙일 수 있는 label sheet를 만드는 것이다.

### 7.17 2026-06-12 Reviewed identity candidates c041

c040 kept 후보 14개를 사람이 확인 가능한 reviewed manifest로 바꿨다.

- 도구: `tools/build_reviewed_identity_manifest.py`
- 테스트: `tests/test_reviewed_identity_manifest.py`
- 수동 라벨: `eval/reviewed_identity_candidates_20260612_c041/manual_visual_labels.jsonl`
- reviewed manifest: `eval/reviewed_identity_candidates_20260612_c041/reviewed_candidate_pairs.jsonl`
- usable positives: `eval/reviewed_identity_candidates_20260612_c041/usable_positive_pairs.jsonl`
- different-character negatives: `eval/reviewed_identity_candidates_20260612_c041/different_character_pairs.jsonl`
- sheet: `eval/reviewed_identity_candidates_20260612_c041/reviewed_candidate_sheet.jpg`
- 보고서: `eval/reviewed_identity_candidates_20260612_c041/report.md`

결과는 reviewed rows 14개 중 `same_character` 6개, `different_character` 3개, `unclear` 5개, `positive_usable` 4개다.

결정: `reviewed_seed_too_small_for_training_gate`

c041 manifest는 true same-character feature probe의 작은 seed로는 쓸 수 있다. 하지만 4개 usable positive만으로 adapter 학습이나 metric-head 학습을 시작하면 오버핏/노이즈 위험이 크다. 다음 루프는 c041 seed로 SigLIP/QwenVL/PE feature가 같은/다른 캐릭터를 분리하는지 sanity probe를 돌리고, 동시에 mining 범위를 넓혀 usable positive를 수십 개 이상으로 늘리는 것이다.

## 8. 현재 판단

### 바로 믿고 쓸 수 있는 것

- PE-Core baseline은 reference-control 영향이 검증되어 있다.
- SigLIP `siglip_kv_init_w14` / `siglip_ref_retrieval_w14` 계열은 자동 속성 프롬프트와 함께 사용할 때 가장 나은 실험 경로다. 다만 c035 기준으로는 완성 모델이 아니라 `not_ready`다. 예전 로그의 `pe_space` / `pe_retrieval` 표현은 PE-Core 노드가 아니라 SigLIP 체크포인트의 학습-time PE teacher/anchor 출처를 뜻한다.

### 아직 믿고 쓰기 어려운 것

- generic prompt + reference image만으로 identity, props, palette를 전부 복구하는 모델은 아직 없다.
- QwenVL adapter-only checkpoint들은 output을 바꾸지만 reference fidelity가 부족하다.
- QwenVL pooled image embedding은 broad similarity 보조 지표로는 쓸 수 있지만, c035 identity/distinctive-trait gate와 직접 정렬되지는 않았다.
- c037 기준 PE/QwenVL/SigLIP2 pooled feature 모두 약한 identity-positive/negative pair도 충분히 분리하지 못했다.
- c038 기준 PE/QwenVL/SigLIP2 pooled feature와 SigLIP token feature는 duplicate crop sanity check는 통과했지만, 이것은 true character identity 해결이 아니다.
- c039 기준 same-page non-duplicate 후보만으로는 true same-character positive를 자동 확정하기 어렵다.
- c040 기준 QwenVL character-centered filter는 후보 노이즈를 줄이지만, same-character label을 자동 확정할 정도는 아니다.
- c041 기준 reviewed seed는 4 usable positive뿐이라 feature sanity probe에는 쓸 수 있지만 학습 gate로는 부족하다.
- 선화 채색은 IP-Adapter 단독 목표가 아니다. line-control/colorize control과 결합해야 한다.
- InterleaveThinker와 i1도 현 단계에서는 완성 IP-Adapter 모델이 아니다. 각각 agentic loop와 T2I recipe 참고 자료로만 사용한다.

### 불가능하다고 판단하지 않은 것

SigLIP 계열은 한 장 overfit이 성공했고, PE-style patch/PE-space/retrieval/attribute prompt를 거치며 결과가 개선됐다. 따라서 “구조적으로 불가능”이라기보다, frozen generic image encoder와 adapter-only denoising objective만으로는 고품질 일반화가 어렵다는 판단이다.

## 9. 다음 단계

1. 현재 SigLIP recipe를 실험용으로 문서화하되, c035 decision은 `not_ready`로 유지한다.
2. 다음 방향은 `agentic_reference_control_loop`를 먼저 만들고, 그 결과로 `train_stronger_encoder`를 실행할지 결정하는 것이다.
3. frozen SigLIP2 adapter-only 반복이 아니라 anime/manhwa 특화 encoder, QwenVL feature calibrator, image-encoder adaptation, 또는 i1식 data/recaption recipe를 검증한다. 단, c037 기준 pooled PE/QwenVL/SigLIP2 feature는 모두 weak identity proxy를 통과하지 못했고 c038은 duplicate sanity만 통과했으며 c039/c040 후보 mining은 아직 true same-character label을 자동 확정하지 못했다. c041 reviewed seed도 4 usable positive뿐이므로, 대규모 reviewed true same-character manifest 전까지 주 지표가 아니라 보조 관찰값으로만 둔다.
4. 자동 attribute prompt vocabulary는 유지하되, 이것만으로 identity 문제를 해결했다고 보지 않는다.
5. single-character suite를 더 큰 held-out set으로 확장하고, metric과 visual audit gate를 계속 같이 사용한다.
6. FaceID-like 목표는 별도 단계로 분리한다. same-character group mining과 애니/만화 identity encoder가 먼저 필요하다.
7. 선화 채색은 reference-control과 분리해서 EasyControl/ControlNet류 spatial colorize checkpoint를 별도 학습한 뒤 결합한다.

## 10. 근거 파일 색인

핵심 문서:

- `docs/ipadapter_reference_research.md`
- `docs/line_colorization_decision.md`
- `docs/siglip_training.md`
- `docs/siglip2_training_launch_readiness.md`
- `docs/ipadapter_agentic_reference_control_plan_ko.md`
- `eval/siglip_runtime_quality_20260612_c035_suite_v1/reference_control_audit_summary.md`
- `eval/qwenvl_metric_probe_20260612_c036_c035/report.md`
- `eval/identity_feature_probe_20260612_c037/report.md`
- `eval/strict_identity_feature_probe_20260612_c038/report.md`
- `eval/true_identity_candidate_review_20260612_c039/report.md`
- `eval/character_filtered_identity_candidates_20260612_c040/report.md`
- `eval/reviewed_identity_candidates_20260612_c041/report.md`

PE baseline:

- `eval/comfy_pe_full_contactsheet_20260610/report.md`
- `eval/comfy_pe_full_contactsheet_20260610/contact_sheet.jpg`

선화 채색:

- `eval/line_color_dataset_pair_easycontrol_ip_20260610/report.md`
- `eval/line_color_dataset_pair_easycontrol_ip_20260610/comparison_sheet.jpg`

SigLIP 주요 평가:

- `eval/siglip_runtime_quality_20260611_c008_ref03_overfit1024_identity/report.md`
- `eval/siglip_runtime_quality_20260611_c021_single_character_diagnostic/report.md`
- `eval/siglip_runtime_quality_20260611_c025_single_character_clean32_runtime/report.md`
- `eval/siglip_runtime_quality_20260611_c028_single_character_pe_space_init_runtime/report.md`
- `eval/siglip_runtime_quality_20260611_c029_single_character_pe_retrieval_runtime/report.md`
- `eval/siglip_runtime_quality_20260611_c032_attribute_prompt_runtime/report.md`
- `eval/siglip_runtime_quality_20260612_c033_auto_caption_runtime/report.md`
- `eval/siglip_runtime_quality_20260612_c034_auto_caption_vocab2_runtime/report.md`
- `eval/siglip_runtime_quality_20260612_c035_suite_v1/report.md`
- `eval/siglip_runtime_quality_20260612_c035_suite_v1/visual_audit.md`
- `docs/ipadapter_next_direction_decision_ko.md`

QwenVL 주요 평가:

- `eval/qwen3vl_embedding_probe_20260611/report.md`
- `eval/qwenvl_native_workflow_eval_20260611/report.md`
- `eval/qwenvl_runtime_quality_20260611_c001_identity128/report.md`
- `eval/qwenvl_runtime_quality_20260611_c004_calibrated_contrastive_weight_sweep/report.md`
- `eval/qwenvl_runtime_quality_20260611_c030_single_character_retrieval/report.md`
- `eval/qwenvl_runtime_quality_20260611_c031_attribute_prompt_runtime/report.md`
- `eval/qwenvl_metric_probe_20260612_c036_c035/report.md`

생성/학습 manifest:

- `training/manifests/local_color_pairs_pilot_20260610.jsonl`
- `training/manifests/local_color_self_identity8_20260611.jsonl`
- `training/manifests/local_color_self_identity128_20260611.jsonl`
- `training/manifests/local_color_single_character_identity4_20260611.jsonl`
- `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`

현재 가장 중요한 실행 레시피 근거:

- `tools/build_reference_prompt_manifest.py`
- `tools/siglip_auto_caption_eval.py`
- `tools/score_siglip_auto_caption_metrics.py`
- `workflows/anima_ipadapter_siglip_native_reference.json`
