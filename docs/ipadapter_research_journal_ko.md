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

### 7.18 2026-06-12 Reviewed seed feature probe c042

c041 reviewed seed를 pair probe manifest로 변환해 SigLIP/QwenVL/PE feature separation을 확인했다.

- 변환 도구: `tools/build_reviewed_pair_probe_manifest.py`
- 테스트: `tests/test_reviewed_pair_probe_manifest.py`
- pair manifest: `eval/reviewed_seed_feature_probe_20260612_c042/pair_probe_manifest.jsonl`
- SigLIP pooled: `eval/reviewed_seed_feature_probe_20260612_c042/siglip_pooled_report.md`
- SigLIP layer -6 token: `eval/reviewed_seed_feature_probe_20260612_c042/siglip_layer_m6_token_report.md`
- QwenVL pooled: `eval/reviewed_seed_feature_probe_20260612_c042/qwenvl_pooled_report.md`
- PE pooled: `eval/reviewed_seed_feature_probe_20260612_c042/pe_pooled_report.md`
- 보고서: `eval/reviewed_seed_feature_probe_20260612_c042/report.md`

결과:

- SigLIP pooled: margin `0.002862`, AUC `0.416667`
- QwenVL pooled: margin `0.024015`, AUC `0.666667`
- PE pooled: margin `-0.057980`, AUC `0.416667`
- SigLIP layer -6 `mean_max_token`: margin `0.043225`, AUC `0.916667`

결정: `reviewed_seed_feature_gate_not_passed`

기준 margin `>= 0.05`, AUC `>= 0.70`을 동시에 만족한 feature는 없다. 다만 SigLIP layer -6 `mean_max_token`은 AUC가 높고 margin이 기준에 가까워, 더 큰 reviewed identity set에서 우선 재검증할 후보로 유지한다.

### 7.19 2026-06-12 Broad face/upper-body identity candidate mining c043

c041/c042 seed가 너무 작았기 때문에 같은 `SG-page` 안의 non-duplicate 후보를 `160`개까지 넓히고, Qwen3-VL image-text retrieval로 양쪽 이미지가 모두 얼굴/상반신 캐릭터 crop인지 필터링했다.

- 도구: `tools/filter_face_upper_body_candidates.py`
- 테스트: `tests/test_face_upper_body_candidate_filter.py`
- raw 후보: `eval/broad_identity_candidate_mining_20260612_c043/raw_candidate_pairs.jsonl`
- QwenVL score: `eval/broad_identity_candidate_mining_20260612_c043/scored_face_candidates.jsonl`
- kept 후보: `eval/broad_identity_candidate_mining_20260612_c043/kept_face_candidates.jsonl`
- 리뷰 sheet: `eval/broad_identity_candidate_mining_20260612_c043/kept_face_candidate_sheet.jpg`
- 보고서: `eval/broad_identity_candidate_mining_20260612_c043/report.md`

결과:

- `threshold=0.0`: `152/160`개 keep, 너무 느슨함
- `threshold=0.05`: `91/160`개 keep, 리뷰 후보로는 많음
- `threshold=0.08`: `30/160`개 keep, 최종 선택
- kept set은 `22`개 SG page에 분산
- kept min side-score range: `0.080176` to `0.114322`

결정: `face_upper_body_filter_expands_review_pool_not_identity_labels`

시각 확인상 얼굴/상반신 후보를 늘리는 데는 성공했다. 하지만 남은 30쌍에도 다른 인물, partial crop, ambiguous pair가 섞여 있다. 따라서 c043은 학습용 정답 manifest가 아니라 c044 수동 라벨링 후보 풀이다.

### 7.20 2026-06-12 Reviewed face identity seed c044

c043 kept 30쌍을 보수적으로 수동 라벨링했다.

- 라벨: `eval/reviewed_face_identity_candidates_20260612_c044/manual_visual_labels.jsonl`
- reviewed manifest: `eval/reviewed_face_identity_candidates_20260612_c044/reviewed_candidate_pairs.jsonl`
- usable positives: `eval/reviewed_face_identity_candidates_20260612_c044/usable_positive_pairs.jsonl`
- negatives: `eval/reviewed_face_identity_candidates_20260612_c044/different_character_pairs.jsonl`
- noisy/unclear: `eval/reviewed_face_identity_candidates_20260612_c044/unclear_or_noisy_same_pairs.jsonl`
- 보고서: `eval/reviewed_face_identity_candidates_20260612_c044/report.md`

결과는 `same_character=12`, `different_character=15`, `unclear=3`, `positive_usable=8`이다.

결정: `reviewed_face_seed_expanded_but_still_small`

c041 usable positive 4개에서 8개로 늘었지만, 아직 adapter 학습을 시작할 규모는 아니다. feature separation 재검증에는 사용할 수 있다.

### 7.21 2026-06-12 Reviewed face seed feature probe c045

c044 seed를 positive/negative pair probe로 변환해 feature separation을 다시 확인했다.

- pair manifest: `eval/reviewed_face_seed_feature_probe_20260612_c045/pair_probe_manifest.jsonl`
- QwenVL pooled: `eval/reviewed_face_seed_feature_probe_20260612_c045/qwenvl_pooled_report.md`
- SigLIP pooled: `eval/reviewed_face_seed_feature_probe_20260612_c045/siglip_pooled_report.md`
- PE pooled: `eval/reviewed_face_seed_feature_probe_20260612_c045/pe_pooled_report.md`
- SigLIP layer -6 token: `eval/reviewed_face_seed_feature_probe_20260612_c045/siglip_layer_m6_token_report.md`
- 보고서: `eval/reviewed_face_seed_feature_probe_20260612_c045/report.md`

결과:

- QwenVL pooled: margin `0.066209`, AUC `0.791667`
- SigLIP pooled: margin `0.015873`, AUC `0.650000`
- PE pooled: margin `0.044773`, AUC `0.783333`
- SigLIP layer -6 `mean_max_token`: margin `0.028728`, AUC `0.708333`

결정: `qwenvl_pooled_passes_small_reviewed_identity_proxy`

QwenVL pooled가 처음으로 reviewed identity proxy gate인 margin `>= 0.05`, AUC `>= 0.70`을 통과했다. 단, positive 8개/negative 15개의 작은 seed이고 일부 캐릭터에 편중되어 있으므로 생성 품질 통과나 adapter 학습 시작으로 해석하지 않는다. 다음 loop는 QwenVL pooled를 후보 ranking metric으로 사용해 더 큰 reviewed set을 만드는 것이다.

### 7.22 2026-06-12 QwenVL-ranked identity candidates c046

c045에서 통과한 QwenVL pooled를 ranking metric으로 사용해 전체 same-page 후보를 다시 정렬했다.

- ranking 도구: `tools/rank_identity_candidate_pairs.py`
- 테스트: `tests/test_rank_identity_candidate_pairs.py`
- raw 후보: `eval/qwenvl_ranked_identity_candidates_20260612_c046/raw_candidate_pairs.jsonl`
- face-filtered 후보: `eval/qwenvl_ranked_identity_candidates_20260612_c046/kept_face_candidates.jsonl`
- QwenVL ranked 후보: `eval/qwenvl_ranked_identity_candidates_20260612_c046/qwenvl_ranked_face_candidates.jsonl`
- top40 후보: `eval/qwenvl_ranked_identity_candidates_20260612_c046/qwenvl_top40_face_candidates.jsonl`
- 보고서: `eval/qwenvl_ranked_identity_candidates_20260612_c046/report.md`

결과:

- raw same-page non-duplicate 후보: `372`
- face/upper-body threshold `0.08` 통과: `65`
- top40은 `27`개 SG page에 분산
- top10 similarity range: `0.9155` to `0.9632`
- top20 lower bound: `0.8801`

결정: `qwenvl_ranking_improves_candidate_precision_top20`

시각 확인상 top10은 대부분 clean same-character pair이고, top20까지는 리뷰 효율이 좋다. rank 21 이후부터는 다른 인물, 다인물 panel, 뒤통수/partial crop 노이즈가 늘어난다. 따라서 다음 loop는 top20을 우선 수동 라벨링하고, 필요할 때 top40 하위권을 추가로 본다.

### 7.23 2026-06-12 QwenVL top20 reviewed identity c047

c046 top20 후보를 수동 라벨링했다.

- 입력 후보: `eval/qwenvl_top20_reviewed_identity_20260612_c047/top20_review_candidates.jsonl`
- 라벨: `eval/qwenvl_top20_reviewed_identity_20260612_c047/manual_visual_labels.jsonl`
- reviewed manifest: `eval/qwenvl_top20_reviewed_identity_20260612_c047/reviewed_candidate_pairs.jsonl`
- 보고서: `eval/qwenvl_top20_reviewed_identity_20260612_c047/report.md`

결과는 `same_character=18`, `different_character=0`, `unclear=2`, `positive_usable=14`다.

결정: `qwenvl_top20_review_precision_good`

QwenVL ranking은 positive mining 효율을 크게 올렸다. 단, top20은 positive 위주라 hard negative가 부족하므로 c044 negative와 결합해 feature gate를 다시 확인한다.

### 7.24 2026-06-12 Combined reviewed seed QwenVL gate c048

c044 reviewed seed와 c047 top20 reviewed seed를 pair_id 기준으로 결합하고 QwenVL pooled feature gate를 반복했다.

- combined reviewed seed: `eval/qwenvl_combined_seed_feature_probe_20260612_c048/combined_reviewed_candidate_pairs.jsonl`
- pair probe manifest: `eval/qwenvl_combined_seed_feature_probe_20260612_c048/pair_probe_manifest.jsonl`
- QwenVL pooled score: `eval/qwenvl_combined_seed_feature_probe_20260612_c048/qwenvl_pooled_scores.json`
- 보고서: `eval/qwenvl_combined_seed_feature_probe_20260612_c048/report.md`

결과:

- combined reviewed rows: `44`
- feature-probe rows: `33`
- positive pairs: `18`
- negative pairs: `15`
- QwenVL pooled margin: `0.087629`
- QwenVL pooled AUC: `0.907407`
- midpoint accuracy: `0.818182`

결정: `qwenvl_pooled_identity_gate_stable_on_combined_seed`

QwenVL pooled가 c045보다 큰 reviewed seed에서도 안정적으로 통과했다. 이제 QwenVL pooled는 후보 ranking/gating metric으로 승격한다. 다만 이것은 여전히 feature gate이며, 생성 품질 통과는 아니다.

### 7.25 2026-06-12 QwenVL rank21-40 reviewed identity c049

c046 QwenVL rank 21-40을 수동 라벨링했다.

- 입력 후보: `eval/qwenvl_rank21_40_reviewed_identity_20260612_c049/rank21_40_review_candidates.jsonl`
- 라벨: `eval/qwenvl_rank21_40_reviewed_identity_20260612_c049/manual_visual_labels.jsonl`
- reviewed manifest: `eval/qwenvl_rank21_40_reviewed_identity_20260612_c049/reviewed_candidate_pairs.jsonl`
- 보고서: `eval/qwenvl_rank21_40_reviewed_identity_20260612_c049/report.md`

결과는 `same_character=9`, `different_character=9`, `unclear=2`, `positive_usable=3`이다.

결정: `qwenvl_rank21_40_precision_drops_adds_hard_negatives`

rank 21-40은 top20보다 노이즈가 뚜렷하게 늘었다. clean positive 확장보다는 hard negative와 noisy same-character 사례를 추가하는 데 더 유용하다.

### 7.26 2026-06-12 Combined rank40 QwenVL gate c050

c048 combined seed에 c049 reviewed rows를 결합해 QwenVL pooled gate를 다시 확인했다.

- combined reviewed seed: `eval/qwenvl_combined_rank40_feature_probe_20260612_c050/combined_reviewed_candidate_pairs.jsonl`
- pair probe manifest: `eval/qwenvl_combined_rank40_feature_probe_20260612_c050/pair_probe_manifest.jsonl`
- QwenVL pooled score: `eval/qwenvl_combined_rank40_feature_probe_20260612_c050/qwenvl_pooled_scores.json`
- QwenVL pooled report: `eval/qwenvl_combined_rank40_feature_probe_20260612_c050/qwenvl_pooled_report.md`
- 보고서: `eval/qwenvl_combined_rank40_feature_probe_20260612_c050/report.md`

결과:

- combined reviewed rows: `52`
- feature-probe rows: `36`
- positive pairs: `19`
- negative pairs: `17`
- QwenVL pooled margin: `0.081599`
- QwenVL pooled AUC: `0.900929`
- midpoint accuracy: `0.861111`

결정: `qwenvl_pooled_identity_gate_stable_on_rank40_combined_seed`

rank21-40의 노이즈를 넣어도 QwenVL pooled gate는 안정적으로 통과했다. 이제 더 큰 reviewed set을 만들 때 QwenVL pooled를 primary ranking/gating metric으로 사용하는 판단을 유지한다.

### 7.27 2026-06-12 QwenVL diverse reviewed identity c051

c050 이후에는 단순히 rank를 더 내리지 않고, 새 `SG-*` page를 우선하는 diverse sampler를 만들었다.

- selector: `tools/select_diverse_review_candidates.py`
- CLI: `tools/select_diverse_review_candidates_cli.py`
- 테스트: `tests/test_select_diverse_review_candidates.py`
- 후보 보고서: `eval/qwenvl_diverse_identity_candidates_20260612_c051/report.md`

조건은 face threshold `0.07`, QwenVL similarity `>= 0.78`, page당 최대 `1`개, target `32`개다. 결과는 `32`개 모두 기존 c050 reviewed set에 없던 새 SG page였다.

수동 라벨 결과:

- reviewed rows: `32`
- same_character: `17`
- different_character: `12`
- unclear: `3`
- positive_usable: `10`

결정: `qwenvl_diverse_sampling_improves_seed_diversity`

c051은 c047 top20보다 positive precision은 낮지만 c049 rank21-40보다 낫고, 무엇보다 새 SG page를 크게 늘렸다. 따라서 larger reviewed identity seed를 만들기 위한 다양성 확장으로 채택한다.

### 7.28 2026-06-12 Combined diverse QwenVL gate c052

c050 combined seed와 c051 diverse reviewed rows를 결합해 QwenVL pooled gate를 다시 확인했다.

- combined reviewed seed: `eval/qwenvl_combined_diverse_feature_probe_20260612_c052/combined_reviewed_candidate_pairs.jsonl`
- pair probe manifest: `eval/qwenvl_combined_diverse_feature_probe_20260612_c052/pair_probe_manifest.jsonl`
- QwenVL pooled score: `eval/qwenvl_combined_diverse_feature_probe_20260612_c052/qwenvl_pooled_scores.json`
- QwenVL pooled report: `eval/qwenvl_combined_diverse_feature_probe_20260612_c052/qwenvl_pooled_report.md`
- 보고서: `eval/qwenvl_combined_diverse_feature_probe_20260612_c052/report.md`

결과:

- combined reviewed rows: `84`
- feature-probe rows: `58`
- positive pairs: `29`
- negative pairs: `29`
- QwenVL pooled margin: `0.072203`
- QwenVL pooled AUC: `0.913199`
- midpoint accuracy: `0.862069`

결정: `qwenvl_pooled_identity_gate_stable_on_diverse_seed`

다양성 확장 후에도 QwenVL pooled gate는 안정적으로 통과했다. 이제 데이터/feature gate 기준으로는 adapter 또는 metric-head의 bounded training pilot을 시작할 수 있는 seed가 만들어졌다. 단, 이것은 아직 생성 품질 통과가 아니다.

### 7.29 2026-06-12 QwenVL c052 bounded training c053

c052의 positive usable pair만 사용해 QwenVL adapter continuation을 한 번 실행했다.

- manifest: `training/manifests/c052_positive_identity_pairs_20260612.jsonl`
- manifest summary: `training/manifests/c052_positive_identity_pairs_20260612.summary.json`
- source positives: `eval/qwenvl_combined_diverse_feature_probe_20260612_c052/usable_positive_pairs.jsonl`
- report: `eval/qwenvl_c052_bounded_training_20260612_c053/report.md`
- log: `eval/qwenvl_c052_bounded_training_20260612_c053/train_stdout.txt`
- summary: `eval/qwenvl_c052_bounded_training_20260612_c053/summary.json`
- local checkpoint: `checkpoints/anima_qwenvl_ip_adapter_c052_identity_retrieval_0064_20260612.safetensors`

결과:

- positive pairs: `29`
- bidirectional training rows: `58`
- steps: `64`
- init checkpoint: `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- final_loss: `0.2033316642`
- finite_loss: `true`
- checkpoint loadable: `true`
- PE checkpoint rejected: `true`

결정: `qwenvl_c052_bounded_training_smoke_passed_generation_gate_pending`

이 실험은 학습 surface와 checkpoint compatibility가 정상임을 확인했다. 그러나 아직 생성 품질을 증명하지 않았다. 새 checkpoint는 `.gitignore`의 `checkpoints/*qwenvl*.safetensors` 정책 때문에 local artifact로 유지하고, 커밋에는 manifest, 로그, summary/report, import 회귀 테스트만 포함한다. 다음 단계는 이 checkpoint를 ComfyUI가 볼 수 있게 노출한 뒤 c035-style single-character generation/contact-sheet gate를 실행하는 것이다.

### 7.30 2026-06-12 QwenVL c052 generation smoke gate c054

c053 checkpoint를 isolated ComfyUI API 서버에서 실제 생성 평가했다. 비교 열은 `reference / no_ip / qwen_prev_retrieval_w14 / qwen_c052_w1 / qwen_c052_w14`로 구성했다.

- contact sheet: `eval/qwenvl_c052_generation_gate_20260612_c054/contact_sheet.jpg`
- summary: `eval/qwenvl_c052_generation_gate_20260612_c054/summary.json`
- PE metric: `eval/qwenvl_c052_generation_gate_20260612_c054/pe_similarity_metrics.json`
- QwenVL metric: `eval/qwenvl_c052_generation_gate_20260612_c054/qwenvl_similarity_metrics.json`
- visual audit: `eval/qwenvl_c052_generation_gate_20260612_c054/visual_audit.md`
- report: `eval/qwenvl_c052_generation_gate_20260612_c054/report.md`

수치:

| variant | PE mean uplift | PE improved | QwenVL mean uplift | QwenVL improved |
|---|---:|---:|---:|---:|
| `qwen_prev_retrieval_w14` | `+0.0983` | `0.875` | `+0.0377` | `0.875` |
| `qwen_c052_w1` | `+0.0377` | `0.625` | `+0.0174` | `0.500` |
| `qwen_c052_w14` | `+0.0394` | `0.625` | `+0.0231` | `0.625` |

결정: `qwen_c052_partial_visual_improvement_metric_regression_not_quality_pass`

c053은 실제 reference signal을 쓰며 `heldout00`, `heldout02`, `heldout07`처럼 어려운 특수 trait에서 눈에 띄는 개선이 있다. 특히 노승의 bald head/white beard/red beads, green demon의 green skin/red eye가 살아났다. 하지만 aggregate PE/QwenVL pooled metric은 이전 retrieval checkpoint가 더 높고, `train14` old exaggerated bearded face와 `heldout05` shouting profile에서 여전히 약하다. 따라서 c053은 “품질 통과”가 아니라 “특수 trait 개선 신호가 있는 metric-regressed branch”로 분류한다.

### 7.31 2026-06-12 QwenVL mixed continuation c055

c054에서 확인한 metric regression을 줄이기 위해 c052 positive seed만 쓰지 않고, 기존 clean32 train self-pair 32개와 c052 reviewed positive identity rows 58개를 섞어 QwenVL adapter continuation을 다시 실행했다.

- mixed manifest: `training/manifests/c055_qwenvl_mixed_clean32_c052_positive_20260612.jsonl`
- manifest summary: `training/manifests/c055_qwenvl_mixed_clean32_c052_positive_20260612.summary.json`
- report: `eval/qwenvl_c055_mixed_training_20260612/report.md`
- log: `eval/qwenvl_c055_mixed_training_20260612/train_stdout.txt`
- summary: `eval/qwenvl_c055_mixed_training_20260612/summary.json`
- local checkpoint: `checkpoints/anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors`

설정:

- init checkpoint: `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- rows: `90`
- steps: `64`
- learning rate: `3e-6`
- contrastive weight: `0.45`
- retrieval weight: `0.15`
- heldout rows used: `0`

결과:

- first_loss: `0.2682350278`
- final_loss: `0.1592893749`
- finite_loss: `true`
- checkpoint loadable: `true`
- PE checkpoint rejected: `true`

결정: `qwenvl_c055_mixed_training_smoke_passed_generation_gate_pending`

c055는 학습/로드 gate를 통과했다. 다만 final loss가 c053보다 낮아도 이것만으로 reference-control 품질을 말할 수 없다. 다음 단계는 ComfyUI generation gate에서 `qwen_prev_retrieval_w14`, `qwen_c052_w14`, `qwen_c055_w1/w14`를 같은 단일 캐릭터 suite에 놓고 비교하는 것이다.

### 7.32 2026-06-12 QwenVL c055 generation gate c056

c055 mixed checkpoint가 실제 생성에서 c052보다 좋아졌는지, 그리고 이전 retrieval checkpoint를 넘을 수 있는지 확인하기 위해 같은 8개 single-character suite에서 ComfyUI API generation gate를 실행했다.

- output: `eval/qwenvl_c055_generation_gate_20260612_c056/`
- contact sheet: `eval/qwenvl_c055_generation_gate_20260612_c056/contact_sheet.jpg`
- report: `eval/qwenvl_c055_generation_gate_20260612_c056/report.md`
- summary: `eval/qwenvl_c055_generation_gate_20260612_c056/summary.json`
- PE metric: `eval/qwenvl_c055_generation_gate_20260612_c056/pe_similarity_metrics.json`
- QwenVL metric: `eval/qwenvl_c055_generation_gate_20260612_c056/qwenvl_similarity_metrics.json`
- visual audit: `eval/qwenvl_c055_generation_gate_20260612_c056/visual_audit.md`

비교 컬럼:

- `reference`
- `no_ip`
- `qwen_prev_retrieval_w14`
- `qwen_c052_w14`
- `qwen_c055_w1`
- `qwen_c055_w14`

수치:

| variant | PE mean uplift | PE improved | QwenVL mean uplift | QwenVL improved |
|---|---:|---:|---:|---:|
| `qwen_prev_retrieval_w14` | `+0.0983` | `0.875` | `+0.0377` | `0.875` |
| `qwen_c052_w14` | `+0.0394` | `0.625` | `+0.0231` | `0.625` |
| `qwen_c055_w1` | `+0.0502` | `0.750` | `+0.0257` | `0.750` |
| `qwen_c055_w14` | `+0.0460` | `0.750` | `+0.0339` | `0.875` |

검증:

- generated PNG: `40`
- blank image: `0`
- contact sheet size: `1588x3304`
- ComfyUI isolated server는 생성 후 종료했고 `127.0.0.1:8116` listen process가 남지 않았다.

결정: `qwen_c055_improves_c052_not_quality_pass_prev_retrieval_still_best`

c055는 c052보다 확실히 나아졌다. 특히 `train14`, `heldout00`, `heldout05`, `heldout07`에서 reference trait가 더 살아났고, QwenVL improved rate는 c055 w1.4가 이전 retrieval과 같은 `0.875`까지 올라왔다. 하지만 PE mean uplift는 이전 retrieval이 여전히 크게 앞서며, QwenVL mean uplift도 이전 retrieval이 근소하게 더 높다. `train00`, `train07`, `train23`에서는 pose/action, page framing, fan 같은 세부 요소가 아직 prompt-driven으로 빠진다.

따라서 c055는 “continuation 방향이 맞다는 증거”이지 “바로 믿고 쓰는 고품질 checkpoint”는 아니다. 다음 루프는 추가 학습 전에 c055의 weight/blend runtime gate를 먼저 확인한다. c055 w1.4가 QwenVL improved rate를 거의 따라잡았으므로, 이전 retrieval과 c055의 장점을 조합하는 낮은 비용 실험을 먼저 하는 것이 합리적이다.

### 7.33 2026-06-12 QwenVL runtime weight/blend gate c057

c056에서 c055가 c052보다 좋아졌지만 이전 retrieval checkpoint를 aggregate로 넘지 못했기 때문에, 추가 학습 전에 runtime 조합만으로 개선할 수 있는지 확인했다. 같은 8개 single-character suite에서 c055 단독 낮은 weight와 previous retrieval + c055 이중 적용을 비교했다.

- output: `eval/qwenvl_c055_runtime_blend_gate_20260612_c057/`
- contact sheet: `eval/qwenvl_c055_runtime_blend_gate_20260612_c057/contact_sheet.jpg`
- report: `eval/qwenvl_c055_runtime_blend_gate_20260612_c057/report.md`
- summary: `eval/qwenvl_c055_runtime_blend_gate_20260612_c057/summary.json`
- PE metric: `eval/qwenvl_c055_runtime_blend_gate_20260612_c057/pe_similarity_metrics.json`
- QwenVL metric: `eval/qwenvl_c055_runtime_blend_gate_20260612_c057/qwenvl_similarity_metrics.json`
- visual audit: `eval/qwenvl_c055_runtime_blend_gate_20260612_c057/visual_audit.md`

비교 recipe:

- `no_ip`
- `prev_w14`
- `c055_w06`
- `c055_w08`
- `c055_w12`
- `blend_prev10_c05506`
- `blend_prev14_c05504`

수치:

| variant | PE mean uplift | PE improved | QwenVL mean uplift | QwenVL improved |
|---|---:|---:|---:|---:|
| `blend_prev14_c05504` | `+0.1064` | `0.875` | `+0.0375` | `0.875` |
| `prev_w14` | `+0.0983` | `0.875` | `+0.0377` | `0.875` |
| `blend_prev10_c05506` | `+0.0768` | `0.750` | `+0.0357` | `0.875` |
| `c055_w12` | `+0.0583` | `0.750` | `+0.0289` | `0.750` |
| `c055_w06` | `+0.0543` | `0.875` | `+0.0096` | `0.500` |
| `c055_w08` | `+0.0282` | `0.625` | `+0.0136` | `0.750` |

검증:

- generated PNG: `56`
- blank image: `0`
- contact sheet size: `2108x3304`
- ComfyUI isolated server는 생성 후 종료했고 `127.0.0.1:8116` listen process가 남지 않았다.

결정: `runtime_blend_prev14_c05504_best_so_far_larger_gate_required`

`blend_prev14_c05504`는 지금까지의 QwenVL runtime recipe 중 가장 좋은 단일 후보로 본다. PE mean uplift는 이전 retrieval `+0.0983`을 넘어 `+0.1064`가 되었고, QwenVL mean uplift도 이전 retrieval `+0.0377`과 거의 같은 `+0.0375`다. 시각적으로도 heldout demon/monk/shouting profile 쪽 안정성이 좋아졌다.

다만 이것은 최종 고퀄 reference-control pass가 아니라 “현재 best runtime candidate”다. `train07`은 여전히 generic close-up이고, `train23`의 fan, `train00`의 정확한 손/동작, speech bubble/page-specific detail은 안정적으로 복구되지 않는다. 또 `train14`와 `train23`에서는 `c055_w06` 같은 낮은 c055 weight가 더 나은 특수 trait를 보인다. 따라서 다음 루프는 `blend_prev14_c05504`를 UI workflow/current recipe로 고정하면서 더 큰 heldout gate를 돌리거나, 이 blend를 단일 checkpoint로 distill/continuation하는 방향이다.

### 7.34 2026-06-12 QwenVL larger runtime blend gate c058

목표: c057에서 best runtime candidate가 된 `blend_prev14_c05504`를 8장 소규모가 아니라 clean32 train 전체와 heldout8 전체, 총 40개 단일 캐릭터 샘플에서 검증한다. 비교군은 `no_ip`, 이전 retrieval checkpoint `prev_w14`, 그리고 runtime blend `blend_prev14_c05504`다.

산출물:

- output: `eval/qwenvl_c055_larger_blend_gate_20260612_c058/`
- train contact sheet: `eval/qwenvl_c055_larger_blend_gate_20260612_c058/contact_sheet_train.jpg`
- heldout contact sheet: `eval/qwenvl_c055_larger_blend_gate_20260612_c058/contact_sheet_heldout.jpg`
- report: `eval/qwenvl_c055_larger_blend_gate_20260612_c058/report.md`
- summary: `eval/qwenvl_c055_larger_blend_gate_20260612_c058/summary.json`
- PE metric: `eval/qwenvl_c055_larger_blend_gate_20260612_c058/pe_similarity_metrics.json`
- QwenVL metric: `eval/qwenvl_c055_larger_blend_gate_20260612_c058/qwenvl_similarity_metrics.json`
- visual audit: `eval/qwenvl_c055_larger_blend_gate_20260612_c058/visual_audit.md`

비교 recipe:

- `no_ip`
- `prev_w14`
- `blend_prev14_c05504`

수치:

| variant | PE mean uplift | PE improved | QwenVL mean uplift | QwenVL improved |
|---|---:|---:|---:|---:|
| `blend_prev14_c05504` | `+0.0496` | `0.725` | `+0.0416` | `0.800` |
| `prev_w14` | `+0.0292` | `0.750` | `+0.0362` | `0.725` |

검증:

- generated PNG: `120`
- blank image: `0`
- train contact sheet size: `1068x12040`
- heldout contact sheet size: `1068x3304`
- ComfyUI isolated server는 생성 후 종료했고 `127.0.0.1:8116` listen process가 남지 않았다.

시각 감사:

- broad character cue, hair/costume/color cue, bald/old/headwear cue는 `no_ip`보다 강해졌고, 평균 metric도 `prev_w14`를 넘었다.
- 그러나 exact pose/crop, speech bubble, panel layout, fan/hand prop, non-human silhouette는 아직 안정적이지 않다.
- `heldout07`은 구조적 실패다. reference는 green demon side-head close-up인데, `prev_w14`/`blend_prev14_c05504`는 full-body dark demon/assassin으로 이동한다.

결정: `best_runtime_candidate_not_final_quality_pass_distillation_or_training_next`

c058 기준 `blend_prev14_c05504`는 현재까지 가장 강한 QwenVL runtime recipe지만, “바로 믿고 쓰는 고퀄 reference-control 모델”은 아니다. 단순 runtime weight sweep만 반복해도 한계가 보이므로 다음 루프는 `prev_w14 + c055_w04`를 단일 checkpoint로 distill하거나, c058 failure class를 반영한 continuation/encoder adaptation으로 가야 한다.

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
- c042 기준 c041 seed에서는 어떤 raw feature도 identity feature gate를 통과하지 못했다. SigLIP layer -6 `mean_max_token`만 더 큰 라벨셋에서 재검증할 후보로 남긴다.
- c043 기준 QwenVL face/upper-body filter는 후보를 30쌍으로 확장했지만, same-character label을 자동 확정하지는 못한다.
- c044/c045 기준 QwenVL pooled는 작은 reviewed identity proxy gate를 통과했다. 그러나 seed가 작고 편중되어 있어 adapter 학습이 아니라 더 큰 후보 ranking/라벨링에 먼저 사용한다.
- c046 기준 QwenVL ranking은 top20의 same-character 후보 밀도를 높인다. 하지만 top40 전체를 자동 positive로 승격할 정도는 아니다.
- c047/c048 기준 QwenVL pooled identity gate는 combined seed에서도 안정적으로 통과했다. 다음 단계는 이 metric을 사용해 더 큰 reviewed identity set을 만들고, 그 다음 adapter/metric-head 학습 여부를 결정하는 것이다.
- c049/c050 기준 top40 하위권은 노이즈가 크지만 QwenVL pooled gate 자체는 rank40 combined seed에서도 안정적이다.
- c051/c052 기준 새 SG page를 우선하는 diverse seed에서도 QwenVL pooled gate는 안정적이다. 이제 다음은 bounded training pilot과 c035-style generation gate다.
- c053 기준 c052 positive seed로 QwenVL bounded continuation은 정상 종료했고 checkpoint도 loadable이다. 다만 이것은 학습/로드 gate 통과일 뿐, 아직 생성 품질 gate 통과가 아니다.
- c054 기준 c053은 일부 특수 trait에서 시각적으로 좋아졌지만, 이전 retrieval checkpoint보다 aggregate metric이 낮아 quality pass는 아니다.
- c055 기준 clean32 train rows와 c052 positive rows를 섞은 QwenVL mixed continuation은 정상 종료했고 checkpoint도 loadable이다.
- c056 기준 c055는 c052 대비 시각/metric 모두 개선됐지만 이전 retrieval checkpoint를 aggregate로 넘지 못했다. quality pass가 아니라 c057 weight/blend runtime gate로 이어간다.
- c057 기준 `blend_prev14_c05504`는 현재 최고 runtime recipe다. PE metric은 이전 retrieval을 넘고 QwenVL metric은 거의 동률이지만, pose/prop/detail 안정성이 아직 부족하므로 최종 quality pass로 보지 않는다.
- c058 기준 `blend_prev14_c05504`는 40-sample larger gate에서 평균 metric best가 되었지만, heldout visual audit에서 pose/crop, speech bubble, prop, non-human silhouette 실패가 남아 final quality pass는 아니다. 다음은 distillation 또는 failure-focused continuation이다.
- c059 기준 단순 parameter-space checkpoint merge는 실행 가능하지만 final quality pass는 아니다. `merge_a040_w14`는 QwenVL metric에서 runtime blend와 동률급이나 PE metric과 시각 감사에서 runtime blend를 대체하지 못했다.
- 선화 채색은 IP-Adapter 단독 목표가 아니다. line-control/colorize control과 결합해야 한다.
- InterleaveThinker와 i1도 현 단계에서는 완성 IP-Adapter 모델이 아니다. 각각 agentic loop와 T2I recipe 참고 자료로만 사용한다.

### 불가능하다고 판단하지 않은 것

SigLIP 계열은 한 장 overfit이 성공했고, PE-style patch/PE-space/retrieval/attribute prompt를 거치며 결과가 개선됐다. 따라서 “구조적으로 불가능”이라기보다, frozen generic image encoder와 adapter-only denoising objective만으로는 고품질 일반화가 어렵다는 판단이다.

## 9. 다음 단계

1. 현재 SigLIP recipe를 실험용으로 문서화하되, c035 decision은 `not_ready`로 유지한다.
2. 다음 방향은 `agentic_reference_control_loop`를 먼저 만들고, 그 결과로 `train_stronger_encoder`를 실행할지 결정하는 것이다.
3. frozen SigLIP2 adapter-only 반복이 아니라 anime/manhwa 특화 encoder, QwenVL feature calibrator, image-encoder adaptation, 또는 i1식 data/recaption recipe를 검증한다. 단, c037 기준 pooled PE/QwenVL/SigLIP2 feature는 모두 weak identity proxy를 통과하지 못했고 c038은 duplicate sanity만 통과했으며 c039/c040 후보 mining은 아직 true same-character label을 자동 확정하지 못했다. c041/c042 reviewed seed도 너무 작고 raw feature gate를 통과하지 못했다. c043-c052 결과 QwenVL pooled가 reviewed identity ranking/gating metric으로 가장 유효하고 diverse seed에서도 안정적이므로, c053에서 bounded QwenVL continuation을 실행했다. c054 생성 gate에서는 일부 특수 trait 개선은 확인했지만 metric regression이 있었다. c055는 clean32와 c052 positives를 섞은 metric-preserving mixed continuation이고, c056에서는 c052보다 개선됐지만 이전 retrieval을 넘지 못했다. c057에서는 previous retrieval + c055 runtime blend가 현재 최고 후보가 되었고, c058 larger gate에서는 평균 metric best까지 확인했다. c059 단순 checkpoint merge는 QwenVL metric 동률에도 PE/visual에서 pass하지 못했으므로, 다음은 단순 merge가 아니라 failure-focused continuation 또는 stronger encoder/feature adaptation이다.
4. 자동 attribute prompt vocabulary는 유지하되, 이것만으로 identity 문제를 해결했다고 보지 않는다.
5. single-character suite를 더 큰 held-out set으로 확장하고, metric과 visual audit gate를 계속 같이 사용한다.
6. FaceID-like 목표는 별도 단계로 분리한다. same-character group mining과 애니/만화 identity encoder가 먼저 필요하다.
7. 선화 채색은 reference-control과 분리해서 EasyControl/ControlNet류 spatial colorize checkpoint를 별도 학습한 뒤 결합한다.

## 10. 2026-06-12 c059 QwenVL checkpoint merge gate

### 목적

c058에서 가장 좋은 runtime recipe는 `blend_prev14_c05504`, 즉 이전 retrieval checkpoint를 weight `1.4`로 적용한 뒤 c055 mixed checkpoint를 weight `0.4`로 약하게 추가 적용하는 방식이었다. 이 방식은 평균 metric은 좋지만 ComfyUI workflow에서 두 adapter apply node가 필요하다. c059는 이 runtime blend를 하나의 단일 QwenVL adapter checkpoint로 근사할 수 있는지 확인하기 위해 시작했다.

### 개발한 것

- `tools/merge_qwenvl_checkpoints.py`
  - 같은 key/shape를 가진 QwenVL adapter checkpoint 두 개를 parameter interpolation으로 병합한다.
  - floating tensor는 `(1 - alpha) * base + alpha * update`로 병합한다.
  - non-floating tensor는 값이 같을 때만 보존하고, 다르면 실패시킨다.
- `tests/test_qwenvl_checkpoint_merge.py`
  - alpha 범위, key mismatch, shape mismatch, non-floating tensor guard, 파일 출력 summary를 검증한다.
- 생성한 checkpoint:
  - `checkpoints/anima_qwenvl_ip_adapter_c059_merge_prev_c055_a0250.safetensors`
  - `checkpoints/anima_qwenvl_ip_adapter_c059_merge_prev_c055_a0400.safetensors`

### 데이터셋과 평가 방식

- c058과 같은 `local_color_single_character_clean32_20260611.jsonl` train 32개와 `local_color_single_character_clean32_heldout8_20260611.jsonl` heldout 8개를 사용했다.
- 비교 열:
  - `no_ip`
  - `prev_w14`
  - `blend_prev14_c05504`
  - `merge_a025_w14`
  - `merge_a040_w14`
- ComfyUI API 생성 결과는 총 40샘플 × 5variant = 200 PNG다.
- 검증 관점:
  - ComfyUI loader model selection에 c059 checkpoint가 보이는지
  - 생성 PNG가 모두 nonblank인지
  - PE/QwenVL similarity metric에서 no-IP 대비 uplift가 있는지
  - contact sheet에서 heldout failure class가 해결되는지

### 결과

- 생성: 200 PNG, blank 0.
- contact sheets:
  - `eval/qwenvl_c059_checkpoint_merge_gate_20260612/contact_sheet_train.jpg`
  - `eval/qwenvl_c059_checkpoint_merge_gate_20260612/contact_sheet_heldout.jpg`
- PE metric:
  - `blend_prev14_c05504`: mean uplift `+0.049596`, improved rate `0.725`
  - `prev_w14`: mean uplift `+0.029240`, improved rate `0.750`
  - `merge_a025_w14`: mean uplift `+0.025643`, improved rate `0.600`
  - `merge_a040_w14`: mean uplift `+0.025837`, improved rate `0.475`
- QwenVL metric:
  - `merge_a040_w14`: mean uplift `+0.041614`, improved rate `0.800`
  - `blend_prev14_c05504`: mean uplift `+0.041589`, improved rate `0.800`
  - `merge_a025_w14`: mean uplift `+0.038867`, improved rate `0.800`
  - `prev_w14`: mean uplift `+0.036187`, improved rate `0.725`

### 판단

`merge_a040_w14`는 QwenVL metric만 보면 runtime blend와 거의 동률이다. 하지만 PE metric은 runtime blend보다 크게 낮고, contact sheet에서도 c058의 핵심 실패 클래스가 해결되지 않았다. 특히 `heldout06`은 관모/수염 cue는 잡지만 표정과 crop이 악역 템플릿으로 밀리고, `heldout07`은 초록 괴물 측면 close-up이 dark demon/assassin 전신 이미지로 무너진다.

따라서 c059 decision은 `single_checkpoint_merge_not_quality_pass_runtime_blend_remains_best`다.

### 다음 결정

단순 parameter-space checkpoint merge는 더 파지 않는다. 다음 루프는 failure-focused continuation 또는 stronger encoder/feature adaptation으로 넘어가야 한다. 학습/평가에서 직접 압박해야 할 실패 클래스는 pose/crop, speech bubble, hand/fan prop, non-human silhouette이다.

## 11. 2026-06-12 c060 QwenVL failure-focused continuation

### 왜 시작했나

c058에서 가장 강한 후보는 `prev_w14 + c055_w04` runtime blend였고, c059 단일 checkpoint merge는 그 runtime blend를 대체하지 못했다. 그래서 c060은 단순 merge 대신 c058/c059에서 반복적으로 무너진 실패류를 학습 데이터 쪽에서 더 압박하는 실험으로 진행했다.

목표는 다음과 같았다.

1. heldout을 학습에 쓰지 않고 clean32 train과 c052 reviewed positive만 사용한다.
2. pose/crop, speech bubble, hand/fan prop, non-human/special silhouette 같은 실패류를 prompt attribute로 찾아 train rows를 반복 노출한다.
3. 이전 QwenVL single-character retrieval checkpoint에서 bounded continuation을 진행한다.
4. ComfyUI API에서 `no_ip`, `prev_w14`, 기존 best runtime blend `blend_prev14_c05504`, 새 `c060_w14`를 같은 seed/prompt/reference로 비교한다.

### 데이터셋과 manifest

- clean train source: `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- positive source: `training/manifests/c052_positive_identity_pairs_20260612.jsonl`
- failure source: `eval/qwenvl_c055_larger_blend_gate_20260612_c058/summary.json`
- output manifest: `training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.jsonl`
- output summary: `training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.summary.json`

manifest summary:

- clean32 rows: `32`
- c052 positive rows: `58`
- failure repeated rows: `64`
- total rows: `154`
- heldout rows used: `0`
- repeat per failure row: `2`

실제로는 clean32 32개가 모두 하나 이상의 failure keyword에 걸려서 clean32가 3회 노출되고 c052 positive가 1회 추가되는 형태가 되었다. 완전히 좁은 failure-only 데이터는 아니지만 heldout 누수 없이 실패류를 더 강하게 반복하는 bounded continuation으로는 유효하다.

### 학습

- init checkpoint: `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- output checkpoint: `checkpoints/anima_qwenvl_ip_adapter_c060_failure_focused_retrieval_0096_20260612.safetensors`
- train report: `eval/qwenvl_c060_failure_focused_training_20260612/report.md`
- train summary: `eval/qwenvl_c060_failure_focused_training_20260612/summary.json`
- train stdout: `eval/qwenvl_c060_failure_focused_training_20260612/train_stdout.txt`

학습 설정:

- steps: `96`
- rows loaded: `154`
- resolution: `256`
- lr: `2e-6`
- contrastive weight: `0.40`
- retrieval weight: `0.25`
- seed: `20260660`

학습 결과:

- final loss: `0.2131887674`
- mean loss: `0.2391075663`
- finite loss: `true`
- checkpoint loadable: `true`
- PE checkpoint rejected: `true`

### ComfyUI generation gate

- output: `eval/qwenvl_c060_failure_focused_gate_20260612/`
- train contact sheet: `eval/qwenvl_c060_failure_focused_gate_20260612/contact_sheet_train.jpg`
- heldout contact sheet: `eval/qwenvl_c060_failure_focused_gate_20260612/contact_sheet_heldout.jpg`
- report: `eval/qwenvl_c060_failure_focused_gate_20260612/report.md`
- visual audit: `eval/qwenvl_c060_failure_focused_gate_20260612/visual_audit.md`
- PE metric: `eval/qwenvl_c060_failure_focused_gate_20260612/pe_similarity_metrics.json`
- QwenVL metric: `eval/qwenvl_c060_failure_focused_gate_20260612/qwenvl_similarity_metrics.json`

gate 구성:

- samples: clean32 train `32` + heldout8 `8` = `40`
- variants: `no_ip`, `prev_w14`, `blend_prev14_c05504`, `c060_w14`
- generated PNGs: `160`
- cleanup: isolated ComfyUI server stopped, port `8116` closed

metric 결과:

PE mean uplift:

- `blend_prev14_c05504`: `+0.049596`
- `prev_w14`: `+0.029240`
- `c060_w14`: `+0.021860`

QwenVL mean uplift:

- `blend_prev14_c05504`: `+0.041589`
- `prev_w14`: `+0.036187`
- `c060_w14`: `+0.031796`

### 시각 감사

c060은 작동한다. `no_ip` 대비 palette, costume, 악역 aura, 수염/관모 같은 일부 cue를 분명히 가져온다. 하지만 reference-control gate는 통과하지 못했다.

대표 실패:

- `heldout01`: 노인 남성의 각진 얼굴과 말풍선 구성 대신 젊은 shouting character로 밀린다.
- `heldout07`: 초록 괴물 측면 reference가 human dark-villain template으로 붕괴한다.
- 일부 수염/관모/검은 의상 cue는 개선되지만, 구체적인 얼굴 구조와 특수 종족 형태가 안정적이지 않다.

### 판단

c060 decision은 `c060_failure_focused_not_quality_pass_runtime_blend_remains_best`다.

이 실험은 failure-focused continuation이 reference-active한 방향이라는 점은 확인했지만, adapter-only continuation만으로는 현재 best runtime blend를 넘지 못했다. 다음 루프는 단순 continuation 반복이 아니라 encoder/feature calibration, stronger image encoder, runtime blend distillation objective, 또는 실패류를 직접 구분하는 teacher objective로 넘어가야 한다.

## 12. 2026-06-12 c061 QwenVL instruction calibration gate

### 왜 시작했나

c060은 adapter-only continuation을 더 했지만 best runtime blend를 넘지 못했다. 그래서 c061은 새 학습 없이 QwenVL image embedding instruction만 바꿔도 reference embedding이 더 유용해지는지 확인했다.

핵심 질문은 다음이었다.

1. 같은 checkpoint와 같은 weight에서 instruction만 바꾸면 결과가 달라지는가?
2. `species`, `face structure`, `non-human trait`, `speech bubble`, `pose crop`, `prop`을 직접 강조하면 c058/c060의 실패류가 줄어드는가?
3. 개선이 있다면 workflow preset으로 쓸 수 있는가, 아니면 prompt-only calibration의 한계로 보고 다음 학습 루프로 넘어가야 하는가?

### 실험 구성

- output: `eval/qwenvl_c061_instruction_calibration_gate_20260612/`
- train contact sheet: `eval/qwenvl_c061_instruction_calibration_gate_20260612/contact_sheet_train.jpg`
- heldout contact sheet: `eval/qwenvl_c061_instruction_calibration_gate_20260612/contact_sheet_heldout.jpg`
- report: `eval/qwenvl_c061_instruction_calibration_gate_20260612/report.md`
- visual audit: `eval/qwenvl_c061_instruction_calibration_gate_20260612/visual_audit.md`
- PE metric: `eval/qwenvl_c061_instruction_calibration_gate_20260612/pe_similarity_metrics.json`
- QwenVL metric: `eval/qwenvl_c061_instruction_calibration_gate_20260612/qwenvl_similarity_metrics.json`

variants:

- `no_ip`
- `blend_default`: 기존 기본 instruction + `prev_w14 1.4 + c055 0.4`
- `blend_identity_exact`: exact identity, face shape, age, species/non-human traits, costume, palette, pose crop, props, speech bubbles 강조
- `blend_species_face`: strict visual identity retrieval, non-human species, monster/demon traits, profile silhouette, beard/headwear, skin tone, glowing eyes, props, costume palette 강조

검증 guard:

- samples: clean32 train `32` + heldout8 `8` = `40`
- generated PNGs: `160`
- blank PNGs: `0`
- min pixel std: `35.883`
- API prompt guard: 같은 sample 안에서 seed, positive/negative prompt, checkpoint sequence, weights `1.4 + 0.4`, start/end range가 모두 같고 `AnimaQwenVLEncodeImage.instruction`만 달라짐
- cleanup: isolated ComfyUI server stopped, port `8116` closed

### metric 결과

PE mean uplift:

- `blend_species_face`: `+0.060893`
- `blend_identity_exact`: `+0.054909`
- `blend_default`: `+0.049596`

QwenVL mean uplift:

- `blend_species_face`: `+0.042190`
- `blend_default`: `+0.041589`
- `blend_identity_exact`: `+0.039557`

heldout PE mean uplift:

- `blend_species_face`: `+0.053534`
- `blend_default`: `+0.039142`
- `blend_identity_exact`: `+0.035494`

heldout QwenVL mean uplift:

- `blend_species_face`: `+0.026471`
- `blend_default`: `+0.022779`
- `blend_identity_exact`: `+0.016944`

### 시각 감사

`blend_species_face`는 c061 안에서 가장 낫다. 수염, 관모, 붉은 눈, 어두운 palette, costume cue가 default보다 조금 더 안정적인 경우가 있다. 따라서 현재 workflow preset으로는 default instruction보다 `species_face` instruction을 쓰는 편이 낫다.

하지만 고퀄 reference-control gate는 통과하지 못했다. 세 adapter column이 대부분 거의 같은 구도/인물로 수렴했고, instruction만으로는 모델이 reference의 구조적 identity를 새로 잡지 못한다.

대표 실패:

- `heldout01`: 노인 남성의 각진 얼굴과 말풍선 구성이 젊은 shouting martial artist로 밀린다.
- `heldout07`: 초록 괴물 측면 reference가 여전히 human dark-villain body template으로 붕괴한다. `species_face`는 red eye/dark palette를 조금 강화하지만 monster head/profile identity는 유지하지 못한다.
- train에서도 비슷하게 palette/costume cue는 가져오지만 정확한 face structure와 특수 silhouette은 불안정하다.

### 판단

c061 decision은 `instruction_calibration_species_face_best_preset_not_quality_pass`다.

즉 QwenVL instruction은 완전히 무의미하지 않다. 같은 checkpoint/weight/seed에서 instruction만 바꿔도 metric과 일부 visual cue가 개선된다. 그러나 prompt-only feature calibration만으로는 원하는 고퀄 reference-control 수준에 도달하지 못한다. 다음 루프는 실제 encoder/feature adaptation 또는 runtime blend distillation objective로 넘어가야 한다.

## 13. c062 QwenVL calibrator/distillation continuation gate

c062는 c061에서 확인한 `species_face` instruction과 기존 최상 runtime preset인 `blend_species_face`를 단일 checkpoint 쪽으로 흡수할 수 있는지 확인하기 위한 실험이었다. 목적은 단순히 loss를 낮추는 것이 아니라, ComfyUI native QwenVL IP-Adapter 경로에서 바로 쓸 수 있고, heldout reference에서도 기존 blend보다 더 강한 identity control을 보이는 checkpoint를 만드는 것이었다.

### 학습 구성

학습 데이터는 `training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.jsonl`을 사용했다. 총 `154` rows이며 heldout row는 학습에 사용하지 않았다. 이 manifest는 clean32 기준 샘플, c052 positive identity pair, c060 failure-focused repeat를 묶어서 구성했다. c060/c061에서 반복적으로 실패한 비인간 profile, 노인 얼굴 구조, 수염/관모, 손 prop, fan/weapon cue, speech-bubble context를 더 강하게 보게 하려는 선택이었다.

초기 checkpoint는 `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`를 사용했다. 출력 checkpoint는 `checkpoints/anima_qwenvl_ip_adapter_c062_calibrator_distill_b128_0096_20260612.safetensors`다. 명령 surface는 `training/qwenvl_contrastive_cli.py`였고, 주요 설정은 `96` steps, `lr=3e-6`, `contrastive_weight=0.35`, `retrieval_weight=0.20`, `calibrator_bottleneck_dim=128`, c061 `species_face` instruction이다.

학습 결과는 finite였고 `first_loss=0.2402564585`, `final_loss=0.1878893971`이었다. checkpoint load 검증은 통과했고, PE checkpoint를 잘못 넣었을 때 reject되는 guard도 통과했다. 다만 `trainable_parameters=308,176,540`으로 기록되었기 때문에 이 실험은 엄밀한 의미의 작은 calibrator-only 학습이라기보다, calibrator bottleneck을 포함한 broad adapter continuation으로 판단해야 한다.

### ComfyUI gate

검증은 isolated ComfyUI API에서 진행했다. `/object_info` 기준으로 `AnimaQwenVLIPAdapterLoader`, `AnimaQwenVLEncodeImage`, `AnimaQwenVLIPAdapterApply`가 노출되었고, c062 checkpoint도 모델 선택 목록에 나타났다. 비교 column은 `no_ip`, 현재 최상 preset인 `blend_species_face`, 신규 `c062_w14`였다.

평가 샘플은 clean32 train `32`장과 heldout `8`장, 총 `40`개였다. 각 샘플마다 `3` variants를 생성해서 총 `120` PNG가 생성되었다. blank image는 `0`개였고, 최소 pixel std는 `35.883`이었다. 실험 후 ComfyUI server는 종료했고 port `8116`도 닫힌 것을 확인했다.

### metric 결과

| metric | blend_species_face | c062_w14 |
| --- | ---: | ---: |
| PE mean uplift | `0.060893` | `0.013234` |
| PE train uplift | `0.062733` | `0.017362` |
| PE heldout uplift | `0.053534` | `-0.003277` |
| QwenVL mean uplift | `0.042190` | `0.026588` |
| QwenVL train uplift | `0.046120` | `0.032966` |
| QwenVL heldout uplift | `0.026471` | `0.001077` |

metric 기준으로 c062는 기존 `blend_species_face` preset을 넘지 못했다. 특히 heldout에서 PE uplift가 음수로 떨어졌고, QwenVL heldout uplift도 거의 0에 가까웠다.

### 시각 감사

c062는 active checkpoint로서 생성 결과에 영향을 준다. palette, robe lighting, hand shape, pose crop이 바뀌는 경우가 있고, 일부 train row에서는 극적인 어두운 costume cue가 강해졌다. 하지만 reference identity를 더 정확하게 가져오지는 못했다.

대표 실패는 `heldout07`이다. 초록색 비인간 side-profile reference가 여전히 human dark-villain body template으로 붕괴했고, monster head/profile identity를 회복하지 못했다. `heldout01`과 `heldout05`에서도 노인 얼굴 구조, 나이감, crop/speech-bubble context가 유지되지 않았다.

### 판단

c062 decision은 `not_promoted`다.

이 실험은 QwenVL native node, checkpoint loading, model selection, API generation, contact sheet generation이 실제로 작동한다는 것은 다시 확인했다. 하지만 고퀄 reference-control checkpoint로 바로 믿고 쓸 수준은 아니다. 같은 계열 checkpoint를 조금 더 이어 학습하는 방향은 효율이 낮아 보이며, 다음 루프는 encoder-side/feature adaptation 또는 failure attribute를 직접 맞히는 더 강한 objective로 넘어가야 한다.

## 14. c063 QwenVL calibrator-only gate

c063은 c062의 중요한 반성에서 시작했다. c062는 이름상 calibrator/distillation이었지만 실제 구현에서는 adapter 전체가 `requires_grad=True`로 열려 있었다. 그래서 `calibrator_bottleneck_dim=128`을 붙였더라도 `trainable_parameters=308,176,540`인 broad adapter continuation이었다. c063의 목적은 같은 방향을 반복하지 않고, 정말로 `feature_calibrator.*`만 학습하는 작은 feature-adaptation이 효과가 있는지 확인하는 것이었다.

### 구현 수정

`training/qwenvl_smoke_checkpoint.py`에 `train_calibrator_only` 경로를 추가했다. 이 옵션이 켜지면 모든 QwenVL adapter parameter를 freeze하고 `feature_calibrator.norm`, `feature_calibrator.down`, `feature_calibrator.up`만 trainable로 둔다. `training/qwenvl_contrastive_smoke.py`는 optimizer가 `requires_grad=True` parameter만 받도록 바꾸었고, trainable parameter가 0개면 fail-loud 하게 했다. `training/qwenvl_contrastive_cli.py`에는 `--train-calibrator-only` 옵션을 추가했다.

이 수정은 `tests/test_qwenvl_feature_calibration.py`에서 검증했다. calibrated checkpoint에서 calibrator-only trainable name이 정확히 네 개로 제한되는지, calibrator가 없는 checkpoint에 `train_calibrator_only`를 걸면 거부되는지를 테스트했다.

### 학습 구성

학습 데이터는 c060/c062와 같은 `training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.jsonl`을 사용했다. 총 `154` rows이고 heldout row는 학습에 쓰지 않았다. 초기 checkpoint는 `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`였고, 출력 checkpoint는 `checkpoints/anima_qwenvl_ip_adapter_c063_calibrator_only_b128_0128_20260612.safetensors`다.

주요 설정은 `128` steps, `lr=8e-5`, `contrastive_weight=0.35`, `retrieval_weight=0.20`, `calibrator_bottleneck_dim=128`, `--train-calibrator-only`, c061 `species_face` instruction이다.

학습 결과:

| metric | value |
| --- | ---: |
| rows_loaded | `154` |
| first_loss | `0.331381` |
| final_loss | `0.193753` |
| mean_loss | `0.237293` |
| finite_loss | `true` |
| trainable_parameters | `528,384` |
| frozen_base_parameters | `4,947,838,963` |
| checkpoint.loadable | `true` |
| checkpoint.pe_checkpoint_rejected | `true` |

즉 c063은 c062와 달리 실제 calibrator-only 학습으로 닫혔다.

### ComfyUI gate

검증은 isolated ComfyUI API에서 진행했다. `/object_info`에서 `AnimaQwenVLIPAdapterLoader`, `AnimaQwenVLEncodeImage`, `AnimaQwenVLIPAdapterApply`가 모두 확인되었고, c063 checkpoint가 모델 선택 목록에 나타났다.

비교 column은 다음과 같다.

- `no_ip`
- `blend_species_face`: 현재 최상 runtime preset, previous retrieval `1.4` + c055 `0.4`
- `c063_calibrator_only_w14`: c063 checkpoint `1.4`

평가 샘플은 clean32 train `32`장과 heldout `8`장, 총 `40`개였다. 총 `120` PNG가 생성되었고, blank image는 `0`개, 최소 pixel std는 `35.883`이었다. 실험 후 ComfyUI server는 종료했고 port `8116`도 닫힌 것을 확인했다.

### metric 결과

| metric | blend_species_face | c063_calibrator_only_w14 |
| --- | ---: | ---: |
| PE mean uplift | `0.060893` | `0.029465` |
| PE train uplift | `0.062733` | `0.035551` |
| PE heldout uplift | `0.053534` | `0.005121` |
| QwenVL mean uplift | `0.042190` | `0.037178` |
| QwenVL train uplift | `0.046120` | `0.040380` |
| QwenVL heldout uplift | `0.026471` | `0.024371` |

QwenVL metric만 보면 c063은 baseline에 꽤 가까워졌다. 하지만 PE metric에서는 여전히 baseline보다 크게 낮고, 특히 heldout PE uplift가 `0.005121`로 거의 이득이 없다. reference-control 모델로 승격하려면 시각적으로도 heldout identity를 회복해야 하는데, 이 조건을 통과하지 못했다.

heldout focus:

| sample | PE blend | PE c063 | QwenVL blend | QwenVL c063 |
| --- | ---: | ---: | ---: | ---: |
| `heldout01` | `0.043998` | `0.023047` | `0.074187` | `0.096285` |
| `heldout05` | `0.093415` | `-0.016610` | `0.016990` | `0.000413` |
| `heldout07` | `-0.095589` | `-0.109479` | `-0.051999` | `-0.053679` |

### 시각 감사

c063은 active checkpoint다. pose, robe tone, black/red costume balance, hand shape, purple lighting을 실제로 바꾼다. 하지만 이 변화가 reference identity 향상으로 이어지지는 않았다.

대표 실패:

- `heldout01`: QwenVL uplift만 보면 c063이 더 높지만, 실제 이미지는 여전히 젊은 shouting warrior 쪽으로 가며 노인 얼굴 구조, 주름, 말풍선/crop context, reference face structure가 약하다.
- `heldout05`: c063은 official black hat/robe cue를 조금 더 넣지만, beard/crop/speech-bubble context와 정확한 얼굴은 회복하지 못했다. PE와 시각 판단 모두 기존 blend 쪽이 낫다.
- `heldout07`: 가장 중요한 실패가 유지된다. 초록색 비인간 side-profile monster reference가 여전히 human dark-villain body template으로 붕괴한다. c063은 PE/QwenVL 둘 다 baseline보다 약간 더 낮다.

### 판단

c063 decision은 `not_promoted`다.

중요한 성과는 “진짜 calibrator-only 학습 경로”가 구현되고, ComfyUI native loader에서 checkpoint가 실제로 선택/생성되는 것을 확인했다는 점이다. 하지만 원하는 고퀄 reference-control에는 부족하다. 얕은 `feature_calibrator`만 학습하는 방식은 reference identity 실패를 해결하지 못한다. 다음 루프는 adapter continuation이나 calibrator-only 반복이 아니라, QwenVL/SigLIP encoder-side adaptation, failure-attribute supervised embedding, 또는 별도 teacher/distillation objective로 넘어가야 한다.

## 15. c064 Failure-Attribute Embedding Probe

c064는 c063 `not_promoted` 이후 바로 학습을 반복하지 않고, 기존 embedding space가 hard failure를 분리할 수 있는지 확인하기 위해 진행했다. 목적은 “QwenVL/SigLIP2/PE 중 하나를 teacher로 삼아 얕은 supervised calibrator나 adapter objective를 더 밀어도 되는가”를 먼저 판단하는 것이었다.

### 입력과 boundary

새 이미지는 생성하지 않았다. c063 gate의 heldout hard case 3개만 사용했다.

- `heldout01`: old-face/speech-bubble/crop/side-profile context
- `heldout05`: old bearded official, black hat, upper-body crop
- `heldout07`: non-human green monster side-profile, red glowing eye

입력 manifest는 `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/probe_manifest.jsonl`이다. 각 row는 color dataset 원본 reference와 c063 gate의 `no_ip`, `blend_species_face`, `c063_calibrator_only_w14` 이미지를 가리킨다. 이 manifest는 오프라인 probe용이며 heldout을 학습에 사용하지 않는다.

### 개발한 것

`tools/probe_failure_attribute_embeddings.py`를 추가했다. 이 도구는 manifest를 읽고 QwenVL/SigLIP2/PE embedder로 reference와 후보 이미지 간 cosine을 계산한다. sample별로 다음 값을 기록한다.

- candidate cosine
- no-IP 대비 uplift
- candidate rank
- c063 vs blend delta
- hard failure별 binary decision

`tests/test_failure_attribute_embedding_probe.py`는 fake embedder로 rank/uplift/summary decision이 맞게 계산되는지 검증한다.

### 실행 결과

세 encoder를 같은 manifest에 대해 순차 실행했다.

- QwenVL metrics: `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/qwenvl_probe_metrics.json`
- SigLIP2 metrics: `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/siglip_probe_metrics.json`
- PE metrics: `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/pe_probe_metrics.json`
- 종합 summary/report: `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/summary.json`, `report.md`

요약:

| encoder | supported cases | 핵심 실패 |
| --- | ---: | --- |
| QwenVL | `1/3` | heldout01만 support. heldout05는 uplift가 약하고 heldout07은 no-IP가 1위 |
| SigLIP2 | `0/3` | 세 hard case 모두 no-IP가 1위 |
| PE | `1/3` | heldout05만 support. heldout07은 no-IP가 1위 |

### 판단

c064 decision은 `encoder_side_checkpoint_required_for_hard_failures`다.

가장 중요한 근거는 `heldout07`이다. non-human green side-profile은 세 encoder 모두에서 `no_ip`가 1위였고, `blend_species_face`와 `c063_calibrator_only_w14`는 모두 negative uplift였다. 즉 현재 off-the-shelf QwenVL/SigLIP2/PE pooled feature는 이 실패 속성을 안정적으로 reference-control teacher로 제공하지 못한다.

따라서 다음 루프는 adapter continuation, checkpoint merge, instruction-only, calibrator-only 반복이 아니다. c065는 color single-character crop 기반으로 failure-attribute encoder-side checkpoint 또는 attribute teacher/reranker를 설계해야 한다. 우선순위는 encoder 자체가 non-human species, side-profile silhouette, beard/headwear, crop context를 generic human template보다 잘 분리하도록 만드는 쪽이다.

## 16. c065 Encoder-Side Failure Attribute Probe

c065는 c064의 결론을 바로 학습으로 옮기기 전에, 실제로 train split 안에서 실패 속성을 분리할 수 있는지 확인하기 위해 진행했다. 목적은 “기존 QwenVL/SigLIP2/PE embedding을 teacher로 삼아 encoder-side checkpoint나 attribute calibrator를 학습해도 되는가”를 train-only pair probe로 검증하는 것이었다.

### 계획과 데이터

계획 문서는 `docs/c065_encoder_side_failure_attribute_plan_ko.md`다. 사용 데이터는 다음과 같다.

- Train manifest: `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- Heldout manifest: `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- Attribute source: `eval/qwenvl_c061_instruction_calibration_gate_20260612/summary.json`
- Image root: `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`

새 도구 `tools/build_c065_failure_attribute_pairs.py`를 추가해 `training/manifests/c065_failure_attribute_pairs_20260612.jsonl`와 summary를 만들었다. 이 manifest는 기존 `tools/score_identity_pair_probe.py`와 호환되도록 `pair_id`, `label`, `anchor_id`, `candidate_id`, `anchor_group`, `candidate_group`를 포함하고, c065 판단을 위해 `attribute_bucket`, `anchor_attributes`, `candidate_attributes`, `matched_keywords`, `negative_reason`, `source_split`도 함께 기록한다.

검증용 bucket은 세 개다.

| bucket | 목적 |
| --- | --- |
| `non_human_red_pale_profile_proxy` | heldout07 green monster 붕괴를 직접 다루기 위한 proxy. 단, clean32 train에는 direct green monster positive가 없으므로 red eye/pale villain 기준만 사용 |
| `beard_headwear_crop` | heldout05의 old bearded official, black hat, upper-body crop 계열 |
| `old_face_crop` | heldout01/05의 old face, beard, crop 계열 |

Manifest summary:

| 항목 | 값 |
| --- | ---: |
| total pairs | 126 |
| positive pairs | 63 |
| negative pairs | 63 |
| heldout rows used | 0 |
| direct green monster positive count | 0 |
| missing pair paths | 0 |

중요한 점은 `direct_green_monster_positive_count=0`이다. 즉 c065는 heldout07을 해결하기 위한 direct green/non-human 학습 데이터가 아직 없음을 확인한 실험이기도 하다.

### score tool 개선

`tools/score_identity_pair_probe.py`에는 `anchor_group`별 `group_summaries`를 추가했다. 기존 전체 margin/AUC만으로는 어떤 실패 속성이 분리되는지 알 수 없었기 때문이다. 테스트 `tests/test_identity_feature_probe.py`도 group summary를 검증하도록 갱신했다.

### 실행 결과

세 encoder를 같은 c065 pair manifest에 대해 실행했다.

- QwenVL: `eval/c065_encoder_side_failure_attribute_20260612/qwenvl_pair_probe.json`
- SigLIP2: `eval/c065_encoder_side_failure_attribute_20260612/siglip_pair_probe.json`
- PE: `eval/c065_encoder_side_failure_attribute_20260612/pe_pair_probe.json`
- 종합: `eval/c065_encoder_side_failure_attribute_20260612/summary.json`, `report.md`

전체 결과:

| encoder | margin | AUC | midpoint acc | 판단 |
| --- | ---: | ---: | ---: | --- |
| QwenVL | -0.005276 | 0.460569 | 0.412698 | fail |
| SigLIP2 | 0.009885 | 0.578231 | 0.595238 | fail |
| PE | -0.026462 | 0.412195 | 0.428571 | fail |

non-human proxy 결과:

| encoder | margin | AUC |
| --- | ---: | ---: |
| QwenVL | -0.021500 | 0.414966 |
| SigLIP2 | -0.001178 | 0.503401 |
| PE | -0.001421 | 0.489796 |

threshold는 margin `>= 0.05`, AUC `>= 0.70`이었다. 어떤 encoder도 통과하지 못했다.

### 판단

c065 decision은 `existing_encoder_feature_separation_not_viable_for_c065_checkpoint`다.

이 결과는 “QwenVL이나 SigLIP2를 쓰면 바로 해결된다”가 아니라는 쪽의 증거다. 현재 clean32 train split과 off-the-shelf feature space만으로는 red/pale non-human proxy조차 분리되지 않는다. 특히 direct green monster positive가 0이므로, 이 상태에서 encoder-side checkpoint를 바로 학습하면 heldout07의 핵심 실패를 학습할 재료가 부족하다.

다음 루프 우선순위는 `direct_green_non_human_mining`이다. color dataset 전체에서 green monster, non-human demon, red eye, side-profile direct positives를 먼저 채굴하고, 같은 pair-separation gate를 다시 통과시켜야 한다. 그 다음에야 encoder-side checkpoint나 attribute teacher/reranker 학습으로 넘어가는 것이 맞다.

## 17. c066 Direct Green / Non-Human Mining

c066은 c065의 `direct_green_monster_positive_count=0`과 proxy feature-separation 실패를 받은 다음 루프다. 목표는 바로 checkpoint를 학습하는 것이 아니라, local color dataset 안에서 heldout 누수 없이 쓸 수 있는 직접 green/non-human positive가 실제로 충분한지 확인하는 것이었다.

계획 문서는 `docs/c066_direct_green_non_human_mining_plan_ko.md`다.

사용 데이터:

- Image root: `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`
- Clean32 train: `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- Clean32 heldout: `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- c061 selected attributes: `eval/qwenvl_c061_instruction_calibration_gate_20260612/summary.json`
- c065 pair attributes: `training/manifests/c065_failure_attribute_pairs_20260612.jsonl`

새 도구:

- `tools/c066_candidate_types.py`
- `tools/build_c066_direct_green_candidates.py`
- `tests/test_c066_direct_green_candidates.py`

이 도구는 세 근거를 같이 본다.

1. sidecar caption keyword scan
2. c061/c065 selected attribute keyword scan
3. image-level green-pixel scan

중요하게, `direct_green_attribute`와 `direct_green_pixel_candidate`를 분리했다. green pixel이 많다고 해서 바로 “green non-human character”라고 판정하면 잎, 배경, 방, 찻잔 같은 색상 노이즈가 학습 positive로 들어가기 때문이다.

실행 산출물:

- Candidate manifest: `training/manifests/c066_direct_green_non_human_candidates_20260612.jsonl`
- Candidate summary: `training/manifests/c066_direct_green_non_human_candidates_20260612.summary.json`
- Pair manifest: `training/manifests/c066_direct_green_non_human_pairs_20260612.jsonl`
- Review sheet: `eval/c066_direct_green_non_human_mining_20260612/green_top16_probe_sheet.jpg`
- Probe summary/report: `eval/c066_direct_green_non_human_mining_20260612/summary.json`, `report.md`

후보 채굴 결과:

- total candidates: `120`
- positive candidates: `78`
- negative candidates: `42`
- direct green character attribute positives: `0`
- direct green pixel candidates: `40`
- non-human proxy positives: `38`
- heldout rows used: `0`
- missing paths: `0`
- sidecar caption keyword hits: `0`

source bucket:

- `direct_green_pixel_candidate`: `40`
- `fang_profile_proxy`: `17`
- `human_negative`: `20`
- `old_headwear_negative`: `22`
- `pale_non_human_proxy`: `11`
- `red_eye_proxy`: `10`

review sheet 기준으로 top green-pixel 후보는 대부분 잎, 배경, 실내 소품, 찻잔, 장면 조명 계열이었다. 즉 color dataset 안에 green pixel은 있지만, clean32 heldout의 `green monster face with red glowing eye` 같은 직접 실패 속성 positive는 train 쪽에서 확인되지 않았다. sidecar caption도 전부 `mrcolor_panel_style`, `full color manga panel`, `character panel` 같은 스타일 설명이라 후보 라벨 근거로는 쓸 수 없었다.

feature probe는 `training/manifests/c066_direct_green_non_human_pairs_20260612.jsonl`의 156 pair를 QwenVL, SigLIP2, PE로 실행했다.

전체 결과:

| encoder | margin | AUC | midpoint | decision |
| --- | ---: | ---: | ---: | --- |
| QwenVL | `-0.000703` | `0.509615` | `0.487179` | `feature_not_sufficiently_separated` |
| SigLIP2 | `0.008130` | `0.537229` | `0.532051` | `feature_not_sufficiently_separated` |
| PE | `0.009635` | `0.523915` | `0.506410` | `feature_not_sufficiently_separated` |

green-pixel bucket만 따로 봐도 기준을 넘지 못했다.

| encoder | green-pixel margin | green-pixel AUC |
| --- | ---: | ---: |
| QwenVL | `0.034769` | `0.543750` |
| SigLIP2 | `0.029138` | `0.600000` |
| PE | `0.039530` | `0.571250` |

gate는 margin `>= 0.05`, AUC `>= 0.70`이었으므로 c066은 통과하지 못했다.

c066 decision은 `direct_green_data_insufficient_attribute_teacher_required`다.

따라서 지금 데이터로 바로 encoder-side checkpoint를 학습하지 않는다. green pixel 후보는 실제 target character positive가 아니고, 기존 encoder feature도 그 후보군을 안정적으로 분리하지 못한다. 다음 루프는 더 긴 IP-Adapter continuation이 아니라 다음 둘 중 하나여야 한다.

1. QwenVL/vision-captioning 기반으로 전체 color dataset에 직접 green/non-human 속성 annotation을 새로 붙이고 review sheet로 확인한다.
2. 직접 green/non-human, red eye, profile, beard/headwear를 맞히는 explicit attribute teacher/reranker를 먼저 만든 뒤, 그 teacher를 사용해 encoder-side objective를 설계한다.

## 18. c067 Attribute Teacher / Reranker Seed

c067은 c066의 두 번째 후속안, 즉 explicit attribute teacher/reranker를 먼저 만드는 루프다. 목표는 checkpoint 학습이 아니다. c066에서 직접 green/non-human positive가 부족하고 green pixel 후보가 배경/소품 오탐으로 섞였기 때문에, 다음 학습에 넣을 후보를 QwenVL 이미지-텍스트 retrieval로 다시 걸러낼 수 있는지 확인했다.

사용 데이터는 heldout 누수를 막기 위해 다음으로 제한했다.

- clean32 train: `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- clean32 heldout exclusion: `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- c066 후보: `training/manifests/c066_direct_green_non_human_candidates_20260612.jsonl`

새 도구는 `tools/c067_attribute_teacher_core.py`다. 이 도구는 clean32 train을 먼저 넣고, c066 후보를 추가로 붙여 중복 image id를 제거한다. 만들어진 candidate manifest는 `eval/c067_attribute_teacher_reranker_seed_20260612/attribute_query_manifest.jsonl`이며, summary는 다음과 같다.

- candidate_count: `72`
- source_counts: clean32_train `32`, c066 `40`
- query_count: `6`
- heldout_rows_used: `0`
- missing_paths: `0`

attribute query는 다음 6개로 고정했다.

- `direct_green_non_human_face`
- `red_glowing_eye`
- `side_profile_silhouette`
- `beard_headwear_crop`
- `human_negative`
- `background_object_green`

scoring은 기존 `tools/build_reference_prompt_manifest.py`의 `Qwen3VLReferenceTextScorer`를 사용했다. 모델은 `Qwen/Qwen3-VL-Embedding-2B`다. 산출물은 다음이다.

- scores: `eval/c067_attribute_teacher_reranker_seed_20260612/attribute_scores.jsonl`
- top-k: `eval/c067_attribute_teacher_reranker_seed_20260612/attribute_topk.json`
- review sheet: `eval/c067_attribute_teacher_reranker_seed_20260612/attribute_review_sheet.jpg`
- report: `eval/c067_attribute_teacher_reranker_seed_20260612/report.md`
- visual audit: `eval/c067_attribute_teacher_reranker_seed_20260612/visual_audit.md`

결과는 `72` candidates x `6` queries = `432` score rows다. QwenVL retrieval 자체는 정상 작동했고, `direct_green_teacher_candidate_count=6`으로 score guard 기준 후보는 생겼다. 하지만 review sheet를 보면 `direct_green_non_human_face` top-k는 clean positive로 보기 어렵다. 상위 후보가 모자/그림자 강한 노인 얼굴, red-eye monk-like character, 일반 인물 클로즈업, 찻잔, 초록 배경/소품 패널까지 섞인다. 즉 c066의 오탐 원인인 green/background/object entanglement가 여전히 남아 있다.

반대로 `red_glowing_eye`, `side_profile_silhouette`, `beard_headwear_crop`은 리뷰 큐로는 유용하다. 해당 top-k에는 실제 붉은 눈, 측면 얼굴, 노인/수염/관모 crop 후보가 꽤 들어온다. `background_object_green`도 찻잔, 나뭇잎, 건물 장식, 초록 배경을 잘 잡아서 false-positive guard로 의미가 있다.

c067 decision은 `attribute_review_queue_manual_annotation_required`다.

따라서 c067 산출물은 다음 encoder-side objective의 자동 positive manifest가 아니라, attribute review queue로만 사용한다. 특히 direct green/non-human은 수동 label 또는 더 강한 captioning teacher가 먼저 필요하다. 다음 루프는 c067 top-k를 사람이 확인해 positive/negative label manifest로 만드는 단계이거나, QwenVL captioning으로 character skin/species와 background/object color를 분리하는 annotation stage여야 한다.

## 19. c068 Reviewed Attribute Label Seed

c068은 c067의 top-k 결과를 바로 학습에 넣지 않고, 먼저 auditable label seed로 바꾸는 루프다. 목표는 checkpoint 학습이 아니다. c067에서 `direct_green_non_human_face` top-k가 실제 non-human green face인지 불명확했기 때문에, 같은 후보들을 명시적 label로 재분류해서 “지금 학습해도 되는가”를 결정했다.

사용 입력은 다음으로 제한했다.

- c067 top-k: `eval/c067_attribute_teacher_reranker_seed_20260612/attribute_topk.json`
- heldout exclusion: `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- source commit: `9b53041`

새 도구는 `tools/c068_reviewed_attribute_labels.py`다. 이 도구는 c067 top-k row만 읽고, clean32 heldout row를 제외한 뒤, query/rank별로 사람이 검토한 label을 붙인다. 산출물은 다음이다.

- reviewed label manifest: `eval/c068_reviewed_attribute_label_seed_20260612/reviewed_attribute_labels.jsonl`
- summary: `eval/c068_reviewed_attribute_label_seed_20260612/summary.json`
- annotated review sheet: `eval/c068_reviewed_attribute_label_seed_20260612/annotated_review_sheet.jpg`
- report: `eval/c068_reviewed_attribute_label_seed_20260612/report.md`

summary는 다음을 기록한다.

- reviewed_rows: `48`
- query_count: `6`
- heldout_rows_used: `0`
- direct_green_target_positive_count: `0`
- decision: `direct_green_reviewed_seed_insufficient_new_annotation_required`

label 분포는 다음과 같다.

- `false_positive_background_object`: `11`
- `false_positive_human_face`: `9`
- `false_positive_human_old_face`: `4`
- `false_positive_red_eye_human`: `1`
- `negative_anchor`: `8`
- `target_positive`: `1`
- `useful_proxy_positive`: `14`

가장 중요한 결과는 direct-green/non-human query의 target positive가 `0`개라는 점이다. 해당 top-k 8개는 노인/관모/그림자 얼굴, red-eye human proxy, 초록 배경/소품으로 분류되었다. 따라서 이 seed를 encoder-side supervised positive로 사용하면 모델이 “green skin/species”가 아니라 “노인 얼굴, 붉은 눈 인간, 초록 물체/배경”을 배우게 될 위험이 크다.

반대로 `red_glowing_eye`는 target positive가 `1`개 있었고, `side_profile_silhouette` 및 `beard_headwear_crop`은 `useful_proxy_positive`로 쓸 수 있는 후보가 있다. 하지만 이것은 실패 속성 분석/negative guard에 유용한 보조 큐이지, direct-green/non-human 학습 양성으로는 충분하지 않다.

c068 decision은 `direct_green_reviewed_seed_insufficient_new_annotation_required`다. 다음 루프는 checkpoint 학습이 아니라, color dataset 전체에서 captioning 또는 수동 review를 통해 direct-green/non-human character positive를 새로 확보하는 단계여야 한다. 최소 목표는 direct-green target positive를 4개 이상 확보하고, background/object green false positive를 별도 negative guard로 유지하는 것이다.

## 20. c069 Direct Green Captioning/Data Acquisition

c069는 c068의 결론을 그대로 받아서, 학습을 멈추고 먼저 데이터 확보 가능성을 검증한 루프다. 목적은 local color dataset 전체에서 heldout 누수 없이 direct-green/non-human character positive를 4개 이상 찾을 수 있는지 확인하는 것이었다.

사용 데이터는 `training/manifests/local_color_self_reconstruct_20260611.jsonl` 전체 color manifest와 `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl` heldout manifest다. 실제 이미지 루트는 `/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best`이며, heldout 8장은 후보 스캔에서도 제외했다.

새로 만든 도구는 `tools/c069_direct_green_acquisition.py`와 `tools/c069_review_sheet.py`다. 1563장의 train-side color image를 green/red pixel 기준으로 스캔하고, `target_score`, `background_score`, `strong_green`, `red_green_mix` 4개 bucket에서 상위 12개씩 총 48 rows를 뽑았다. 산출물은 `eval/c069_direct_green_captioning_acquisition_20260612/candidate_manifest.jsonl`, `reviewed_candidate_labels.jsonl`, `annotated_review_sheet.jpg`, `summary.json`, `report.md`다.

검증 결과는 `heldout_rows_used=0`, `missing_paths=0`, `scanned_beyond_c067_topk=true`, `candidate_count=48`, `reviewed_rows=48`이다. 라벨은 `false_positive_background_object=46`, `useful_proxy_non_human=2`, `direct_green_target_positive_count=0`이었다. 즉, color dataset 전체를 더 넓게 훑어도 확정 direct-green/non-human target positive는 아직 확보되지 않았다.

c069 decision은 `new_dataset_captioning_required`다. 일부 proxy 후보는 있지만 target positive가 아니므로 encoder-side 학습으로 넘기지 않는다. 다음 루프는 새 데이터/수동 라벨링/QwenVL caption search를 통해 direct-green/non-human target positive 4개 이상을 확보하는 방향이어야 한다.

## 21. c070 QwenVL / Caption Search Direct-Green Acquisition

c070은 c069 이후 바로 학습하지 않고, semantic/caption search로 direct-green/non-human target positive를 확보할 수 있는지 확인한 루프다. c067/c068에서 이미 QwenVL image-text attribute rerank는 direct-green positive를 만들지 못했기 때문에, c070은 color dataset의 sidecar caption이 실제 semantic signal을 갖는지 먼저 점검했다.

입력은 `training/manifests/local_color_self_reconstruct_20260611.jsonl`, heldout 제외 manifest `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`, 그리고 c069 reviewed labels `eval/c069_direct_green_captioning_acquisition_20260612/reviewed_candidate_labels.jsonl`이다. heldout 8장은 스캔에서도 제외했고, c069에서 이미 리뷰한 33개 image id는 visual fallback bucket에서 제외했다.

실제 `.txt` sidecar caption은 1571개가 있지만 대부분 `mrcolor_panel_style, full color manga panel, clean webtoon coloring, manhwa panel art, character panel...` 같은 템플릿 태그다. `green`, `monster`, `creature`, `demon`, `skin`, `red eye` 같은 semantic keyword hit는 0개였다.

새로 만든 도구는 `tools/c070_qwenvl_caption_search.py`와 `tools/c070_color_metrics.py`다. 산출물은 `eval/c070_qwenvl_direct_green_caption_search_20260612/candidate_manifest.jsonl`, `reviewed_candidate_labels.jsonl`, `annotated_review_sheet.jpg`, `summary.json`, `report.md`다.

결과는 `scanned_image_count=1563`, `heldout_rows_used=0`, `missing_paths=0`, `caption_keyword_hit_images=0`, `candidate_count=36`, `reviewed_rows=36`이다. 라벨은 `direct_green_target_positive_count=0`, `useful_proxy_non_human_count=12`, `false_positive_human_count=12`, `false_positive_background_object_count=12`였다.

c070 decision은 `external_manual_data_required`다. c067/c068의 QwenVL attribute rerank와 c069/c070의 pixel/caption search가 모두 direct-green positive를 만들지 못했기 때문에, 이제 local dataset만으로 checkpoint 학습을 강행하면 false-positive를 학습할 위험이 높다. 다음 루프는 외부/추가 데이터셋, 수동 라벨링, 또는 실제 caption generator/VLM 질의 모델로 semantic annotation을 생성하는 방향이어야 한다.

## 22. c071 Direct-Green Manual / External Seed Package

c071은 c070의 `external_manual_data_required` 결론을 받은 뒤 이어진 data gate 루프다. 목표는 또 다른 checkpoint를 학습하는 것이 아니라, c068/c069/c070에서 확인한 proxy/guard 후보를 사람이 라벨링할 수 있는 패키지로 만들고, 그 라벨을 엄격하게 검증해 다음 학습으로 넘기는 것이다.

새 도구는 `tools/c071_seed_package.py`와 `tools/c071_import_manual_labels.py`다. 테스트는 `tests/test_c071_manual_seed_package.py`에 만들었다. 패키지 생성기는 c068 reviewed attributes, c069 reviewed candidates, c070 reviewed candidates를 읽고 clean32 heldout 8장을 제외한 뒤 unique image 기준으로 dedupe한다. 산출물은 `eval/c071_direct_green_seed_package_20260612/annotation_candidates.jsonl`, `annotation_template.csv`, `annotated_review_sheet.jpg`, `summary.json`이다.

수동 라벨 스키마는 `target_positive`, `useful_proxy_non_human`, `guard_false_positive_human`, `guard_false_positive_background_object`, `reject_unclear` 5개다. 중요한 점은 자동 suggested label이 절대 `target_positive`를 확정하지 않는다는 것이다. c068의 red-eye target이나 c069/c070의 useful proxy도 direct-green/non-human target positive로 자동 승격하지 않는다.

실제 결과는 source rows c068 `48`, c069 `48`, c070 `36`, raw candidate rows `132`, unique candidates `84`, heldout rows used `0`, missing paths `0`이다. suggested label은 `useful_proxy_non_human=29`, `guard_false_positive_background_object=40`, `guard_false_positive_human=15`로 나뉘었다.

example import는 자동 suggested label을 그대로 넣은 안전 예시다. 결과는 imported rows `84`, unique target positives `0`, decision `external_manual_data_required`다. importer는 unknown label, heldout row, duplicate `target_positive` image id를 거부한다. 따라서 다음 학습은 사람이 최소 4개의 unique `target_positive`를 실제로 확인한 뒤 importer가 `ready_for_encoder_training`을 반환할 때만 진행한다.

c071 decision은 `external_manual_data_required`다. 이제 남은 실질 작업은 외부/수동 라벨을 넣는 것이고, 라벨이 채워지면 같은 importer로 gate를 다시 열어 encoder-side training manifest를 만들 수 있다.

## 23. c072 External Direct-Green Source Discovery

c072는 c071의 `external_manual_data_required` 결론 이후 이어진 외부 공개 소스 탐색 루프다. 목적은 즉시 학습을 재개하는 것이 아니라, direct-green/non-human `target_positive`를 만들 수 있는 공개 데이터셋 후보가 있는지 확인하고, 다음 시각 검수 단계에 넘길 metadata package를 만드는 것이다.

조사는 Hugging Face Dataset API와 Dataset Viewer API로 제한했다. 대용량 tar/parquet 다운로드는 하지 않았고, 각 소스의 metadata와 작은 train row probe만 확인했다. 조사 대상은 `Wenaka/anima-ip-adapter-dataset`, `mrzjy/AniGamePersonaCaps`, `mrzjy/AnimeMangaCharacters-247K`, `alfredplpl/anime-with-caption-cc0`, `CaptionEmporium/furry-e621-safe-llama3.2-11b`다.

새 도구는 `tools/c072_external_source_discovery.py`와 `tools/c072_source_probe.py`다. 테스트는 `tests/test_c072_external_source_discovery.py`에 만들었다. live artifact는 `eval/c072_external_direct_green_source_discovery_20260612/` 아래에 `source_manifest.jsonl`, `external_candidates.jsonl`, `external_candidate_template.csv`, `summary.json`, `report.md`로 저장했다.

실제 결과는 inspected sources `5`, large downloads `false`, heldout rows used `0`, metadata candidates `12`, confirmed target positives `0`이다. source별 potential candidate count는 `AniGamePersonaCaps=63`, `AnimeMangaCharacters-247K=7`, `anime-with-caption-cc0=21`이었다. `Wenaka/anima-ip-adapter-dataset`은 공개 repo지만 card/license가 없고 이번 rows probe가 HTTP 500으로 실패해 학습 후보에서 제외했다. `CaptionEmporium/furry-e621-safe-llama3.2-11b`는 non-human caption/tag 밀도는 높지만 viewer row에 직접 image URL이 없어 c071 호환 이미지 후보 패키지에는 바로 넣지 않았다.

c072 decision은 `external_candidates_found_manual_confirmation_required`다. 외부 소스에서 후보를 찾는 것은 가능해 보이지만, 현재 후보는 metadata-only이고 `image_path`도 외부 URL이다. 따라서 아직 학습용 `target_positive`가 아니며, 다음 c073 루프에서 소량 다운로드/contact sheet/수동 라벨 검수를 통해 unique `target_positive >= 4`를 확인해야 한다.

## 24. c073 External Candidate Visual Review

c073은 c072 commit `ad60ea7`이 만든 metadata-only 외부 후보 12장을 실제 이미지로 확인하는 visual gate다. 목적은 "외부 후보가 있으니 바로 학습"이 아니라, 눈으로 확인한 `target_positive`가 최소 4개 있는지 검증하는 것이다. 이 기준을 통과하지 못하면 encoder-side 학습으로 넘기지 않는다.

새 도구는 `tools/c073_external_candidate_visual_review.py`다. 이 도구는 `eval/c072_external_direct_green_source_discovery_20260612/external_candidates.jsonl`을 읽고, 후보 이미지만 `.tmp/c073_external_candidate_visual_review/images/`에 소량 다운로드한다. 다운로드 이미지는 외부 라이선스/원본 URL이 섞여 있으므로 커밋하지 않는다. 검수용 contact sheet도 `.tmp/c073_external_candidate_visual_review/contact_sheet.jpg`에만 둔다. 커밋 대상은 `download_manifest.jsonl`, `visual_label_template.csv`, `manual_visual_labels.csv`, `reviewed_external_labels.jsonl`, `summary.json`, `report.md`다.

실제 결과는 candidate `12`, downloaded `12`, failed `0`, reviewed `12`, large downloads `false`, heldout rows used `0`이다. 수동 시각 라벨은 `useful_proxy_non_human=5`, `guard_false_positive_human=7`, `target_positive=0`이었다. Conway, Meatenstein, Rodent, Maha, Aaron Newt는 non-human proxy로는 의미가 있지만, 직접 green-skin/non-human anime reference target으로 쓰기에는 색/스타일/해상도/장르가 맞지 않았다. 나머지는 대부분 cat/fox girl 또는 인간형 캐릭터라 false positive human으로 분류했다.

c073 decision은 `external_manual_data_required`다. 즉 c072 외부 후보만으로는 학습을 재개하지 않는다. 다음 단계는 더 정확한 외부 데이터셋/수동 이미지 제공/새로운 검색 쿼리로 unique `target_positive >= 4`를 확보한 뒤, c071/c073 label schema를 통해 training manifest를 만드는 것이다. 이 결론은 품질을 위해 의도적으로 보수적으로 잡은 것이다.

## 25. c074 Tag-Backed Direct-Green Source Acquisition

c074는 c073의 `external_manual_data_required` 이후 진행한 tag-backed source acquisition 루프다. c073에서 metadata/caption 후보가 실제 이미지로는 false positive였기 때문에, c074에서는 `green_skin`, `colored_skin`, `tail`, `monster_girl`처럼 tag 근거가 명확하고 실제 이미지 asset이 있는 소스를 우선했다.

조사한 source는 `CyberHarem/neeko_leagueoflegends`, `OneIG-Bench/OneIG-Bench`, `CaptionEmporium/anime-caption-danbooru-2021-sfw-5m-hq`, `mrzjy/splash-art-gacha-collection-10k`다. 가장 강한 소스는 `CyberHarem/neeko_leagueoflegends`였다. 이 dataset은 card에 MIT와 core tags `green_skin`, `colored_skin`, `tail`, `monster_girl`가 명시되어 있고, repo에 sample PNG가 직접 존재한다. 반대로 `OneIG-Bench`는 prompt-only, `CaptionEmporium`은 tag/caption 중심이지만 image row가 없고 검색이 불안정, `splash-art-gacha`는 image+caption source지만 green query가 timeout되어 이번 package에는 넣지 않았다.

새 도구는 `tools/c074_tag_backed_source_acquisition.py`이고 테스트는 `tests/test_c074_tag_backed_source_acquisition.py`다. adult tag가 보이는 Neeko cluster 0은 제외하고 cluster 1/2 sample PNG 10장만 `.tmp/c074_tag_backed_direct_green_source_acquisition/`에 다운로드했다. raw external image와 contact sheet는 커밋하지 않았고, 커밋 대상은 `eval/c074_tag_backed_direct_green_source_acquisition_20260612/` 아래의 `source_manifest.jsonl`, `external_candidates.jsonl`, `download_manifest.jsonl`, `candidate_template.csv`, `manual_visual_labels.csv`, `reviewed_external_labels.jsonl`, `summary.json`, `report.md`다.

실제 결과는 inspected sources `4`, candidate rows `10`, downloaded `10`, reviewed `10`, target positives `10`, heldout rows used `0`, large downloads `false`, committed external images `0`이다. c074 decision은 `ready_for_encoder_training`이다. 다만 `CyberHarem/neeko_leagueoflegends`는 `not-for-all-audiences` 표시가 있고 원천 이미지가 여러 사이트에서 수집된 것이므로, 공개 배포/재배포 전 권리 검토가 필요하다는 caveat를 유지한다.

다음 루프는 c075다. c075는 c074 target positives 10장을 기존 c071/c073 guard negatives와 섞어 작은 training manifest를 만들고, bounded encoder-feature training을 수행한 뒤 c035-style single-character gate 또는 direct-green heldout/proxy gate로 검증해야 한다.

## 26. c075 Tag-Positive QwenVL Calibrator Training And Gate

c075는 c074에서 확인한 direct-green/non-human target positive 10장을 실제 학습에 넣어 본 첫 루프다. 목적은 단순히 green prompt를 강화하는 것이 아니라, QwenVL IP-Adapter embedding이 non-human species, green skin, tail, monster silhouette 같은 reference trait를 더 잘 전달하는지 확인하는 것이었다.

새 도구는 `tools/c075_tag_positive_manifest.py`, `tools/c075_manifest_files.py`, `tools/c075_tag_positive_manifest_types.py`이고 테스트는 `tests/test_c075_tag_positive_manifest.py`다. manifest는 `training/manifests/c075_tag_positive_direct_green_20260612.jsonl`에 만들었다. 구성은 c074 target positive 10장을 4회 반복한 `40` rows와 c060 source rows `80`개, 총 `120` rows다. heldout rows used는 `0`, missing paths는 `0`, committed external raw image는 `0`이다. 외부 raw 이미지는 `.tmp/c075_tag_positive_direct_green_root/` 아래 symlink/caption 형태로만 두었다.

학습은 previous retrieval checkpoint `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`에서 시작했고, c063에서 만든 `--train-calibrator-only` 경로를 사용했다. 출력 checkpoint는 `checkpoints/anima_qwenvl_ip_adapter_c075_tag_positive_calibrator_b128_0064_20260612.safetensors`다. 첫 실행은 root filesystem 여유 공간 부족으로 checkpoint 저장 중 실패했고, 깨진 partial checkpoint를 삭제한 뒤 재생성 가능한 ignored ComfyUI/eval raw PNG cache를 정리하고 같은 명령을 재실행했다. 최종 학습 summary는 finite_loss `true`, rows_loaded `120`, trainable_parameters `528384`, checkpoint.loadable `true`, pe_checkpoint_rejected `true`를 기록했다.

ComfyUI gate는 isolated `127.0.0.1:8116`에서 진행했다. 비교 column은 `no_ip`, 현재 최상 runtime preset인 `blend_species_face`, 신규 `c075_tag_positive_w14`다. clean32+heldout8 `40` samples에서 `120` PNG를 만들고, c074 direct-green target positive `10` samples에서 `30` PNG를 추가로 만들었다. 총 generated PNG는 `150`, blank image는 `0`, 최소 pixel std는 `35.8830680847168`이다. `contact_sheet_train.jpg`, `contact_sheet_heldout.jpg`, `contact_sheet_direct_green.jpg`를 생성했다.

metric 결과는 c075가 current best를 넘지 못했다. clean32+heldout8 PE mean uplift는 blend `0.0608932152`, c075 `0.0262199253`이고, QwenVL mean uplift는 blend `0.0421902567`, c075 `0.0349742755`다. direct-green focus에서는 더 약하다. PE mean uplift는 blend `0.0379917264`, c075 `-0.0206880599`이고, QwenVL mean uplift는 blend `-0.0121086836`, c075 `-0.0143850207`이다.

시각적으로도 c075는 green skin/tail 같은 큰 trait는 일부 유지하지만, reference의 뿔/머리 실루엣, 네온/귀여운 채색 스타일, 얼굴 identity를 안정적으로 가져오지 못하고 무협풍 성인 humanoid로 수렴했다. heldout07의 괴물 side-profile reference도 여전히 사람형 dark villain으로 바뀐다.

c075 decision은 `not_promoted_c075_tag_positive_calibrator_weaker_than_blend_species_face`다. runtime은 pass지만 품질은 pass가 아니다. 다음 루프는 같은 calibrator-only target-positive 반복이 아니라, 더 강한 encoder-side/reference feature objective 또는 실제 paired direct-green 데이터 구축으로 가야 한다.

## 27. c076 Paired Direct-Green Source Expansion

c076은 c075 실패 이후 바로 더 큰 checkpoint 학습으로 넘어가지 않고, paired/direct-green/non-human target-positive 데이터가 충분히 확장 가능한지 확인한 데이터 전제 검증 루프다. 핵심 질문은 "c074의 10장만 반복하면 c075와 같은 실패를 반복하는가, 아니면 새롭게 학습 가능한 target-positive가 충분히 늘어났는가"였다.

새 도구는 `tools/c076_paired_source_expansion.py`, `tools/c076_source_expansion_io.py`, `tools/c076_source_expansion_report.py`이고 테스트는 `tests/test_c076_paired_source_expansion.py`다. 계획 문서는 `docs/c076_paired_direct_green_source_expansion_plan_ko.md`에 작성했다. 프로브 대상은 c074 prior seed, `Wenaka/anima-ip-adapter-dataset`, `mrzjy/AniGamePersonaCaps`, `mrzjy/AnimeMangaCharacters-247K`, `alfredplpl/anime-with-caption-cc0`, `CaptionEmporium/furry-e621-safe-llama3.2-11b`다. 외부 원본 이미지는 `.tmp/c076_paired_direct_green_source_expansion/` 아래에만 두고 commit하지 않았다.

실행 결과는 inspected sources `6`, candidate count `13`, downloaded/materialized `13`, network downloaded `3`, reviewed rows `13`이다. 기존 c074 seed 10장은 계속 `target_positive`지만 새 metadata 후보 3개는 contact sheet 검수 후 target-positive로 승격하지 않았다. `c076_meta_002`는 인간형/머리색 false positive, `c076_meta_000`은 3D glove/object 계열 false positive, `c076_meta_001`은 off-domain non-human proxy다. 따라서 `new_target_positive_confirmed_count`는 `0`이다.

feature boundary는 새 reviewed target-positive가 없으므로 pair probe를 새로 돌리지 않고 c075 direct-green 결과를 기준으로 보류했다. c075 direct-green PE uplift는 blend `0.0379917264`, c075 `-0.0206880599`였고, QwenVL uplift는 blend `-0.0121086836`, c075 `-0.0143850207`였다. 이 수치 때문에 c074 seed만으로 다음 학습을 반복하는 것은 금지한다.

c076 decision은 `more_data_required`다. `ready_for_c077_training` 조건은 전체 unique target-positive 24장 이상, 그중 c074가 아닌 신규 target-positive 12장 이상인데, 현재는 unique target-positive 10장, 신규 0장이다. 다음 단계는 checkpoint 학습이 아니라 새 데이터 소스 확보 또는 사람이 직접 승인한 direct-green/non-human positive 라벨링이다.

## 28. c077 Direct-Green Target-Positive Acquisition

c077은 c076의 `more_data_required` 결론을 이어받아, metadata caption row가 아니라 Hugging Face sample asset tree에서 실제 이미지를 더 확보할 수 있는지 확인한 source acquisition 루프다. 목표는 새 checkpoint 학습이 아니라, 학습 전에 필요한 direct-green/non-human 신규 `target_positive` 12장 이상을 확보할 수 있는지 판정하는 것이었다.

새 도구는 `tools/c077_hf_sample_sources.py`, `tools/c077_target_positive_acquisition.py`, `tools/c077_acquisition_report.py`이고 테스트는 `tests/test_c077_target_positive_acquisition.py`다. 계획 문서는 `docs/c077_direct_green_target_positive_acquisition_plan_ko.md`에 작성했다. 외부 raw 이미지는 `.tmp/c077_direct_green_target_positive_acquisition/` 아래에만 두고 커밋하지 않았다.

source probe 대상은 c074 prior seed와 `CyberHarem/green_heart_azurlane`, `CyberHarem/poppy_leagueoflegends`, `CyberHarem/tristana_leagueoflegends`, `CyberHarem/lulu_leagueoflegends`, `CyberHarem/soraka_leagueoflegends`, `CyberHarem/nami_leagueoflegends`, `CyberHarem/vex_leagueoflegends`다. HF tree 기준 inspected source count는 `8`, candidate manifest count는 `60`, 실제 materialized/download candidate count는 `46`, 신규 다운로드는 `36`이다.

contact sheet 검수 결과 c074 seed 10장은 기존대로 `target_positive`지만, 신규 후보 36장은 target-positive로 승격하지 않았다. `CyberHarem/green_heart_azurlane` 8장은 초록 머리/의상 인간형이라 `guard_false_positive_human`으로 라벨링했고, Poppy/Tristana/Lulu/Soraka/Nami 계열 28장은 non-human proxy로는 의미가 있지만 direct-green skin target-positive 기준에는 부족해 `useful_proxy_non_human`으로 둔다. 따라서 신규 target-positive는 `0`, 전체 unique target-positive는 `10`이다.

c077 decision은 `manual_needed_more_target_positives`다. source가 완전히 막힌 것은 아니지만, 지금 후보군은 고품질 direct-green reference-control checkpoint를 학습할 만큼의 양성 데이터가 아니다. 다음 루프는 checkpoint training이 아니라 더 강한 source acquisition, 사용자 제공 direct-green 샘플, 또는 synthetic/direct-green bootstrap source 검토로 가야 한다.

## 29. c078 Synthetic Direct-Green Bootstrap Source

c078은 c077에서 public/HF sample source가 신규 target-positive를 만들지 못한 뒤, synthetic source가 실제 학습 전제로 쓸 수 있는지 확인한 루프다. 목적은 checkpoint training이 아니라, c079 학습 manifest로 넘길 수 있는 visually confirmed direct-green/non-human reference 후보를 확보하는 것이었다.

새 도구는 `tools/c078_synthetic_bootstrap.py`, `tools/c078_comfy_generation.py`이고 테스트는 `tests/test_c078_synthetic_bootstrap.py`다. 계획 문서는 `docs/c078_synthetic_direct_green_bootstrap_plan_ko.md`에 작성했다. ComfyUI02 API `http://127.0.0.1:8102`에서 `anima-base-v1.0.safetensors`, `qwen_3_06b_base.safetensors`, `qwen/qwen_image_vae.safetensors`를 사용해 text-only 생성했다.

prompt manifest는 24개 single-character direct-green/non-human prompt로 구성했다. goblin, lizardfolk, oni, slime, alien, frog yokai, dragonkin, orc, serpent folk, plant monster, insectoid 등 다양한 species cue를 넣었고, negative에는 multiple characters, normal human skin, text/watermark, nude/nsfw를 넣었다. 생성 결과는 24장 모두 성공했고 blank image는 0개다. raw generated PNG는 `.tmp/c078_synthetic_direct_green_bootstrap/generated/` 아래에만 두고 커밋하지 않았다.

contact sheet 수동 검수 결과 23장이 `target_positive`로 승인되었고, `c078_synth_21` 1장은 두 캐릭터가 함께 생성되어 `reject_unclear`로 제외했다. 따라서 c078은 신규 target-positive 23장으로 threshold 12장을 통과했고, decision은 `ready_for_c079_training_manifest`다.

다음 루프는 c079다. c079는 c074 real seed 10장과 c078 synthetic target-positive 23장을 섞되, c077 guard/proxy false-positive와 기존 guard data를 함께 넣어 과도한 green-human collapse를 막아야 한다. 학습 후에는 clean32+heldout8 및 direct-green focus gate에서 current best runtime preset과 비교해야 한다.

## 30. c079 Synthetic-Positive QwenVL Calibrator

c079는 c078에서 확보한 synthetic direct-green target-positive 23장과 c074 real target-positive 10장을 실제 QwenVL IP-Adapter 학습에 투입한 루프다. 목표는 단순히 녹색 피부 prompt를 강하게 만드는 것이 아니라, current best runtime preset `blend_species_face` 및 c075보다 direct-green/non-human reference-control이 좋아지는지 확인하는 것이었다.

새 도구는 `tools/c079_manifest_types.py`, `tools/c079_manifest_io.py`, `tools/c079_synthetic_positive_manifest.py`이고 테스트는 `tests/test_c079_synthetic_positive_manifest.py`다. manifest는 `training/manifests/c079_synthetic_positive_direct_green_20260612.jsonl`에 만들었다. 구성은 c074 real positive `10`, c078 synthetic target positive `23`, c077 guard/proxy `36`, source rows `80`이고, 반복 후 총 `248` rows다. `heldout_rows_used`는 `0`이다.

학습은 previous retrieval checkpoint `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`에서 시작했고, c063에서 만든 `--train-calibrator-only` 경로를 사용했다. 출력 checkpoint는 `checkpoints/anima_qwenvl_ip_adapter_c079_synthetic_positive_calibrator_b128_0128_20260612.safetensors`다. 첫 저장은 `/data/ai/models/ipadapter` 권한 문제로 실패했으나, 실패 로그를 `eval/qwenvl_c079_synthetic_positive_training_20260612/train_stdout_permission_failure.txt`에 보존하고 repo-local ignored checkpoint 경로로 재실행했다. 최종 학습 summary는 `rows_loaded=248`, `steps=128`, `final_loss=0.2041460872`, `finite_loss=true`, `trainable_parameters=528384`, checkpoint `loadable=true`, `pe_checkpoint_rejected=true`를 기록했다.

ComfyUI gate는 isolated `127.0.0.1:8116`에서 진행했다. 비교 column은 `no_ip`, current best `blend_species_face`, c075 baseline `c075_tag_positive_w14`, 신규 `c079_synthetic_positive_w14`다. clean32+heldout8 `40` samples에서 `160` PNG를 만들고, direct-green focus `33` samples에서 `132` PNG를 추가로 만들었다. 총 generated PNG는 `292`, blank image는 `0`, 최소 pixel std는 `35.321`이다. disk full로 ComfyUI temp output 복사 중 한 번 실패했지만, 중복 temp output만 정리하고 손상된 PNG 한 장을 재생성해 최종 artifact consistency를 통과했다. 생성 후 ComfyUI server를 종료했고 port `8116`은 닫혔다.

metric 결과는 부분 개선과 미승격을 동시에 보여준다. clean32+heldout8 PE mean uplift는 `blend_species_face=0.0608932152`, `c075=0.0262199253`, `c079=0.0329968661`이고, QwenVL mean uplift는 `blend_species_face=0.0421902567`, `c075=0.0349742755`, `c079=0.0338256791`이다. direct-green focus에서는 PE mean uplift가 `blend_species_face=0.3416856171`, `c075=0.2800143618`, `c079=0.2937260229`이고, QwenVL mean uplift가 `blend_species_face=0.0291833697`, `c075=0.0386207013`, `c079=0.0388706634`다.

시각 검수 기준으로 c079는 c075보다 direct-green 속성 신호를 조금 보강했지만, 원본별 identity를 안정적으로 유지하지 못한다. `contact_sheet_direct_green.jpg`에서 녹색 피부와 뿔/귀 실루엣은 일부 유지되지만 작은 체형, 밝은 색감, 장식, 귀여운 표정, 비인간 얼굴 구조가 대부분 성인형 green humanoid villain으로 수렴한다. `contact_sheet_heldout.jpg`에서도 c079는 c075와 거의 같은 방향이고, 핵심 실패인 heldout07 non-human side-profile monster는 여전히 사람형 dark villain으로 붕괴한다.

c079 decision은 `not_promoted_c079_synthetic_positive_calibrator_partial_direct_green_gain`이다. runtime은 pass이고 direct-green QwenVL에서는 c075를 아주 조금 앞섰지만, current best `blend_species_face`를 넘지 못했고 high-quality reference-control checkpoint로 승격할 수 없다. 다음 루프는 synthetic-positive 단순 반복이 아니라 실제 paired direct-green 데이터, synthetic source-target identity pair, 또는 QwenVL/SigLIP encoder-side reference feature objective 쪽으로 가야 한다.

## 31. c080 Paired Direct-Green Identity Supervision

c080은 c079의 한계를 받은 다음 루프다. c079는 c074 real target-positive 10장과 c078 synthetic target-positive 23장을 `ref_id == tgt_id`에 가까운 target-positive 방식으로 넣어 녹색/비인간 속성은 조금 보강했지만, 참조별 identity 다양성은 유지하지 못했다. 따라서 c080의 목표는 같은 synthetic-positive 반복이 아니라, c074 Neeko 계열 real direct-green 샘플을 `ref_id != tgt_id`인 paired supervision으로 바꿔 reference identity가 다른 target view로 전달되는지 확인하는 것이었다.

새 도구는 `tools/c080_paired_direct_green_manifest.py`이고 테스트는 `tests/test_c080_paired_direct_green_manifest.py`다. manifest는 `training/manifests/c080_paired_direct_green_identity_20260613.jsonl`에 만들었다. 구성은 c074 pair source `10`, c074 paired training rows `80`, c078 unpaired positive count `23`, c078 training rows `0`, guard/proxy rows `36`, source rows `80`, 총 `196` rows다. 핵심 조건인 `direct_self_pair_rows=0`, `heldout_rows_used=0`을 기록했다. c078 synthetic direct-green 이미지는 같은 identity의 다른 target view가 없어서 이번 paired 학습에는 넣지 않았다.

학습은 previous retrieval checkpoint `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`에서 시작했고, c063에서 만든 `--train-calibrator-only` 경로를 사용했다. 출력 checkpoint는 `checkpoints/anima_qwenvl_ip_adapter_c080_paired_direct_green_b128_0128_20260613.safetensors`다. 학습 조건은 `steps=128`, `max_rows=196`, `lr=5e-6`, `contrastive_weight=0.35`, `retrieval_weight=0.2`, `calibrator_bottleneck_dim=128`이다. 최종 summary는 `rows_loaded=196`, `final_loss=0.2474229336`, `finite_loss=true`, `trainable_parameters=528384`, checkpoint `loadable=true`, `pe_checkpoint_rejected=true`를 기록했다.

ComfyUI gate는 isolated `127.0.0.1:8116`에서 진행했다. 비교 column은 `no_ip`, current best `blend_species_face`, c075 baseline `c075_tag_positive_w14`, c079 baseline `c079_synthetic_positive_w14`, 신규 `c080_paired_direct_green_w14`다. clean32+heldout8 `40` samples에서 `200` PNG를 만들고, c074 paired direct-green focus `10` samples에서 `50` PNG를 추가로 만들었다. 총 generated PNG는 `250`, blank image는 `0`이다. 초기 runner에서 `c079_synthetic_positive_w14` 라벨이 c080 checkpoint를 가리키는 mapping bug를 발견했고, c079 산출물 50개를 삭제한 뒤 실제 c079 checkpoint로 재생성했다. 생성 후 ComfyUI server를 종료했고 port `8116`은 닫혔다.

metric 결과는 c080이 목표에 실패했음을 보여준다. clean32+heldout8 PE mean uplift는 `blend_species_face=0.0608932152`, `c075=0.0262199253`, `c079=0.0329968661`, `c080=0.0229417309`이고, QwenVL mean uplift는 `blend_species_face=0.0421902567`, `c075=0.0349742755`, `c079=0.0338256791`, `c080=0.0341175169`다. QwenVL clean aggregate에서 c080이 c079보다 아주 조금 높지만 `0.00029` 수준이라 실질 개선으로 보기 어렵다. direct-green focus에서는 PE mean uplift가 `blend_species_face=0.0844765946`, `c075=0.0325176731`, `c079=0.0640649319`, `c080=0.0482934043`이고, QwenVL mean uplift는 `blend_species_face=-0.0102016628`, `c075=-0.0095478654`, `c079=0.0040470958`, `c080=-0.0087357759`다.

시각 검수 기준으로도 c080은 미승격이다. `contact_sheet_direct_green.jpg`에서 c080은 녹색 피부, 뿔/귀, 어두운 무협풍 복식 같은 큰 속성은 유지하지만 reference column의 밝은 색감, 여성형 얼굴, 작은 체형, 과장된 장식, 귀여운 표정, 캐릭터별 다른 실루엣은 거의 전달하지 못한다. 대부분 성인형 green humanoid villain으로 수렴하며, c079보다 안정적으로 낫다고 볼 수 없다. `contact_sheet_heldout.jpg`에서도 c080은 c075/c079와 거의 같은 결과를 내고, 핵심 실패인 비인간 side-profile/monster face는 여전히 인간형 dark villain으로 흡수된다.

c080 decision은 `not_promoted_c080_paired_direct_green_weaker_than_c079_and_blend`다. runtime은 pass지만 품질은 pass가 아니다. c074 10장 규모의 작은 paired supervision만으로는 QwenVL calibrator가 참조별 identity를 분리해 전달하지 못했다. 다음 루프는 c074 pair 반복이 아니라, 실제 paired source-target color/reference 데이터 확보, identity-preserving synthetic pair generation, 또는 QwenVL/SigLIP encoder-side reference feature objective 강화로 이동해야 한다.

## 32. 근거 파일 색인

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
- `eval/reviewed_seed_feature_probe_20260612_c042/report.md`
- `eval/broad_identity_candidate_mining_20260612_c043/report.md`
- `eval/reviewed_face_identity_candidates_20260612_c044/report.md`
- `eval/reviewed_face_seed_feature_probe_20260612_c045/report.md`
- `eval/qwenvl_ranked_identity_candidates_20260612_c046/report.md`
- `eval/qwenvl_top20_reviewed_identity_20260612_c047/report.md`
- `eval/qwenvl_combined_seed_feature_probe_20260612_c048/report.md`
- `eval/qwenvl_rank21_40_reviewed_identity_20260612_c049/report.md`
- `eval/qwenvl_combined_rank40_feature_probe_20260612_c050/report.md`
- `eval/qwenvl_diverse_identity_candidates_20260612_c051/report.md`
- `eval/qwenvl_combined_diverse_feature_probe_20260612_c052/report.md`
- `eval/qwenvl_c052_bounded_training_20260612_c053/report.md`
- `eval/qwenvl_c052_generation_gate_20260612_c054/report.md`
- `eval/qwenvl_c055_mixed_training_20260612/report.md`
- `eval/qwenvl_c055_generation_gate_20260612_c056/report.md`
- `eval/qwenvl_c055_runtime_blend_gate_20260612_c057/report.md`
- `eval/qwenvl_c055_larger_blend_gate_20260612_c058/report.md`
- `eval/qwenvl_c059_checkpoint_merge_gate_20260612/report.md`
- `eval/qwenvl_c060_failure_focused_training_20260612/report.md`
- `eval/qwenvl_c060_failure_focused_gate_20260612/report.md`
- `eval/qwenvl_c061_instruction_calibration_gate_20260612/report.md`
- `eval/qwenvl_c062_calibrator_distillation_training_20260612/report.md`
- `eval/qwenvl_c062_calibrator_distillation_gate_20260612/report.md`
- `eval/qwenvl_c062_calibrator_distillation_gate_20260612/visual_audit.md`
- `docs/c063_qwenvl_calibrator_only_plan_ko.md`
- `eval/qwenvl_c063_calibrator_only_training_20260612/report.md`
- `eval/qwenvl_c063_calibrator_only_gate_20260612/report.md`
- `eval/qwenvl_c063_calibrator_only_gate_20260612/visual_audit.md`
- `docs/c064_failure_attribute_embedding_probe_plan_ko.md`
- `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/report.md`
- `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/summary.json`
- `docs/c065_encoder_side_failure_attribute_plan_ko.md`
- `eval/c065_encoder_side_failure_attribute_20260612/report.md`
- `eval/c065_encoder_side_failure_attribute_20260612/summary.json`
- `docs/c066_direct_green_non_human_mining_plan_ko.md`
- `eval/c066_direct_green_non_human_mining_20260612/report.md`
- `eval/c066_direct_green_non_human_mining_20260612/summary.json`
- `eval/c067_attribute_teacher_reranker_seed_20260612/report.md`
- `eval/c067_attribute_teacher_reranker_seed_20260612/summary.json`
- `eval/c067_attribute_teacher_reranker_seed_20260612/visual_audit.md`
- `eval/c068_reviewed_attribute_label_seed_20260612/report.md`
- `eval/c068_reviewed_attribute_label_seed_20260612/summary.json`
- `docs/c069_direct_green_captioning_acquisition_plan_ko.md`
- `eval/c069_direct_green_captioning_acquisition_20260612/report.md`
- `eval/c069_direct_green_captioning_acquisition_20260612/summary.json`
- `docs/c070_qwenvl_direct_green_caption_search_plan_ko.md`
- `eval/c070_qwenvl_direct_green_caption_search_20260612/report.md`
- `eval/c070_qwenvl_direct_green_caption_search_20260612/summary.json`
- `docs/c071_direct_green_seed_package_plan_ko.md`
- `eval/c071_direct_green_seed_package_20260612/report.md`
- `eval/c071_direct_green_seed_package_20260612/summary.json`
- `docs/c072_external_direct_green_source_discovery_ko.md`
- `eval/c072_external_direct_green_source_discovery_20260612/report.md`
- `eval/c072_external_direct_green_source_discovery_20260612/summary.json`
- `eval/c073_external_candidate_visual_review_20260612/report.md`
- `eval/c073_external_candidate_visual_review_20260612/summary.json`
- `docs/c074_tag_backed_source_acquisition_ko.md`
- `eval/c074_tag_backed_direct_green_source_acquisition_20260612/report.md`
- `eval/c074_tag_backed_direct_green_source_acquisition_20260612/summary.json`

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
- `eval/qwenvl_c052_bounded_training_20260612_c053/report.md`
- `eval/qwenvl_c052_generation_gate_20260612_c054/report.md`
- `eval/qwenvl_c055_mixed_training_20260612/report.md`
- `eval/qwenvl_c055_generation_gate_20260612_c056/report.md`
- `eval/qwenvl_c055_generation_gate_20260612_c056/visual_audit.md`
- `eval/qwenvl_c055_generation_gate_20260612_c056/pe_similarity_metrics.json`
- `eval/qwenvl_c055_generation_gate_20260612_c056/qwenvl_similarity_metrics.json`
- `eval/qwenvl_c055_runtime_blend_gate_20260612_c057/report.md`
- `eval/qwenvl_c055_runtime_blend_gate_20260612_c057/visual_audit.md`
- `eval/qwenvl_c055_runtime_blend_gate_20260612_c057/pe_similarity_metrics.json`
- `eval/qwenvl_c055_runtime_blend_gate_20260612_c057/qwenvl_similarity_metrics.json`
- `eval/qwenvl_c055_larger_blend_gate_20260612_c058/report.md`
- `eval/qwenvl_c055_larger_blend_gate_20260612_c058/visual_audit.md`
- `eval/qwenvl_c055_larger_blend_gate_20260612_c058/pe_similarity_metrics.json`
- `eval/qwenvl_c055_larger_blend_gate_20260612_c058/qwenvl_similarity_metrics.json`
- `eval/qwenvl_c059_checkpoint_merge_gate_20260612/report.md`
- `eval/qwenvl_c059_checkpoint_merge_gate_20260612/visual_audit.md`
- `eval/qwenvl_c059_checkpoint_merge_gate_20260612/merge_summary.json`
- `eval/qwenvl_c059_checkpoint_merge_gate_20260612/pe_similarity_metrics.json`
- `eval/qwenvl_c059_checkpoint_merge_gate_20260612/qwenvl_similarity_metrics.json`
- `eval/qwenvl_c060_failure_focused_training_20260612/report.md`
- `eval/qwenvl_c060_failure_focused_gate_20260612/report.md`
- `eval/qwenvl_c060_failure_focused_gate_20260612/visual_audit.md`
- `eval/qwenvl_c060_failure_focused_gate_20260612/pe_similarity_metrics.json`
- `eval/qwenvl_c060_failure_focused_gate_20260612/qwenvl_similarity_metrics.json`
- `eval/qwenvl_c061_instruction_calibration_gate_20260612/report.md`
- `eval/qwenvl_c061_instruction_calibration_gate_20260612/visual_audit.md`
- `eval/qwenvl_c061_instruction_calibration_gate_20260612/pe_similarity_metrics.json`
- `eval/qwenvl_c061_instruction_calibration_gate_20260612/qwenvl_similarity_metrics.json`
- `eval/qwenvl_c062_calibrator_distillation_training_20260612/report.md`
- `eval/qwenvl_c062_calibrator_distillation_gate_20260612/report.md`
- `eval/qwenvl_c062_calibrator_distillation_gate_20260612/visual_audit.md`
- `eval/qwenvl_c062_calibrator_distillation_gate_20260612/pe_similarity_metrics.json`
- `eval/qwenvl_c062_calibrator_distillation_gate_20260612/qwenvl_similarity_metrics.json`
- `eval/qwenvl_c063_calibrator_only_training_20260612/report.md`
- `eval/qwenvl_c063_calibrator_only_gate_20260612/report.md`
- `eval/qwenvl_c063_calibrator_only_gate_20260612/visual_audit.md`
- `eval/qwenvl_c063_calibrator_only_gate_20260612/pe_similarity_metrics.json`
- `eval/qwenvl_c063_calibrator_only_gate_20260612/qwenvl_similarity_metrics.json`
- `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/report.md`
- `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/summary.json`
- `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/qwenvl_probe_metrics.json`
- `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/siglip_probe_metrics.json`
- `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/pe_probe_metrics.json`
- `eval/c065_encoder_side_failure_attribute_20260612/qwenvl_pair_probe.json`
- `eval/c065_encoder_side_failure_attribute_20260612/siglip_pair_probe.json`
- `eval/c065_encoder_side_failure_attribute_20260612/pe_pair_probe.json`
- `eval/c066_direct_green_non_human_mining_20260612/qwenvl_pair_probe.json`
- `eval/c066_direct_green_non_human_mining_20260612/siglip_pair_probe.json`
- `eval/c066_direct_green_non_human_mining_20260612/pe_pair_probe.json`
- `eval/c067_attribute_teacher_reranker_seed_20260612/attribute_scores.jsonl`
- `eval/c067_attribute_teacher_reranker_seed_20260612/attribute_topk.json`
- `eval/c067_attribute_teacher_reranker_seed_20260612/attribute_review_sheet.jpg`
- `eval/c068_reviewed_attribute_label_seed_20260612/reviewed_attribute_labels.jsonl`
- `eval/c068_reviewed_attribute_label_seed_20260612/annotated_review_sheet.jpg`
- `eval/c069_direct_green_captioning_acquisition_20260612/candidate_manifest.jsonl`
- `eval/c069_direct_green_captioning_acquisition_20260612/reviewed_candidate_labels.jsonl`
- `eval/c069_direct_green_captioning_acquisition_20260612/annotated_review_sheet.jpg`
- `eval/c070_qwenvl_direct_green_caption_search_20260612/candidate_manifest.jsonl`
- `eval/c070_qwenvl_direct_green_caption_search_20260612/reviewed_candidate_labels.jsonl`
- `eval/c070_qwenvl_direct_green_caption_search_20260612/annotated_review_sheet.jpg`
- `eval/c071_direct_green_seed_package_20260612/annotation_candidates.jsonl`
- `eval/c071_direct_green_seed_package_20260612/annotation_template.csv`
- `eval/c071_direct_green_seed_package_20260612/annotated_review_sheet.jpg`
- `eval/c071_direct_green_seed_package_20260612/manual_labels_example.csv`
- `eval/c071_direct_green_seed_package_20260612/example_import/import_summary.json`
- `eval/c071_direct_green_seed_package_20260612/example_import/imported_confirmed_positives.jsonl`
- `eval/c072_external_direct_green_source_discovery_20260612/source_manifest.jsonl`
- `eval/c072_external_direct_green_source_discovery_20260612/external_candidates.jsonl`
- `eval/c072_external_direct_green_source_discovery_20260612/external_candidate_template.csv`
- `eval/c073_external_candidate_visual_review_20260612/download_manifest.jsonl`
- `eval/c073_external_candidate_visual_review_20260612/reviewed_external_labels.jsonl`
- `eval/c073_external_candidate_visual_review_20260612/summary.json`
- `eval/c074_tag_backed_direct_green_source_acquisition_20260612/source_manifest.jsonl`
- `eval/c074_tag_backed_direct_green_source_acquisition_20260612/external_candidates.jsonl`
- `eval/c074_tag_backed_direct_green_source_acquisition_20260612/reviewed_external_labels.jsonl`
- `eval/c074_tag_backed_direct_green_source_acquisition_20260612/summary.json`
- `eval/qwenvl_c075_tag_positive_training_20260612/report.md`
- `eval/qwenvl_c075_tag_positive_training_20260612/summary.json`
- `eval/qwenvl_c075_tag_positive_gate_20260612/report.md`
- `eval/qwenvl_c075_tag_positive_gate_20260612/visual_audit.md`
- `eval/qwenvl_c075_tag_positive_gate_20260612/pe_similarity_metrics.json`
- `eval/qwenvl_c075_tag_positive_gate_20260612/qwenvl_similarity_metrics.json`
- `eval/qwenvl_c075_tag_positive_gate_20260612/direct_green_pe_similarity_metrics.json`
- `eval/qwenvl_c075_tag_positive_gate_20260612/direct_green_qwenvl_similarity_metrics.json`
- `docs/c076_paired_direct_green_source_expansion_plan_ko.md`
- `eval/c076_paired_direct_green_source_expansion_20260612/report.md`
- `eval/c076_paired_direct_green_source_expansion_20260612/summary.json`
- `eval/c076_paired_direct_green_source_expansion_20260612/source_manifest.jsonl`
- `eval/c076_paired_direct_green_source_expansion_20260612/external_candidates.jsonl`
- `eval/c076_paired_direct_green_source_expansion_20260612/download_manifest.jsonl`
- `eval/c076_paired_direct_green_source_expansion_20260612/reviewed_external_labels.jsonl`
- `eval/c076_paired_direct_green_source_expansion_20260612/manual_visual_labels.csv`
- `eval/c076_paired_direct_green_source_expansion_20260612/visual_audit.md`
- `eval/c076_paired_direct_green_source_expansion_20260612/feature_boundary_metrics.json`
- `docs/c077_direct_green_target_positive_acquisition_plan_ko.md`
- `eval/c077_direct_green_target_positive_acquisition_20260612/report.md`
- `eval/c077_direct_green_target_positive_acquisition_20260612/summary.json`
- `eval/c077_direct_green_target_positive_acquisition_20260612/source_manifest.jsonl`
- `eval/c077_direct_green_target_positive_acquisition_20260612/candidate_manifest.jsonl`
- `eval/c077_direct_green_target_positive_acquisition_20260612/download_manifest.jsonl`
- `eval/c077_direct_green_target_positive_acquisition_20260612/reviewed_external_labels.jsonl`
- `eval/c077_direct_green_target_positive_acquisition_20260612/manual_visual_labels.csv`
- `eval/c077_direct_green_target_positive_acquisition_20260612/visual_audit.md`
- `docs/c078_synthetic_direct_green_bootstrap_plan_ko.md`
- `eval/c078_synthetic_direct_green_bootstrap_20260612/report.md`
- `eval/c078_synthetic_direct_green_bootstrap_20260612/summary.json`
- `eval/c078_synthetic_direct_green_bootstrap_20260612/generation_summary.json`
- `eval/c078_synthetic_direct_green_bootstrap_20260612/prompt_manifest.jsonl`
- `eval/c078_synthetic_direct_green_bootstrap_20260612/generation_manifest.jsonl`
- `eval/c078_synthetic_direct_green_bootstrap_20260612/reviewed_synthetic_labels.jsonl`
- `eval/c078_synthetic_direct_green_bootstrap_20260612/manual_visual_labels.csv`
- `eval/c078_synthetic_direct_green_bootstrap_20260612/visual_audit.md`
- `eval/c078_synthetic_direct_green_bootstrap_20260612/visual_label_template.csv`
- `eval/qwenvl_c079_synthetic_positive_training_20260612/report.md`
- `eval/qwenvl_c079_synthetic_positive_training_20260612/summary.json`
- `eval/qwenvl_c079_synthetic_positive_gate_20260612/report.md`
- `eval/qwenvl_c079_synthetic_positive_gate_20260612/visual_audit.md`
- `eval/qwenvl_c079_synthetic_positive_gate_20260612/pe_similarity_metrics.json`
- `eval/qwenvl_c079_synthetic_positive_gate_20260612/qwenvl_similarity_metrics.json`
- `eval/qwenvl_c079_synthetic_positive_gate_20260612/direct_green_pe_similarity_metrics.json`
- `eval/qwenvl_c079_synthetic_positive_gate_20260612/direct_green_qwenvl_similarity_metrics.json`
- `eval/qwenvl_c080_paired_direct_green_training_20260613/report.md`
- `eval/qwenvl_c080_paired_direct_green_training_20260613/summary.json`
- `eval/qwenvl_c080_paired_direct_green_gate_20260613/report.md`
- `eval/qwenvl_c080_paired_direct_green_gate_20260613/visual_audit.md`
- `eval/qwenvl_c080_paired_direct_green_gate_20260613/pe_similarity_metrics.json`
- `eval/qwenvl_c080_paired_direct_green_gate_20260613/qwenvl_similarity_metrics.json`
- `eval/qwenvl_c080_paired_direct_green_gate_20260613/direct_green_pe_similarity_metrics.json`
- `eval/qwenvl_c080_paired_direct_green_gate_20260613/direct_green_qwenvl_similarity_metrics.json`

생성/학습 manifest:

- `training/manifests/local_color_pairs_pilot_20260610.jsonl`
- `training/manifests/local_color_self_identity8_20260611.jsonl`
- `training/manifests/local_color_self_identity128_20260611.jsonl`
- `training/manifests/local_color_single_character_identity4_20260611.jsonl`
- `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- `training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl`
- `training/manifests/c052_positive_identity_pairs_20260612.jsonl`
- `training/manifests/c055_qwenvl_mixed_clean32_c052_positive_20260612.jsonl`
- `training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.jsonl`
- `training/manifests/c065_failure_attribute_pairs_20260612.jsonl`
- `training/manifests/c065_failure_attribute_pairs_20260612.summary.json`
- `training/manifests/c066_direct_green_non_human_candidates_20260612.jsonl`
- `training/manifests/c066_direct_green_non_human_candidates_20260612.summary.json`
- `training/manifests/c066_direct_green_non_human_pairs_20260612.jsonl`
- `training/manifests/c075_tag_positive_direct_green_20260612.jsonl`
- `training/manifests/c075_tag_positive_direct_green_20260612.summary.json`
- `training/manifests/c079_synthetic_positive_direct_green_20260612.jsonl`
- `training/manifests/c079_synthetic_positive_direct_green_20260612.summary.json`
- `training/manifests/c080_paired_direct_green_identity_20260613.jsonl`
- `training/manifests/c080_paired_direct_green_identity_20260613.summary.json`
- `eval/c067_attribute_teacher_reranker_seed_20260612/attribute_query_manifest.jsonl`
- `eval/c068_reviewed_attribute_label_seed_20260612/reviewed_attribute_labels.jsonl`

현재 가장 중요한 실행 레시피 근거:

- `tools/build_reference_prompt_manifest.py`
- `tools/c067_attribute_teacher_core.py`
- `tools/c068_reviewed_attribute_labels.py`
- `tests/test_c068_reviewed_attribute_labels.py`
- `tools/c069_direct_green_acquisition.py`
- `tools/c069_review_sheet.py`
- `tests/test_c069_direct_green_acquisition.py`
- `tools/c070_qwenvl_caption_search.py`
- `tools/c070_color_metrics.py`
- `tests/test_c070_qwenvl_caption_search.py`
- `tools/c071_seed_package.py`
- `tools/c071_import_manual_labels.py`
- `tests/test_c071_manual_seed_package.py`
- `tools/c072_external_source_discovery.py`
- `tools/c072_source_probe.py`
- `tests/test_c072_external_source_discovery.py`
- `tools/c073_external_candidate_visual_review.py`
- `tests/test_c073_external_candidate_visual_review.py`
- `tools/c074_tag_backed_source_acquisition.py`
- `tests/test_c074_tag_backed_source_acquisition.py`
- `tools/c075_tag_positive_manifest.py`
- `tools/c075_manifest_files.py`
- `tools/c075_tag_positive_manifest_types.py`
- `tests/test_c075_tag_positive_manifest.py`
- `tools/c076_paired_source_expansion.py`
- `tools/c076_source_expansion_io.py`
- `tools/c076_source_expansion_report.py`
- `tests/test_c076_paired_source_expansion.py`
- `tools/c077_hf_sample_sources.py`
- `tools/c077_target_positive_acquisition.py`
- `tools/c077_acquisition_report.py`
- `tests/test_c077_target_positive_acquisition.py`
- `tools/c078_synthetic_bootstrap.py`
- `tools/c078_comfy_generation.py`
- `tests/test_c078_synthetic_bootstrap.py`
- `tools/c079_manifest_types.py`
- `tools/c079_manifest_io.py`
- `tools/c079_synthetic_positive_manifest.py`
- `tests/test_c079_synthetic_positive_manifest.py`
- `tools/c080_paired_direct_green_manifest.py`
- `tests/test_c080_paired_direct_green_manifest.py`
- `tools/siglip_auto_caption_eval.py`
- `tools/score_siglip_auto_caption_metrics.py`
- `workflows/anima_ipadapter_siglip_native_reference.json`
