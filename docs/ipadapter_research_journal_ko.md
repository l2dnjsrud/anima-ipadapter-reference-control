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

## 12. 근거 파일 색인

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

현재 가장 중요한 실행 레시피 근거:

- `tools/build_reference_prompt_manifest.py`
- `tools/siglip_auto_caption_eval.py`
- `tools/score_siglip_auto_caption_metrics.py`
- `workflows/anima_ipadapter_siglip_native_reference.json`
