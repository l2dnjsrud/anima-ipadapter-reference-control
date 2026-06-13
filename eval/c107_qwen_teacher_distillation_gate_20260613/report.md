# C107 Qwen Teacher Distillation Generation Gate

작성일: 2026-06-14 KST

## 목적

C106에서 QwenVL teacher가 hard-shape positive/explicit-negative 쌍을 강하게 분리했다. C107은 그 teacher 판단을 `neg_id` contrastive/retrieval manifest로 변환해 128-step calibrator-only 학습을 수행했고, 이 gate에서는 실제 ComfyUI/API 생성 결과가 current best `blend_species_face`를 넘는지 확인했다.

## 실행 표면

- ComfyUI: isolated root `.tmp/comfy_siglip_base`, port `8116`
- 모델 노출 확인: `/object_info/AnimaQwenVLIPAdapterLoader`
- 비교 variant:
  - `no_ip`
  - `blend_species_face`: `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors` weight `1.4` + `anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors` weight `0.4`
  - `c107_qwen_teacher_w14`: `anima_qwenvl_ip_adapter_c107_qwen_teacher_calibrator_b128_0128_20260613.safetensors` weight `1.4`
- 평가 샘플: train `32`, heldout `8`, 총 `40`
- 생성 결과: `120` PNG, blank `0`
- cleanup: `cleanup_port_8116.txt`에서 `curl_exit=7`, `lsof` listener 없음

ComfyUI02 venv에는 `sentence_transformers`가 없어 첫 실행이 실패했다. root-owned venv를 직접 변경하지 않고 `.tmp/comfy_siglip_base/sentence_only` thin `PYTHONPATH`를 붙여 재시작했고, 이후 object_info와 생성 gate가 정상 완료됐다. `run_stdout.txt` 상단에는 이 초기 실패 로그가 남아 있다.

## 자동 지표

### PE metric

| variant | split | cases | mean cosine | mean uplift | improved rate |
|---|---:|---:|---:|---:|---:|
| `blend_species_face` | train | 32 | 0.817763 | 0.062733 | 0.8125 |
| `blend_species_face` | heldout | 8 | 0.804082 | 0.053534 | 0.8750 |
| `blend_species_face` | all | 40 | 0.815027 | 0.060893 | 0.8250 |
| `c107_qwen_teacher_w14` | train | 32 | 0.783889 | 0.028859 | 0.7188 |
| `c107_qwen_teacher_w14` | heldout | 8 | 0.756995 | 0.006447 | 0.6250 |
| `c107_qwen_teacher_w14` | all | 40 | 0.778510 | 0.024377 | 0.7000 |

PE direct comparison에서 C107은 blend 대비 `11/40`만 승리했고, 평균 차이는 `-0.036517`이다.

### QwenVL metric

| variant | split | cases | mean cosine | mean uplift | improved rate |
|---|---:|---:|---:|---:|---:|
| `blend_species_face` | train | 32 | 0.820411 | 0.046120 | 0.8125 |
| `blend_species_face` | heldout | 8 | 0.834693 | 0.026471 | 0.7500 |
| `blend_species_face` | all | 40 | 0.823268 | 0.042190 | 0.8000 |
| `c107_qwen_teacher_w14` | train | 32 | 0.811002 | 0.036711 | 0.7500 |
| `c107_qwen_teacher_w14` | heldout | 8 | 0.837083 | 0.028860 | 0.7500 |
| `c107_qwen_teacher_w14` | all | 40 | 0.816218 | 0.035141 | 0.7500 |

QwenVL direct comparison에서 C107은 heldout 평균만 blend보다 `+0.002390` 높다. 전체 평균은 `-0.007049`로 blend가 앞서며, direct win은 `14/40`이다.

## 육안 감사

`visual_audit.md/json` 기준으로 C107은 `no_ip` 대비 reference-control이 켜진다. 다만 current best blend보다 더 강한 identity lock은 보이지 않았다. 주요 속성은 따라가지만 나이, 얼굴형, 수염, 대머리, 관모, 소품, 말풍선 맥락 같은 세부 단서가 안정적으로 개선되지 않는다.

## 결정

결정: `c107_generation_gate_not_promoted`

C107은 “Qwen teacher score를 사용해 finite/loadable calibrator checkpoint를 만들 수 있다”는 학습 증거다. 그러나 실제 생성 gate에서는 current best `blend_species_face`를 넘지 못했다. 다음 루프는 standalone C107 승격이 아니라, teacher embedding/target 직접 정렬 또는 blend 가능한 Qwen teacher route로 이동한다.

## 주요 산출물

- `summary.json`
- `metric_rollup.json`
- `pe_similarity_metrics.json`
- `qwenvl_similarity_metrics.json`
- `contact_sheet_train.jpg`
- `contact_sheet_heldout.jpg`
- `visual_audit.md`
- `visual_audit.json`
- `cleanup_port_8116.txt`
