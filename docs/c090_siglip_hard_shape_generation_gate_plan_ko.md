# c090 SigLIP Hard-Shape Generation Gate 계획

## 목적

c089는 PE-teacher hard-shape distillation pilot이 학습 가능하고, checkpoint가 SigLIP loader에 loadable하다는 것을 확인했다. 하지만 이것은 생성 품질 검증이 아니다. c090의 목적은 c089 checkpoint를 native SigLIP ComfyUI 경로에 올려 실제 hard-shape reference-control 생성 결과가 좋아졌는지 확인하는 것이다.

## 검사한 surface

- `native_siglip.py`: `AnimaSigLIPIPAdapterLoader`, `AnimaSigLIPEncodeImage`, `AnimaSigLIPIPAdapterApply`
- `workflows/anima_ipadapter_siglip_native_reference.json`: 표준 ComfyUI graph 형태의 SigLIP workflow
- `tools/siglip_auto_caption_eval.py`: SigLIP API prompt construction과 SaveImage/API 기록 방식
- `tools/c088_probe_manifest.py`, `tools/c088_shape_metrics.py`: c087/c088 hard-shape reference와 edge/projection/silhouette metric
- `.tmp/run_c087_qwenvl_expanded_crop_positive_eval.py`: current QwenVL hard-shape baseline output set
- `tools/comfyui_extra_model_paths.yaml`: `/data/ai/models`와 repo-local `checkpoints/`를 `ipadapter` selector에 노출

## Checkpoint 노출 방식

source checkpoint:

```text
checkpoints/anima_siglip_ip_adapter_c089_shape_pe_teacher_0032_20260613.safetensors
```

`/data/ai/models/ipadapter`는 root 소유라 현재 사용자로 직접 쓰기 어렵다. c090에서는 source artifact를 이동하지 않고, isolated ComfyUI를 다음 방식으로 띄운다.

```text
/data/ai/comfyui02/.venv/bin/python3 /data/ai/comfyui02/main.py \
  --port 8116 \
  --base-directory .tmp/comfy_siglip_base \
  --extra-model-paths-config tools/comfyui_extra_model_paths.yaml \
  --input-directory .tmp/comfy_siglip_base/input \
  --output-directory .tmp/comfy_siglip_base/output \
  --temp-directory .tmp/comfy_siglip_base/temp \
  --user-directory .tmp/comfy_siglip_base/user \
  --disable-auto-launch --disable-manager-ui --log-stdout
```

이 설정은 `repo_ipadapter: checkpoints`를 통해 c089 checkpoint를 ComfyUI `ipadapter_name` selector에 노출한다.

## 평가 샘플

입력은 c088 hard-shape probe manifest를 사용한다.

```text
eval/c088_shape_silhouette_feature_probe_20260613/probe_manifest.jsonl
```

구성:

- c087 crop-pair hard-shape rows 10개
- heldout07 non-human side-profile hard case 1개
- heldout training rows used: 0

## 비교 variant

새로 생성하는 SigLIP variants:

- `no_ip`
- `siglip_pilot_w14`: 기존 `anima_siglip_ip_adapter_pilot_20260610.safetensors`
- `c089_shape_w10`: c089 checkpoint weight 1.0
- `c089_shape_w14`: c089 checkpoint weight 1.4

contact sheet에는 기존 hard-shape QwenVL baseline도 같은 row에 복사해서 비교한다.

- `blend_species_face`
- `c086_hard_negative_w14`
- `c087_expanded_crop_positive_w14`

## 산출물

```text
eval/c090_siglip_hard_shape_generation_gate_20260613/readiness.json
eval/c090_siglip_hard_shape_generation_gate_20260613/summary.json
eval/c090_siglip_hard_shape_generation_gate_20260613/contact_sheet_hard_shape.jpg
eval/c090_siglip_hard_shape_generation_gate_20260613/shape_metrics.json
eval/c090_siglip_hard_shape_generation_gate_20260613/metric_rollup.json
eval/c090_siglip_hard_shape_generation_gate_20260613/visual_audit.md
eval/c090_siglip_hard_shape_generation_gate_20260613/report.md
```

Raw PNG/API prompt/response/history 파일은 eval 폴더에 보관한다. c090은 실제 결과 검토가 중요해서 contact sheet, summary, metric, visual audit, cleanup receipt와 함께 raw PNG/API receipt도 같은 eval 폴더에 남긴다.

## 판정 기준

c090는 최종 production gate가 아니라 hard-shape branch gate다.

- PASS 후보: c089가 기존 SigLIP pilot보다 shape metric과 visual audit에서 개선되고, current QwenVL hard-shape baseline에 근접한다.
- 부분 개선: c089가 SigLIP pilot은 넘지만 QwenVL baseline에는 못 미친다. 이 경우 SigLIP PE-teacher tuning을 더 키울지, encoder-side checkpoint로 갈지 결정한다.
- 실패: c089가 SigLIP pilot/no-IP 대비 개선이 없거나 shape collapse가 유지된다. 이 경우 frozen SigLIP adapter tuning은 중단하고 encoder-side shape checkpoint 학습으로 넘어간다.

## Cleanup

각 ComfyUI API 실행 후 다음을 확인한다.

```text
curl http://127.0.0.1:8116/object_info -> connection refused
lsof -i :8116 -sTCP:LISTEN -> no listener
```

runtime process가 남아 있으면 해당 criterion은 PASS로 기록하지 않는다.

## Runtime 결과

상태: `final_runtime_results`

- 실행일: 2026-06-13
- isolated ComfyUI: `http://127.0.0.1:8116`
- source checkpoint: `checkpoints/anima_siglip_ip_adapter_c089_shape_pe_teacher_0032_20260613.safetensors`
- exposure: `tools/comfyui_extra_model_paths.yaml`의 `repo_ipadapter: checkpoints`
- `/data/ai/models/ipadapter` 직접 복사는 하지 않음
- source_not_moved: `true`
- generated: 11 samples x 4 SigLIP variants = 44 PNG
- contact sheet: `eval/c090_siglip_hard_shape_generation_gate_20260613/contact_sheet_hard_shape.jpg`
- pixel audit: `low_variance_count = 2`
- cleanup: `eval/c090_siglip_hard_shape_generation_gate_20260613/cleanup_port_8116.txt`에 `port_8116_closed_or_refused` 기록

metric rollup:

| variant | mean uplift | improved rate |
|---|---:|---:|
| `siglip_pilot_w14` | `-0.0662400384` | `0.1818181818` |
| `c089_shape_w10` | `0.0027134620` | `0.5454545455` |
| `c089_shape_w14` | `0.0249214356` | `0.7272727273` |
| `c086_hard_negative_w14` | `0.0501749091` | `0.7272727273` |
| `c087_expanded_crop_positive_w14` | `0.1089544056` | `0.9090909091` |

판정:

`c089_partial_siglip_improvement_not_promoted_escalate_encoder_side`

c089는 prior SigLIP pilot보다 개선되었지만, current QwenVL hard-shape baseline에는 못 미친다. 특히 frog/yokai crop에서 reference의 chibi/non-human shape를 일부 당기지만 green humanoid face로 수렴하는 현상이 남고, heldout07 non-human side-profile identity는 보존하지 못한다. 따라서 c089는 promotion하지 않고, 다음 루프는 stronger encoder-side/feature adaptation으로 이동한다.
