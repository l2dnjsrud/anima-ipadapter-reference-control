# SigLIP2 Native Branch And Training Plan

## What landed locally

- Native SigLIP2/IP-Adapter scaffolding lives in `siglip_model.py`, `siglip_checkpoint.py`, and `native_siglip.py`.
- Real frozen-Anima SigLIP2 smoke training lives in `training/siglip_real_smoke.py` with small helpers under `training/siglip_smoke_*.py`.
- The upstream Wenaka `TransformerEncoderLayer(memory=...)` bug is avoided by a real cross-attention fusion layer.
- The upstream loader constructor mismatch is avoided by detecting checkpoint tensor shapes and constructing `IPAdapterSigLIP` with explicit dimensions.
- PE-Core checkpoints are rejected by the SigLIP loader with a clear message instead of being partially loaded as SigLIP.
- `AnimaSigLIPIPAdapterLoader` uses the ComfyUI `ipadapter` model selector, not a raw filesystem path.
- `AnimaSigLIPIPAdapterApply` clones the ComfyUI `MODEL` and installs a PE-style UNet/model wrapper that temporarily patches the live Anima DiT `cross_attn.forward` calls during sampling. The previous `attn2_patch`-only path can execute as a node but is not sufficient for the Anima/Qwen workflow surface.
- A one-step smoke checkpoint was written to `checkpoints/anima_siglip_ip_adapter_smoke_20260610.safetensors` and loads through `siglip_checkpoint.load_siglip_adapter`.

## Dataset facts checked on 2026-06-10

Primary source: `https://huggingface.co/datasets/Wenaka/anima-ip-adapter-dataset`

- Dataset id: `Wenaka/anima-ip-adapter-dataset`
- Commit: `687b09ba08d8fda8c415b43cc0aacf6fc1d2661c`
- Public and ungated.
- Hub API `usedStorage`: `39,190,230,897` bytes, about `36.5 GiB`.
- Direct HEAD requests for `images_00.tar` through `images_13.tar` total `34,190,796,800` bytes, about `31.843 GiB`; the difference is Hub storage metadata and non-tar overhead.
- Files: `.gitattributes` plus `images_00.tar` through `images_13.tar`.
- Dataset Viewer split: `default/train`.
- Preview features: `jpg` image, `__key__` string, `__url__` string.
- Preview rows show 1024 x 1024 JPG images inside the tar shards.
- Dataset Viewer reported preview support, but no full viewer/search/filter/statistics support; `/size` returned a failed `config-size` result, so the Hub file metadata is the reliable size source.

Commands used:

```bash
curl 'https://datasets-server.huggingface.co/is-valid?dataset=Wenaka/anima-ip-adapter-dataset'
curl 'https://datasets-server.huggingface.co/splits?dataset=Wenaka/anima-ip-adapter-dataset'
curl 'https://datasets-server.huggingface.co/first-rows?dataset=Wenaka/anima-ip-adapter-dataset&config=default&split=train'
curl 'https://huggingface.co/api/datasets/Wenaka/anima-ip-adapter-dataset'
curl 'https://huggingface.co/api/datasets/Wenaka/anima-ip-adapter-dataset/tree/main?recursive=false&expand=true'
python - <<'PY'
import urllib.request
total = 0
for idx in range(14):
    request = urllib.request.Request(
        f"https://huggingface.co/datasets/Wenaka/anima-ip-adapter-dataset/resolve/main/images_{idx:02d}.tar",
        method="HEAD",
    )
    with urllib.request.urlopen(request) as response:
        total += int(response.headers["Content-Length"])
print(total)
PY
```

## Dry-run proof

The proof script does not download data and does not start full Anima training. It validates optional Wenaka-style pair rows and runs one synthetic optimization step through:

`SigLIP deep+shallow features -> CrossLayerEncoder -> TimeResampler -> IPCrossAttn`

Run with the existing Anima virtualenv:

```bash
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_proof.py
```

With local paired metadata:

```bash
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_proof.py \
  --pairs-path /path/to/training_pairs_final2.jsonl \
  --image-dir /path/to/extracted/images \
  --rows-to-check 8
```

Expected pair row shape:

```json
{"ref_id": "4279862", "tgt_id": "4279889", "prompt": "clean character reference prompt"}
```

## Local color-panel real smoke

The first real, bounded SigLIP2 smoke was run on 2026-06-10 with the curated
local color-panel dataset:

```text
/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best
```

Generated manifest:

```text
training/manifests/local_color_pairs_pilot_20260610.jsonl
training/manifests/local_color_pairs_pilot_20260610.summary.json
```

Manifest audit summary:

- 1,537 pair rows
- 1,460 train rows and 77 validation rows by deterministic sorted 95/5 split
- every row has `ref_id`, `tgt_id`, and `prompt`
- every referenced JPG and TXT sidecar exists
- malformed missing-caption input fails safely

Real smoke command:

```bash
HF_HUB_DISABLE_XET=1 /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_real_smoke.py \
  --manifest-path training/manifests/local_color_pairs_pilot_20260610.jsonl \
  --image-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  --steps 1 \
  --resolution 256 \
  --device cuda:0 \
  --output-path checkpoints/anima_siglip_ip_adapter_smoke_20260610.safetensors \
  --max-rows 4
```

Observed result:

```json
{
  "steps": 1,
  "rows_loaded": 4,
  "first_loss": 0.1699729710817337,
  "final_loss": 0.1699729710817337,
  "finite_loss": true,
  "trainable_parameters": 335860892,
  "frozen_base_parameters": 2913827059,
  "checkpoint": {
    "output_path": "checkpoints/anima_siglip_ip_adapter_smoke_20260610.safetensors",
    "loadable": true,
    "pe_checkpoint_rejected": true
  }
}
```

The canonical write target
`/data/ai/models/ipadapter/anima_siglip_ip_adapter_smoke_20260610.safetensors`
was attempted after the real training step reached checkpoint save, but the
directory is `root:root` with mode `755`, so `safetensors` returned
`Permission denied (os error 13)`. The repo-local checkpoint is the current
loadable smoke artifact.

## Local color-panel pilot

A bounded 16-step pilot was run on 2026-06-10 from the same local color-panel
manifest, using 32 rows at 256 px resolution:

```bash
HF_HUB_DISABLE_XET=1 /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_real_smoke.py \
  --manifest-path training/manifests/local_color_pairs_pilot_20260610.jsonl \
  --image-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
  --steps 16 \
  --resolution 256 \
  --device cuda:0 \
  --output-path checkpoints/anima_siglip_ip_adapter_pilot_20260610.safetensors \
  --max-rows 32
```

Observed result:

- final_loss: `0.1321239024400711`
- mean_loss: `0.22480368381366134`
- finite_loss: `true`
- trainable_parameters: `335860892`
- checkpoint: `checkpoints/anima_siglip_ip_adapter_pilot_20260610.safetensors`
- SigLIP loader: loadable as `IPAdapterSigLIP`
- PE-Core checkpoint: rejected by the SigLIP loader, as intended

Proxy evaluation artifacts:

```text
eval/siglip_color_pilot_20260610/metrics.json
eval/siglip_color_pilot_20260610/report.md
```

Proxy result:

- key_match: `true`
- common_tensors: `255`
- changed_tensors: `142`
- relative_l2_delta: `0.0008295682970535739`
- decision: `scale_after_siglip_workflow_eval`

This proves the pilot moved away from the one-step smoke checkpoint and stayed
checkpoint-compatible. It does not prove visual quality yet because the native
SigLIP ComfyUI/API image-generation workflow has not produced contact sheets.

## What blocks real full training

- No high-quality trained SigLIP2 TimeResampler/IPCrossAttn checkpoint is present in this repo. The current SigLIP checkpoints are a one-step smoke artifact and a 16-step pilot artifact only.
- The Hugging Face dataset is about 36.5 GiB; I did not download it without explicit approval.
- The public dataset appears to contain image tar shards only. Wenaka's training script expects a paired `training_pairs_final2.jsonl` with `ref_id`, `tgt_id`, and `prompt`, which is not exposed in the dataset preview or file list.
- Full training still needs explicit runtime approval, a target output path that the current user can write, and a native SigLIP reference-control contact-sheet quality gate.

## Full-training outline

1. Choose the approved dataset scope: local color-panel pairs, downloaded Wenaka shards, or both.
2. Generate paired metadata with `ref_id`, `tgt_id`, and `prompt`.
3. Run the dry-run proof and local manifest audit against the selected rows.
4. Run bounded real smoke through the frozen Anima DiT/VAE/text-encoder loss path.
5. Save a SigLIP checkpoint with `resampler.time_proj.*`, `resampler.layers.*`, `intermediate_encoder.*`, `ip_cross_attns.*`, and `ip_scales.*` keys.
6. Load it through `AnimaSigLIPIPAdapterLoader`; do not use the PE-Core checkpoint with this loader.
7. Wire the native SigLIP ComfyUI/API workflow, then evaluate against no-IP and PE-Core baselines with contact sheets before calling it usable.

## 2026-06-11 runtime and tuning result

The native SigLIP apply path was changed from an `attn2_patch`-only node to a
PE-style sampling wrapper that temporarily patches the live Anima DiT
`cross_attn.forward` calls. This fixed the zero-effect bug:

- no-IP and `weight=0` are pixel-identical,
- `weight>0` changes generated pixels,
- the node can produce good-looking prompt-aligned comic panels.

However, this does **not** prove a finished reference-control model. The latest
quality run is documented in:

```text
eval/siglip_runtime_quality_20260611_c007_self512_identity/report.md
```

Summary:

- `color64_continue` improved prompt-aligned visual strength when the prompt
  explicitly contained old monk / white beard / prayer beads hints.
- `color64_continue`, `self64_continue`, and `self512_continue` all failed the
  stricter identity test where those identity tokens were removed from the
  prompt.
- The best matched-prompt contact sheet is:
  `eval/siglip_runtime_quality_20260611_c004_color64_matched/contact_sheet.jpg`.
- The identity failure evidence is:
  `eval/siglip_runtime_quality_20260611_c007_self512_identity/contact_sheet.jpg`.

Therefore the current local SigLIP checkpoints should be treated as research
artifacts, not a ready-to-trust IP-Adapter. More short tuning on adjacent panel
pairs is unlikely to solve identity control. The next training step needs
identity-aware or same-image reconstruction data, cached feature training for
longer runs, and a validation gate that removes identity words from the prompt.

## 2026-06-11 one-image overfit gate

A one-image overfit gate was run after the `self512` identity failure. It used
the same `codex_contact_ref03.png` monk image as both reference and target, but
the training and evaluation prompt intentionally removed direct identity words
such as old, bald, white beard, and prayer beads.

Result:

- Checkpoint:
  `checkpoints/anima_siglip_ip_adapter_ref03_overfit1024_20260611.safetensors`
  (local ignored artifact)
- Evidence:
  `eval/siglip_runtime_quality_20260611_c008_ref03_overfit1024_identity/contact_sheet.jpg`
- Report:
  `eval/siglip_runtime_quality_20260611_c008_ref03_overfit1024_identity/report.md`
- Decision: `overfit_pass_generalization_required`

The overfit checkpoint visibly recovered the bald monk face, red beads, robe
color, speech bubble, and crop without prompt-side identity hints. This means
the native SigLIP route is not a proven dead end. The remaining problem is
generalization and scale stability: the current local color/self checkpoints do
not carry identity across held-out prompts, and high overfit weights can still
collapse.

Training code now caches repeated one-row target latents, text embeddings, and
SigLIP features so the next focused overfit/ablation runs do not waste most of
their time recomputing fixed inputs.

## 2026-06-11 identity8 reference sweep

The cache was expanded to precompute repeated target latents, text embeddings,
and SigLIP features for up to 16 rows. This made a small 8-reference
self-reconstruction run practical:

- Manifest:
  `training/manifests/local_color_self_identity8_20260611.jsonl`
- Checkpoint:
  `checkpoints/anima_siglip_ip_adapter_identity8_1024_20260611.safetensors`
  (local ignored artifact)
- Report:
  `eval/siglip_runtime_quality_20260611_c009_identity8_reference_sweep/report.md`
- Contact sheet:
  `eval/siglip_runtime_quality_20260611_c009_identity8_reference_sweep/contact_sheet.jpg`
- Decision: `reference_influence_seen_identity_generalization_incomplete`

The checkpoint changes generated images differently for different references,
including held-out references, so reference influence is no longer limited to
the single-image overfit case. It still does not faithfully recover held-out
identity. The next useful scale-up is a larger balanced self-reconstruction set
with a held-out validation sheet, not another adjacent-panel-only run.

## 2026-06-11 identity128 and fp32 continuation

A 128-reference color self-reconstruction manifest was built from the local
`image_dataset_color_panel_style_v5_best` set:

- Manifest:
  `training/manifests/local_color_self_identity128_20260611.jsonl`
- Selection:
  `eval/identity128_reference_selection_20260611.json`
- bf16 continuation checkpoints:
  `checkpoints/anima_siglip_ip_adapter_identity128_1024_20260611.safetensors`
  and
  `checkpoints/anima_siglip_ip_adapter_identity128_2048_20260611.safetensors`
  (local ignored artifacts)
- bf16 evidence:
  `eval/siglip_runtime_quality_20260611_c010_identity128_reference_sweep/report.md`,
  `eval/siglip_runtime_quality_20260611_c011_identity128_neutral_prompt/report.md`,
  and
  `eval/siglip_runtime_quality_20260611_c012_identity128_2048_neutral_prompt/report.md`

The bf16 identity128 continuations lowered finite losses and changed pixels, but
`ip_scales` stayed effectively unchanged from the earlier identity8 checkpoint.
That made the poor visual result ambiguous: the route might have been weak, or
the trainable adapter might have been numerically stalled.

The training path was then changed to keep the trainable adapter in fp32 during
bounded CUDA smoke training. A fp32 1024-step continuation from the identity128
2048-step checkpoint produced:

- Checkpoint:
  `checkpoints/anima_siglip_ip_adapter_identity128_fp32_3072_20260611.safetensors`
  (local ignored artifact)
- Evidence:
  `eval/siglip_runtime_quality_20260611_c013_identity128_fp32_neutral_prompt/report.md`
- Pair sheet:
  `eval/siglip_runtime_quality_20260611_c013_identity128_fp32_neutral_prompt/reference_output_pairs.jpg`
- Decision: `fp32_training_moves_weights_but_quality_still_fail`

Runtime inspection showed the fp32 run moved 253 of 255 adapter tensors, with
relative L2 movement `0.01685` from the previous checkpoint versus `0.00440` for
the preceding bf16 continuation. This confirms there was a real training
precision/stability problem. However, the c013 sheet still collapses most
references into similar two-character court/interior scenes and does not recover
train or held-out reference identity/layout reliably.

Current conclusion: the native SigLIP route is not impossible in principle,
because the one-image overfit passed and fp32 training can move the adapter. But
the current frozen SigLIP2 encoder plus this small adapter/objective is not a
ready path to high-quality generalized Anima reference control. Further work
should shift to a stronger anime/VL image encoder or a pair objective that trains
the image encoder/adapter together, rather than only extending short local
SigLIP2 adapter tuning.

## 2026-06-11 reference-swap contrastive objective

The next ablation added a bounded reference-swap objective. For the same
target/noise/prompt, the model is trained so prediction with the correct
reference is closer to the target than prediction with a deterministic wrong
reference:

```text
loss = correct_reference_mse
     + contrastive_weight * relu(correct_mse - wrong_mse + margin)
```

Implementation:

- `training/siglip_reference_loss.py`
- `training/siglip_prepared_cache.py`
- `training/siglip_contrastive_smoke.py`
- `training/siglip_contrastive_cli.py`
- `tests/test_siglip_reference_loss.py`

Smoke evidence:

- `checkpoints/anima_siglip_ip_adapter_identity128_contrastive_0064_20260611.safetensors`
  (local ignored artifact)
  - `steps=64`, `rows_loaded=16`, `mean_contrastive_loss=0.04606`,
    checkpoint loadable.
- `checkpoints/anima_siglip_ip_adapter_identity128_contrastive_0512_20260611.safetensors`
  (local ignored artifact)
  - `steps=512`, `rows_loaded=32`, `mean_contrastive_loss=0.09111`,
    checkpoint loadable.
  - Tensor movement from the fp32 baseline: 253/255 tensors changed,
    relative L2 `0.00723`, max scale movement `0.00056`.

Visual evidence:

- `eval/siglip_runtime_quality_20260611_c014_identity128_contrastive_neutral_prompt/report.md`
- `eval/siglip_runtime_quality_20260611_c014_identity128_contrastive_neutral_prompt/reference_output_pairs.jpg`
- Decision: `contrastive_improves_reference_distance_but_quality_still_fail`

Compared with c013, c014 lowers mean pixel distance to several references and
increases reference-dependent variation. It also avoids some of the prior
two-character conversation collapse. However, it still does not faithfully
recover train or held-out identity/layout/character count. This suggests the
objective was a useful correction, but the frozen SigLIP2 adapter path still
needs either a stronger anime/VL encoder or a trainable image-encoder stage to
reach the requested high-quality reference-control target.

## 2026-06-11 contrastive weight sweep

The c014 contrastive checkpoint was evaluated through the isolated ComfyUI API at
weights `0.6`, `1.0`, `1.4`, and `1.8` across the same three train and three
held-out references:

- Evidence:
  `eval/siglip_runtime_quality_20260611_c015_contrastive_weight_sweep/report.md`
- Contact sheet:
  `eval/siglip_runtime_quality_20260611_c015_contrastive_weight_sweep/contact_sheet.jpg`
- Decision: `weight_sweep_no_usable_quality`

The sweep confirms that the checkpoint has a tunable reference-influence
channel. Higher weights can lower simple pixel distance to the reference,
especially for sky/crowd/color palette cases. The visual result still fails the
target bar: identity, layout, panel composition, and character count are not
stable, and higher weights often introduce generic group scenes or excessive
speech bubbles.

Current conclusion after c015: frozen SigLIP2 adapter tuning has learned a weak
style/composition pressure, but not the high-fidelity reference-control channel
the user wants. More inference-scale tuning on this checkpoint is not a credible
route to production quality. The next implementation path needs a trainable
image encoder/calibration stage or a stronger anime/VL encoder while preserving
the native ComfyUI patch surface.

## 2026-06-11 feature calibration path

An identity-initialized trainable feature-calibration module was added in front
of the native SigLIP adapter path. Calibrated checkpoints keep the existing
ComfyUI node surface and add `feature_calibrator.*` tensors to the SigLIP
checkpoint state; old uncalibrated checkpoints still load through the same
loader.

The first calibrated seed wrapped
`anima_siglip_ip_adapter_identity128_contrastive_0512_20260611.safetensors` with
a 256-wide feature calibrator. A 64-step contrastive continuation produced:

- `checkpoints/anima_siglip_ip_adapter_identity128_calibrated_contrastive_0064_20260611.safetensors`
- load type: `CalibratedIPAdapterSigLIP`
- changed tensors: 261
- changed calibration tensors: 8
- relative L2 from calibrated seed: `0.00232`
- API evidence:
  `eval/siglip_runtime_quality_20260611_c016_calibrated_contrastive_neutral_prompt/report.md`
- Decision: `calibration_path_executes_and_improves_some_rows_but_not_quality_pass`

A longer 512-step continuation from the 64-step checkpoint produced:

- `checkpoints/anima_siglip_ip_adapter_identity128_calibrated_contrastive_0576_20260611.safetensors`
- load type: `CalibratedIPAdapterSigLIP`
- changed tensors from c016: 261
- changed calibration tensors from c016: 8
- relative L2 from c016: `0.00671`
- API evidence:
  `eval/siglip_runtime_quality_20260611_c017_calibrated_contrastive0576_neutral_prompt/report.md`
- Decision: `longer_calibrated_contrastive_training_overfits_scene_average`

Interpretation: feature calibration is a real trainable path and gives a better
early signal than frozen adapter-only tuning. It still does not pass the
high-quality reference-control target. More steps on the same contrastive
objective can overfit toward generic group/court scenes, so the next attempt
should change the objective or image-encoder signal instead of only increasing
step count.

## 2026-06-11 PE-teacher distillation attempt

The next bounded attempt used the stronger PE adapter as a frozen teacher for
the native SigLIP path. The training objective kept the previous
base/contrastive denoising losses and added an MSE term that asks the SigLIP
adapter prediction to match the PE adapter prediction for the same
target/noise/timestep/prompt.

Code added:

- `training/pe_teacher_features.py`
- `training/pe_teacher_distillation.py`
- `training/siglip_teacher_smoke.py`
- `training/siglip_teacher_cli.py`
- `tests/test_pe_teacher_distillation.py`

Smoke and candidate evidence:

- 2-step smoke checkpoint:
  `checkpoints/anima_siglip_ip_adapter_teacher_smoke_0002_20260611.safetensors`
  (local ignored artifact), finite loss, SigLIP-loadable, PE-rejected.
- 64-step candidate:
  `checkpoints/anima_siglip_ip_adapter_identity128_pe_teacher_0064_20260611.safetensors`
  (local ignored artifact)
  - `steps=64`, `rows_loaded=16`, `mean_loss=0.23133`,
    `mean_base_loss=0.20576`, `mean_contrastive_loss=0.04206`,
    `mean_teacher_loss=0.03011`
  - `trainable_parameters=336650396`
  - checkpoint loadable through `AnimaSigLIPIPAdapterLoader`, PE checkpoint
    rejected by the SigLIP loader.

ComfyUI API evidence:

- `eval/siglip_runtime_quality_20260611_c018_pe_teacher_distill/report.md`
- `eval/siglip_runtime_quality_20260611_c018_pe_teacher_distill/contact_sheet.jpg`
- `eval/siglip_runtime_quality_20260611_c018_pe_teacher_weight_sweep/contact_sheet.jpg`

Decision: `pe_teacher_distillation_changes_outputs_but_not_quality_pass`

Interpretation: the PE-teacher path proves that the native SigLIP adapter can
receive a stronger supervised signal and that the runtime patch remains
functional. It does not solve the actual quality target. The c018 contact sheet
shows reference-dependent outputs, but the generated images do not reliably
inherit reference color, character identity, panel layout, or composition.
The weight sweep confirms this is not an inference-strength issue: lower
weights collapse toward no-IP and higher weights distort scenes rather than
recovering the reference.

Current conclusion after c018: this exact frozen SigLIP2 plus adapter/calibrator
route is a weak research branch, not a production reference-control checkpoint.
The next credible path is a stronger anime/VL image encoder or a trainable
image-encoder adaptation stage, most likely Qwen-VL style features or an
anime-domain SigLIP/PE-like encoder, before another broad training run.

## 2026-06-12 c035 single-character suite result

The c034 auto-attribute prompt run was expanded into a 32-case single-character
suite:

```text
eval/siglip_reference_suite_v1_20260612/reference_suite_v1.jsonl
eval/siglip_runtime_quality_20260612_c035_suite_v1/
```

Runtime path:

- ComfyUI API: `http://127.0.0.1:8116`
- node family: `AnimaSigLIPIPAdapterLoader`,
  `AnimaSigLIPEncodeImage`, `AnimaSigLIPIPAdapterApply`
- variants: no-IP, `siglip_kv_init_w14`, `siglip_ref_retrieval_w14`
- weight: `1.4`

Metric result:

| variant | mean uplift | improved rate | decision |
| --- | ---: | ---: | --- |
| `siglip_kv_init_w14` | +0.0292 | 0.65625 | fail |
| `siglip_ref_retrieval_w14` | +0.0577 | 0.65625 | fail |

Visual audit:

- palette/costume/expression/framing acceptable: `31/32`
- identity/distinctive trait acceptable: `16/32`
- non-human/special trait acceptable: `0/1`
- blank outputs: `0`
- decision: `not_ready`

Interpretation: `siglip_ref_retrieval_w14` remains the best current SigLIP
variant, and the generated images are often attractive. It is still not a
trustworthy high-quality reference-control checkpoint. The failure mode is a
repeated collapse toward black long-haired wuxia characters, purple/night palace
lighting, red-eye villain traits, and generic official/elder templates.

The selected next direction is `train_stronger_encoder`, documented in:

```text
docs/ipadapter_next_direction_decision_ko.md
```

## 2026-06-12 c036 QwenVL pooled metric probe

Before starting another longer training run, c035 was rescored with
`Qwen/Qwen3-VL-Embedding-2B` image embeddings to test whether QwenVL pooled
similarity is a better quality gate than PE pooled-cosine.

Evidence:

- `tools/score_auto_caption_qwenvl_metrics.py`
- `tests/test_score_auto_caption_qwenvl_metrics.py`
- `eval/qwenvl_metric_probe_20260612_c036_c035/qwenvl_similarity_metrics.json`
- `eval/qwenvl_metric_probe_20260612_c036_c035/report.md`

Metric result:

| variant | mean uplift | improved rate |
| --- | ---: | ---: |
| `siglip_kv_init_w14` | +0.0422 | 0.84375 |
| `siglip_ref_retrieval_w14` | +0.0446 | 0.90625 |

Decision: `qwenvl_pooled_metric_auxiliary_only`

QwenVL pooled similarity is useful as an auxiliary broad-similarity metric, but
it is not aligned enough with the c035 identity/distinctive-trait visual audit.
Identity-fail rows scored higher mean uplift than identity-pass rows. The next
stronger-encoder loop must first build identity-positive/negative pairs and
measure feature separation before using QwenVL pooled embeddings as a primary
training signal or pass/fail gate.

## 2026-06-12 c037 identity feature probe

The c036 prerequisite was executed with a 128-pair weak identity proxy manifest
from the local color dataset. Same `SG-*` folder pairs were treated as positive
proxy pairs, and next `SG-*` folder pairs were treated as negative proxy pairs.
This is not a verified same-character benchmark, but it is a useful early gate:
if pooled features fail here, they should not become the primary identity loss.

Evidence:

- `tools/build_identity_pair_probe_manifest.py`
- `tools/image_feature_embedders.py`
- `tools/score_identity_pair_probe.py`
- `tests/test_identity_feature_probe.py`
- `eval/identity_feature_probe_20260612_c037/report.md`

Result:

| encoder | margin | pairwise AUC | decision |
| --- | ---: | ---: | --- |
| `Qwen/Qwen3-VL-Embedding-2B` | +0.0326 | 0.5913 | fail |
| `google/siglip2-base-patch16-512` | +0.0132 | 0.5759 | fail |
| `pe` | +0.0156 | 0.5806 | fail |

Decision: `pooled_identity_feature_not_ready`

Do not launch a long adapter training run that relies mainly on QwenVL,
SigLIP2, or PE pooled-image cosine for identity supervision. The next branch
should mine or label true same-character positives, add hard negatives within
the same style/scene bucket, and test token-level or learned metric features
before another adapter run.

## 2026-06-12 c038 strict panel sanity probe

c037 was followed by a stricter feature-pipeline sanity control. Positive pairs
were v4/v5 duplicate crops with the same panel key; negatives were different
panel keys from the same `SG-*` folder. This does not prove character identity
generalization, but it checks whether the encoders can separate true near
duplicates from same-style hard negatives.

Evidence:

- `tools/build_strict_panel_pair_probe_manifest.py`
- `tools/score_siglip_token_pair_probe.py`
- `tools/token_pair_probe_metrics.py`
- `tests/test_strict_identity_probe.py`
- `eval/strict_identity_feature_probe_20260612_c038/report.md`

Result:

| encoder/metric | margin | pairwise AUC | decision |
| --- | ---: | ---: | --- |
| `Qwen/Qwen3-VL-Embedding-2B` pooled | +0.2061 | 1.0000 | pass |
| `google/siglip2-base-patch16-512` pooled | +0.1058 | 1.0000 | pass |
| `pe` pooled | +0.1404 | 0.9998 | pass |
| SigLIP2 `mean_max_token` | +0.3170 | 1.0000 | pass |
| SigLIP2 layer `-6` pooled | +0.4739 | 0.9998 | pass |

Decision: `strict_duplicate_feature_sanity_pass_identity_unsolved`

The feature pipeline is not broken: pooled and token features can detect
near-duplicate panel crops. The identity problem remains unsolved because
duplicate detection is easier than same-character reference control across
pose, expression, costume, and scene. The next branch should exclude duplicate
crops, build true same-character positives with same-scene hard negatives, and
re-test SigLIP layer `-6` pooled plus `mean_max_token` on that stricter manifest.

## 2026-06-12 c039 true identity candidate review

The next loop mined same-page, non-duplicate panel candidates and rendered a
small review sheet.

Evidence:

- `tools/build_true_identity_candidate_review.py`
- `tests/test_true_identity_candidate_review.py`
- `eval/true_identity_candidate_review_20260612_c039/candidate_pairs.jsonl`
- `eval/true_identity_candidate_review_20260612_c039/candidate_sheet.jpg`
- `eval/true_identity_candidate_review_20260612_c039/report.md`

Decision: `same_page_candidates_need_character_filtering`

Same-page candidates are useful for review-sheet generation, but visual review
showed many different-character, background, prop, and multi-character crops.
They should not be promoted directly into a true same-character training
manifest. The next branch should filter for character-centered crops before
same-character labeling or metric-head training.

## 2026-06-12 c040 character-filtered candidates

Qwen3-VL image-text retrieval was used as a first-pass character-centered crop
filter on the c039 candidates.

Evidence:

- `tools/filter_character_candidate_pairs.py`
- `tests/test_character_candidate_filter.py`
- `eval/character_filtered_identity_candidates_20260612_c040/scored_candidate_pairs.jsonl`
- `eval/character_filtered_identity_candidates_20260612_c040/kept_candidate_pairs.jsonl`
- `eval/character_filtered_identity_candidates_20260612_c040/kept_candidate_sheet.jpg`
- `eval/character_filtered_identity_candidates_20260612_c040/report.md`

Decision: `character_filter_reduces_noise_not_identity_labels`

At threshold `0.15`, 14 of 24 candidates remained. The filter removes some
background/object noise, but visual review still shows different-character and
ambiguous torso-only crops. Keep this as an auxiliary candidate filter, not an
automatic true-identity labeler. The next branch needs an explicit reviewed
manifest with `same_character`, `different_character`, and `unclear` labels.

## 2026-06-12 c041 reviewed identity candidates

The c040 kept candidates were converted into a manually reviewed identity seed
manifest.

Evidence:

- `tools/build_reviewed_identity_manifest.py`
- `tests/test_reviewed_identity_manifest.py`
- `eval/reviewed_identity_candidates_20260612_c041/manual_visual_labels.jsonl`
- `eval/reviewed_identity_candidates_20260612_c041/reviewed_candidate_pairs.jsonl`
- `eval/reviewed_identity_candidates_20260612_c041/usable_positive_pairs.jsonl`
- `eval/reviewed_identity_candidates_20260612_c041/different_character_pairs.jsonl`
- `eval/reviewed_identity_candidates_20260612_c041/reviewed_candidate_sheet.jpg`
- `eval/reviewed_identity_candidates_20260612_c041/report.md`

Decision: `reviewed_seed_too_small_for_training_gate`

The reviewed set contains 14 rows: 6 same-character, 3 different-character, 5
unclear, and only 4 positive-usable pairs. This is useful as a tiny feature
sanity probe seed, but it is not enough to launch adapter or metric-head
training. The next branch should score this seed with SigLIP layer `-6`,
SigLIP `mean_max_token`, QwenVL pooled, and PE pooled, while expanding mining
to collect many more face/upper-body positives.

## 2026-06-12 c042 reviewed seed feature probe

The c041 reviewed seed was converted into a positive/negative pair probe and
scored with the current feature candidates.

Evidence:

- `tools/build_reviewed_pair_probe_manifest.py`
- `tests/test_reviewed_pair_probe_manifest.py`
- `eval/reviewed_seed_feature_probe_20260612_c042/pair_probe_manifest.jsonl`
- `eval/reviewed_seed_feature_probe_20260612_c042/siglip_pooled_report.md`
- `eval/reviewed_seed_feature_probe_20260612_c042/siglip_layer_m6_token_report.md`
- `eval/reviewed_seed_feature_probe_20260612_c042/qwenvl_pooled_report.md`
- `eval/reviewed_seed_feature_probe_20260612_c042/pe_pooled_report.md`
- `eval/reviewed_seed_feature_probe_20260612_c042/report.md`

Decision: `reviewed_seed_feature_gate_not_passed`

No raw feature passed the current identity gate of margin `>= 0.05` and AUC
`>= 0.70`. QwenVL pooled was the best pooled feature but only reached margin
`0.024015` and AUC `0.666667`. SigLIP layer `-6` `mean_max_token` is the most
interesting next candidate with margin `0.043225` and AUC `0.916667`, but the
seed has only 4 positive and 3 negative rows, so this is not enough to train or
ship. Expand reviewed face/upper-body positives before the next training run.

## 2026-06-11 Qwen3-VL embedding probe

The next encoder candidate was checked before writing another adapter training
loop. `Qwen/Qwen3-VL-Embedding-2B` exists on Hugging Face, is a
sentence-transformers multimodal embedding model, and locally produced
embeddings for six identity128 color-panel reference images.

Evidence:

- `eval/qwen3vl_embedding_probe_20260611/report.md`
- `eval/qwen3vl_embedding_probe_20260611/summary.json`

Probe result:

- embedding shape: `[6, 2048]`
- off-diagonal cosine mean: `0.563116`
- off-diagonal cosine min/max: `0.424949` / `0.737467`

Interpretation: the Qwen3-VL embedding model loads locally and separates the
test references better than a collapsed generic style vector would. This makes
it a credible next branch for adapter training. The previous QwenVL plan should
not assume `1024` dimensions: the current public 2B embedding checkpoint emits
`2048` dimensions by default, with optional custom/MRL dimensions documented by
the model card.

## 2026-06-11 QwenVL adapter family scaffold

The first QwenVL implementation step adds a separate checkpoint family instead
of overloading the SigLIP loader:

- `qwenvl_model.py`
- `qwenvl_checkpoint.py`
- `tests/test_qwenvl_adapter.py`

The new `IPAdapterQwenVL` reuses the proven Anima-side `TimeResampler` and
`IPCrossAttn` modules, but consumes normalized Qwen3-VL embeddings directly.
The default input contract is `[B, 2048]` or `[B, T, 2048]`, matching the public
`Qwen/Qwen3-VL-Embedding-2B` default output. Checkpoints include a persistent
`qwenvl_family` marker so checkpoint families stay fail-loud:

- QwenVL detector rejects PE-Core checkpoints.
- QwenVL detector rejects SigLIP checkpoints without the marker.
- SigLIP detector rejects QwenVL-marked checkpoints.

Current status: this is a model/checkpoint scaffold and synthetic shape smoke,
not a quality claim. The next useful step is adding a QwenVL image encoder node
or cached embedding loader, then running the same bounded Anima denoising smoke
that the SigLIP branch uses.

## 2026-06-11 QwenVL native ComfyUI node surface

The QwenVL branch now has a native ComfyUI node surface:

- `AnimaQwenVLIPAdapterLoader`
- `AnimaQwenVLEncodeImage`
- `AnimaQwenVLIPAdapterApply`

`AnimaQwenVLEncodeImage` uses the public
`Qwen/Qwen3-VL-Embedding-2B` sentence-transformers path and returns normalized
`2048`-dim image embeddings. `AnimaQwenVLIPAdapterApply` reuses the same Anima
cross-attention patch surface that made the SigLIP path visibly affect pixels.

Evidence:

- `eval/qwenvl_native_workflow_eval_20260611/report.md`
- `eval/qwenvl_native_workflow_eval_20260611/object_info_AnimaQwenVLIPAdapterLoader.json`
- `eval/qwenvl_native_workflow_eval_20260611/object_info_AnimaQwenVLEncodeImage.json`
- `eval/qwenvl_native_workflow_eval_20260611/object_info_AnimaQwenVLIPAdapterApply.json`

Current status: node registration and synthetic runtime patch tests pass. The
branch still needs a trained QwenVL checkpoint before any image quality claim.

## 2026-06-11 QwenVL bounded training and ComfyUI quality check

A native QwenVL training smoke was added for the same Anima denoising objective
used by the SigLIP branch:

- `training/qwenvl_real_smoke.py`
- `training/qwenvl_smoke_checkpoint.py`
- `training/qwenvl_smoke_cli.py`
- `training/qwenvl_prepared_cache.py`
- `training/qwenvl_contrastive_smoke.py`
- `training/qwenvl_contrastive_cli.py`
- `tests/test_qwenvl_smoke.py`
- `tests/test_qwenvl_contrastive.py`

Smoke evidence:

- `eval/qwenvl_runtime_quality_20260611_c001_smoke/report.md`
- `checkpoints/anima_qwenvl_ip_adapter_smoke_0002_20260611.safetensors`
  (local ignored artifact)
- `checkpoints/anima_qwenvl_ip_adapter_identity128_0064_20260611.safetensors`
  (local ignored artifact)
- `checkpoints/anima_qwenvl_ip_adapter_identity128_contrastive_0064_20260611.safetensors`
  (local ignored artifact)

The 64-step identity128 candidate trained with finite loss and saved a
QwenVL-marked checkpoint:

- rows loaded: `16`
- first/final loss: `0.22585` / `0.10778`
- checkpoint loadable through the QwenVL loader
- PE checkpoint rejected by the QwenVL loader

ComfyUI API quality evidence:

- `eval/qwenvl_runtime_quality_20260611_c001_identity128/report.md`
- `eval/qwenvl_runtime_quality_20260611_c002_identity128_weight_sweep/report.md`
- `eval/qwenvl_runtime_quality_20260611_c002_identity128_weight_sweep/contact_sheet.jpg`
- `eval/qwenvl_runtime_quality_20260611_c003_contrastive_weight_sweep/report.md`
- `eval/qwenvl_runtime_quality_20260611_c003_contrastive_weight_sweep/contact_sheet.jpg`

Decision: `qwenvl_adapter_only_changes_outputs_but_not_quality_pass`

Interpretation: QwenVL is a stronger and cleaner embedding candidate than the
frozen SigLIP2 branch, and the native ComfyUI path works end to end. However,
the current adapter-only QwenVL runs still fail the requested reference-control
bar. The adapter changes images relative to no-IP, but both the plain denoising
checkpoint and the reference-swap contrastive checkpoint collapse most
references toward a similar yellow-robed interior/crowd scene. They do not
preserve reference-specific identity, layout, character count, or color palette.

Current conclusion after QwenVL c003: adapter-only tuning on frozen generic
image embeddings is not enough for production-quality Anima reference control.
The next credible training stage needs one of:

- trainable image-encoder or feature-calibrator adaptation with a stronger
  reference discrimination objective;
- a PE-quality teacher/control target that supervises more than denoising MSE;
- an anime-domain image encoder trained or adapted specifically for panel
  identity, color palette, and layout.

## 2026-06-11 QwenVL feature calibration check

A QwenVL feature-calibration branch was added after the c003 collapse. It keeps
the same QwenVL checkpoint family and ComfyUI node surface, but adds an
identity-initialized trainable `feature_calibrator.*` module before the
TimeResampler:

- `qwenvl_feature_calibration.py`
- `tests/test_qwenvl_feature_calibration.py`

The calibrated continuation command wrapped
`anima_qwenvl_ip_adapter_identity128_contrastive_0064_20260611.safetensors`
with a 128-wide feature calibrator and ran 64 contrastive steps on the same
local color self-reconstruction manifest. The run completed with finite loss:

- checkpoint:
  `checkpoints/anima_qwenvl_ip_adapter_identity128_calibrated_contrastive_0064_20260611.safetensors`
  (local ignored artifact)
- rows loaded: `16`
- first/final loss: `0.25603` / `0.19008`
- mean loss: `0.21313`
- mean contrastive loss: `0.05004`
- checkpoint loadable through the QwenVL loader
- PE checkpoint rejected by the QwenVL loader

ComfyUI API quality evidence:

- `eval/qwenvl_runtime_quality_20260611_c004_calibrated_contrastive_weight_sweep/report.md`
- `eval/qwenvl_runtime_quality_20260611_c004_calibrated_contrastive_weight_sweep/contact_sheet.jpg`

Decision: `qwen_feature_calibration_changes_outputs_but_reference_collapse_remains`

Interpretation: the feature-calibration path trains and loads correctly, but it
does not solve the visual target. The c004 sheet still collapses most
references toward the same yellow-robed street/interior conversation scene. It
does not preserve reference-specific color, identity, character count, panel
layout, or held-out composition. This makes a longer run on the same frozen
QwenVL embedding plus adapter/calibrator recipe a poor next step unless the
objective or encoder supervision changes.

## 2026-06-11 PE-style query patch correction

A runtime/training geometry mismatch was found by comparing the native SigLIP
patch with the PE-Core patch. The PE path computes the normal Anima
cross-attention query through `compute_qkv` and attends that query to cached IP
K/V. The native SigLIP/QwenVL patch was instead adding a separate
`IPCrossAttn(x, image_tokens)` stream based on pre-attention hidden states.

The native patch now prefers the PE-style query path when the attention module
exposes `compute_qkv`, while keeping the old path as a fallback for tests or
older attention objects:

- `native_ip_attention.py`
- `native_siglip_runtime.py`
- `training/siglip_smoke_patch.py`
- `tests/test_native_ip_attention.py`

ComfyUI API evidence:

- `eval/siglip_runtime_quality_20260611_c019_pe_query_patch_weight_sweep/report.md`
- `eval/siglip_runtime_quality_20260611_c019_pe_query_patch_weight_sweep/contact_sheet.jpg`
- `eval/siglip_runtime_quality_20260611_c020_pe_query_patch_trained_weight_sweep/report.md`
- `eval/siglip_runtime_quality_20260611_c020_pe_query_patch_trained_weight_sweep/contact_sheet.jpg`

Decision:

- c019: `query_patch_increases_reference_influence_but_not_quality_pass`
- c020: `short_query_patch_retrain_not_quality_pass`

Interpretation: the PE-style query patch is a real correction. It increases
reference-dependent variation compared with the earlier generic yellow-robed
collapse. It still does not pass the high-quality gate: train and held-out rows
do not reliably recover reference palette, layout, or identity, and higher
weights can distort figures. The corrected patch surface should be kept, but a
64-step continuation is not enough to produce a production-quality SigLIP
reference-control checkpoint.

## 2026-06-11 Single-character diagnostic

The page/contact-sheet references were simplified to four manually selected
single-character color crops from the color-panel dataset. The evaluation used a
fixed solo portrait prompt, one no-IP baseline, and SigLIP weights `0.7`, `1.0`,
and `1.4` through the isolated ComfyUI API on port `8116`.

Evidence:

- `eval/siglip_runtime_quality_20260611_c021_single_character_diagnostic/report.md`
- `eval/siglip_runtime_quality_20260611_c021_single_character_diagnostic/contact_sheet.jpg`
- `eval/siglip_runtime_quality_20260611_c021_single_character_diagnostic/summary.json`
- `eval/siglip_runtime_quality_20260611_c021_single_character_diagnostic/candidate_sheet.jpg`

Decision: `single_character_diagnostic_not_quality_pass`

Interpretation: simplifying the reference to single-character crops does not
rescue the current SigLIP checkpoint. The adapter changes no-IP output and
produces reference-dependent variation, but it still fails to preserve core
attributes such as beard/age, blue robe palette, gold hair/fire palette, and
stable identity. The failure is therefore not only a multi-panel page-layout
problem. The next useful work should change the learning signal or encoder
adaptation strategy rather than spending a long run on the same frozen-SigLIP
adapter-only recipe.

## 2026-06-11 Single-character micro-training

A four-reference single-character manifest was created from the c021 selected
references:

```text
training/manifests/local_color_single_character_identity4_20260611.jsonl
training/manifests/local_color_single_character_identity4_20260611.summary.json
```

The manifest contains these labels:

- `bearded_tan_robe`
- `blue_robed_elder`
- `black_robe_closeup`
- `golden_angry_face`

The run continued from the PE-style query-patch checkpoint for 256 steps:

```text
checkpoints/anima_siglip_ip_adapter_single_character_identity4_pe_query_patch_0256_20260611.safetensors
```

Observed training summary:

- rows loaded: `4`
- first/final loss: `0.22382895648479462` / `0.13005425035953522`
- mean loss: `0.2278406125260517`
- mean base loss: `0.1967822091828566`
- mean contrastive loss: `0.03507533890660852`
- mean teacher loss: `0.02704146476389724`
- finite loss: `true`

ComfyUI API quality evidence:

- `eval/siglip_runtime_quality_20260611_c022_single_character_identity4_trainfit/report.md`
- `eval/siglip_runtime_quality_20260611_c022_single_character_identity4_trainfit/contact_sheet.jpg`
- `eval/siglip_runtime_quality_20260611_c023_single_character_identity4_heldout/report.md`
- `eval/siglip_runtime_quality_20260611_c023_single_character_identity4_heldout/contact_sheet.jpg`

Decision:

- c022: `single_character_identity4_trainfit_improves_palette_but_not_identity_pass`
- c023: `single_character_identity4_heldout_partial_transfer_not_quality_pass`

Interpretation: the user was right that single-character references are the
cleaner comparison target. The train-fit sheet shows an actual learning signal:
blue robe palette, black/red robe palette, and side/profile direction improve
after the 256-step micro-train. The held-out sheet shows partial transfer on
black/tan references, but also overfit and missed identity details. Beard,
age, glasses, gold/fire palette, and unusual facial intensity are still not
reliable. This keeps the goal active: single-character-first training is useful
as the next diagnostic path, but the current checkpoint is not a quality pass.

## 2026-06-11 Clean32 single-character color run

The next run scaled the single-character diagnostic from four hand-picked
references to a curated color-panel subset:

```text
training/manifests/local_color_single_character_clean32_20260611.jsonl
training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl
training/manifests/local_color_single_character_clean32_20260611.summary.json
```

Selection evidence:

- `eval/siglip_runtime_quality_20260611_c024_single_character_clean32_selection/candidate_sheet.jpg`

The clean32 checkpoint continued from the PE-style query-patch checkpoint for
512 steps:

```text
checkpoints/anima_siglip_ip_adapter_single_character_clean32_pe_query_patch_0512_20260611.safetensors
```

Observed training summary:

- rows loaded: `32`
- first/final loss: `0.4276636838912964` / `0.14298595488071442`
- mean loss: `0.23064063434139825`
- mean base loss: `0.18559964589803712`
- mean contrastive loss: `0.043380077069741674`
- mean teacher loss: `0.025011859186633956`
- finite loss: `true`
- trainable parameters: `336650396`

ComfyUI API quality evidence:

- `eval/siglip_runtime_quality_20260611_c025_single_character_clean32_runtime/report.md`
- `eval/siglip_runtime_quality_20260611_c025_single_character_clean32_runtime/contact_sheet.jpg`
- `eval/siglip_runtime_quality_20260611_c025_single_character_clean32_runtime/summary.json`

Decision: `single_character_clean32_runtime_not_quality_pass`

Interpretation: single-character evaluation is the correct first gate. It makes
reference influence visible and avoids confusing page-layout difficulty with
adapter failure. The clean32 checkpoint changes outputs beyond no-IP and can
push broad dark/red/blue wuxia palette, stern faces, and robe styling. However,
it still does not preserve reference-specific identity, props, beard/age,
glasses/fan, non-human face, or exact palette reliably. Stronger `1.4` weight
usually amplifies a learned template instead of improving fidelity. The next
step should keep this single-character gate, but change the objective or encoder
adaptation path rather than launching a long run of the same frozen-SigLIP
adapter-only recipe.

## 2026-06-11 Clean32 token-separation continuation

A token-level reference separation loss was added and tested as the first
objective change after the clean32 run. The loss compares correct-reference and
wrong-reference image tokens and penalizes overly high cosine similarity:

```text
training/siglip_reference_loss.py
training/siglip_teacher_smoke.py
training/siglip_teacher_cli.py
```

The run continued from the clean32 checkpoint for 256 steps:

```text
checkpoints/anima_siglip_ip_adapter_single_character_clean32_token_sep_0256_20260611.safetensors
```

Observed training summary:

- rows loaded: `32`
- first/final loss: `0.3259349763393402` / `0.19840562343597412`
- mean loss: `0.29138473694911227`
- mean base loss: `0.1944809732667636`
- mean contrastive loss: `0.04216717180679552`
- mean teacher loss: `0.0204733534837942`
- mean token loss: `0.15726202292717062`
- finite loss: `true`
- trainable parameters: `336650396`

ComfyUI API quality evidence:

- `eval/siglip_runtime_quality_20260611_c026_single_character_token_sep_runtime/report.md`
- `eval/siglip_runtime_quality_20260611_c026_single_character_token_sep_runtime/contact_sheet.jpg`
- `eval/siglip_runtime_quality_20260611_c026_single_character_token_sep_runtime/summary.json`

Decision: `single_character_token_sep_not_quality_pass`

Interpretation: token separation works mechanically and increases visual
variation relative to clean32, but it does not improve reference fidelity. The
token variants still miss beard/age, glasses/fan, demon face, cropped screaming
face, exact palette, and stable identity. In several cases the token run pushes
the output into a different learned template instead of preserving the
reference. Do not launch a longer token-separation-only run unchanged. The next
branch needs a semantic reference anchor, such as anime/VL teacher features,
explicit identity/palette/prop attributes, paired reference-target supervision,
or a trainable image encoder/calibrator optimized for reference retrieval.

## 2026-06-11 PE-token and PE-space semantic-anchor runs

Two stronger semantic-anchor branches were tested after token separation.

The first branch added direct PE K/V descriptor alignment:

```text
training/pe_teacher_token_alignment.py
training/siglip_teacher_summary.py
```

The PE-token-anchor checkpoint continued from clean32 for 256 steps:

```text
checkpoints/anima_siglip_ip_adapter_single_character_clean32_pe_token_anchor_0256_20260611.safetensors
```

Observed c027 training summary:

- rows loaded: `32`
- first/final loss: `0.3050529956817627` / `0.21933485567569733`
- mean loss: `0.26382563475635834`
- mean base loss: `0.1882707678596489`
- mean contrastive loss: `0.043121844122651964`
- mean teacher loss: `0.022011573988493183`
- mean PE-token loss: `0.13854585964872967`
- finite loss: `true`

ComfyUI API quality evidence:

- `eval/siglip_runtime_quality_20260611_c027_single_character_pe_token_anchor_runtime/report.md`
- `eval/siglip_runtime_quality_20260611_c027_single_character_pe_token_anchor_runtime/contact_sheet.jpg`
- `eval/siglip_runtime_quality_20260611_c027_single_character_pe_token_anchor_runtime/summary.json`

Decision: `single_character_pe_token_anchor_not_quality_pass`

The second branch changed the native SigLIP architecture so `dit_dim=2048` and
`ip_hidden_dim=1024` can differ, then initialized the SigLIP adapter with the
PE adapter's trained K/V projections and gates:

```text
siglip_model.py
siglip_checkpoint.py
training/pe_space_siglip_adapter.py
```

The PE-space checkpoint trained for 512 steps:

```text
checkpoints/anima_siglip_ip_adapter_single_character_clean32_pe_space_init_0512_20260611.safetensors
```

Observed c028 training summary:

- rows loaded: `32`
- first/final loss: `0.5026331543922424` / `0.18201853334903717`
- mean loss: `0.22448686529241968`
- mean base loss: `0.19500588972005062`
- mean contrastive loss: `0.0499068612116389`
- mean teacher loss: `0.016648300783572267`
- mean PE-token loss: `0.01605273197731094`
- finite loss: `true`
- trainable parameters: `218159260`

ComfyUI API quality evidence:

- `eval/siglip_runtime_quality_20260611_c028_single_character_pe_space_init_runtime/report.md`
- `eval/siglip_runtime_quality_20260611_c028_single_character_pe_space_init_runtime/contact_sheet.jpg`
- `eval/siglip_runtime_quality_20260611_c028_single_character_pe_space_init_runtime/summary.json`

Decision: `single_character_pe_space_init_not_quality_pass`

Interpretation: c027 and c028 are meaningful engineering progress but not
quality passes. PE-token alignment improves stability; PE-space initialization
proves the native loader/runtime can use asymmetric PE-token-space checkpoints.
However, both still miss identity-specific details such as elder beard/baldness,
glasses/fan, cropped screaming expression, and green demon face. The c028
failure is especially informative: once PE K/V is reused, the model becomes
clean and sharp but collapses toward a narrow young black-haired wuxia male
template. This points away from more adapter-only tuning and toward the
encoder/resampler side. The next branch should train a stronger
image-feature calibrator or retrieval/ID/palette objective before broad
denoising, or use Qwen/PE teacher features to supervise identity/palette/prop
tokens directly.

## 2026-06-11 PE-space retrieval pilot

After c028, a PE-token retrieval branch was added:

```text
training/pe_token_retrieval.py
training/siglip_teacher_step.py
training/siglip_teacher_runtime.py
```

The branch continues from the PE-space checkpoint and adds a margin loss that
makes native SigLIP image tokens prefer the matching PE tokens over a
deterministic wrong-reference PE token set. This is a direct resampler-side
retrieval signal, not only another denoiser-output loss.

Checkpoint:

```text
checkpoints/anima_siglip_ip_adapter_single_character_clean32_pe_retrieval_0128_20260611.safetensors
```

Observed 128-step summary:

- rows loaded: `32`
- first/final loss: `0.22356687486171722` / `0.4331858456134796`
- mean loss: `0.3135300036519766`
- mean base loss: `0.19486997387139127`
- mean teacher loss: `0.01949366783082951`
- mean PE-token loss: `0.0016404704861088248`
- mean PE-retrieval loss: `0.20024060737341642`
- finite loss: `true`

ComfyUI API quality evidence:

- `eval/siglip_runtime_quality_20260611_c029_single_character_pe_retrieval_runtime/report.md`
- `eval/siglip_runtime_quality_20260611_c029_single_character_pe_retrieval_runtime/contact_sheet.jpg`
- `eval/siglip_runtime_quality_20260611_c029_single_character_pe_retrieval_runtime/summary.json`

Decision: `single_character_pe_retrieval_not_quality_pass`

Interpretation: PE-space K/V initialization plus pairwise PE-token retrieval is
still insufficient. The images remain clean, but elder identity, scholar props,
screaming crop, and green demon identity are not preserved. The failure points
more strongly at frozen SigLIP feature insufficiency for anime identity,
palette, and prop attributes. The next branch should train a stronger
image-feature calibrator/encoder or use Qwen/PE teacher features to produce
explicit identity/palette/prop tokens before denoising.

## 2026-06-11 QwenVL single-character retrieval pilot

After the PE retrieval pilot, a QwenVL token-retrieval branch was added:

```text
training/qwenvl_token_retrieval.py
training/qwenvl_step.py
```

The branch continues from the calibrated QwenVL identity128 checkpoint and adds
a margin loss that makes the adapter token mean prefer the matching QwenVL image
embedding over a deterministic wrong-reference embedding.

Checkpoint:

```text
checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors
```

Observed 128-step summary:

- rows loaded: `32`
- first/final loss: `0.2704607844352722` / `0.3180232048034668`
- mean loss: `0.32662612583953887`
- mean base loss: `0.21053104143356904`
- mean contrastive loss: `0.04986031912267208`
- mean QwenVL retrieval loss: `0.20726001169532537`
- finite loss: `true`

ComfyUI API quality evidence:

- `eval/qwenvl_runtime_quality_20260611_c030_single_character_retrieval/report.md`
- `eval/qwenvl_runtime_quality_20260611_c030_single_character_retrieval/contact_sheet.jpg`
- `eval/qwenvl_runtime_quality_20260611_c030_single_character_retrieval/summary.json`

Decision: `qwen_retrieval_single_character_not_quality_pass`

Interpretation: the QwenVL retrieval adapter changes images, but it still
collapses toward a generic black-haired wuxia male template. It misses old
bearded identity, glasses/fan/scholar props, screaming close-crop expression,
and green demon/non-human traits. This is a useful negative result: adding a
short token-retrieval loss to the existing adapter is not enough. The next
branch should train a stronger image-feature calibrator/encoder or explicit
identity/palette/prop tokens before another denoising-centered run.

## 2026-06-11 Attribute-prompt runtime gate

The c030 failure was re-tested with per-reference attribute prompts instead of
the generic solo portrait prompt. The attribute prompts name visible identity,
palette, prop, expression, and non-human traits from the same eight
single-character references.

QwenVL evidence:

- `eval/qwenvl_runtime_quality_20260611_c031_attribute_prompt_runtime/report.md`
- `eval/qwenvl_runtime_quality_20260611_c031_attribute_prompt_runtime/contact_sheet.jpg`
- `eval/qwenvl_runtime_quality_20260611_c031_attribute_prompt_runtime/pe_similarity_metrics.json`

SigLIP evidence:

- `eval/siglip_runtime_quality_20260611_c032_attribute_prompt_runtime/report.md`
- `eval/siglip_runtime_quality_20260611_c032_attribute_prompt_runtime/contact_sheet.jpg`
- `eval/siglip_runtime_quality_20260611_c032_attribute_prompt_runtime/pe_similarity_metrics.json`

Decision: `siglip_attribute_prompt_reference_control_pass`

The native SigLIP path now produces visually high-quality, reference-controlled
ComfyUI outputs when the prompt carries the reference's visible attributes.
The runtime node path is `AnimaSigLIP*`. User-facing reports should now call the
two practical variants `siglip_kv_init_w14` and `siglip_ref_retrieval_w14`.
Older logs used `siglip_pe_space_*` and `siglip_pe_retrieval_*` because these
SigLIP checkpoints were trained with PE-space initialization/retrieval anchors;
they are not PE-Core ComfyUI nodes. PE pooled-cosine is still used as an
auxiliary reference-similarity metric. `siglip_kv_init_w14` improves over no-IP
on 8/8 cases with mean uplift `0.0603`, and `siglip_ref_retrieval_w14` improves
on 7/8 cases with mean uplift `0.0670`.

This is not a claim that generic-prompt reference-only generation is solved.
The practical recipe is prompt/caption + adapter: good attribute prompts give
the base model the semantic target, while the native SigLIP adapter pulls
costume, palette, face framing, expression, and visual style toward the
reference. The next improvement should automate those attributes by generating
caption/attribute manifests for train/eval and by exposing the same recipe in
the ComfyUI workflow docs.

## 2026-06-12 automatic single-character attribute prompts

The next step automated the manual attribute prompt recipe for the color
single-character dataset. `tools/build_reference_prompt_manifest.py` scores a
bounded attribute vocabulary against each reference image with
`Qwen/Qwen3-VL-Embedding-2B`, writes an auto prompt manifest, and
`tools/siglip_auto_caption_eval.py` runs the native SigLIP ComfyUI API workflow
against no-IP plus two SigLIP checkpoints:

- `siglip_kv_init_w14`:
  `anima_siglip_ip_adapter_single_character_clean32_pe_space_init_0512_20260611.safetensors`
- `siglip_ref_retrieval_w14`:
  `anima_siglip_ip_adapter_single_character_clean32_pe_retrieval_0128_20260611.safetensors`

Evidence:

- `eval/siglip_runtime_quality_20260612_c033_auto_caption_runtime/report.md`
- `eval/siglip_runtime_quality_20260612_c033_auto_caption_runtime/contact_sheet.jpg`
- `eval/siglip_runtime_quality_20260612_c034_auto_caption_vocab2_runtime/report.md`
- `eval/siglip_runtime_quality_20260612_c034_auto_caption_vocab2_runtime/contact_sheet.jpg`
- `eval/siglip_runtime_quality_20260612_c034_auto_caption_vocab2_runtime/pe_similarity_metrics.json`

Decision: `siglip_auto_caption_single_character_visual_pass_pe_metric_caveat`

The c033 vocabulary was useful but under-described the red-haired female and
green monster references. The c034 expanded vocabulary fixes those visible
failures. On c034, `siglip_kv_init_w14` reaches mean PE pooled-cosine `0.7878`
with mean uplift `+0.1103` over no-IP and improves 7/8 cases.
`siglip_ref_retrieval_w14` reaches mean PE pooled-cosine `0.8227` with mean
uplift `+0.1452` and also improves 7/8 cases. The monster row is visually
closer with SigLIP but lower under PE pooled-cosine, so the metric is a useful
auxiliary signal, not the only pass/fail criterion.

## 2026-06-12 c035-c043 stronger-encoder prerequisite loop

c035 expanded the native SigLIP runtime gate to a 32-case single-character
suite. The best recipe, `siglip_ref_retrieval_w14`, reached mean uplift
`+0.0577` and improved rate `0.65625`; visual identity/distinctive-trait
coverage was `16/32`. Decision: `not_ready`.

The follow-up c036-c043 loop did not launch another adapter training run. It
tested whether the current feature spaces can provide a stronger identity gate:

- c036: QwenVL pooled metric is auxiliary only.
- c037: PE/QwenVL/SigLIP pooled features do not separate weak identity proxies.
- c038: the same feature pipeline passes duplicate-panel sanity checks.
- c039-c041: same-page mining plus character filtering produced only a tiny
  reviewed seed, with 4 usable positives.
- c042: reviewed seed feature probe did not pass; SigLIP layer `-6`
  `mean_max_token` is the most promising underpowered signal.
- c043: broad QwenVL face/upper-body filtering expanded the review pool to 30
  candidate pairs across 22 SG pages.
- c044-c045: conservative labels produced 8 usable positives and 15 negatives;
  QwenVL pooled passed the small reviewed identity proxy with margin `0.066209`
  and AUC `0.791667`.
- c046: QwenVL pooled ranked 65 face-filtered candidates from all 372
  same-page candidates; visual review confirms top20 has much better
  same-character density than the unranked pool.
- c047-c048: QwenVL top20 review yielded 14 usable positives, and the combined
  reviewed seed passed QwenVL pooled with margin `0.087629` and AUC `0.907407`.

Current decision: do not start long SigLIP adapter training yet. First convert
c048 into a larger, more diverse reviewed identity manifest using QwenVL pooled
as the primary ranking/gating metric. If QwenVL remains stable on the larger
set, use it as a metric for downstream adapter or metric-head training. If raw
features fail on the larger reviewed set, train a small metric head/calibrator
before any IP-Adapter K/V training.
