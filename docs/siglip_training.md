# SigLIP2 Native Branch And Training Plan

## What landed locally

- Native SigLIP2/IP-Adapter scaffolding lives in `siglip_model.py`, `siglip_checkpoint.py`, and `native_siglip.py`.
- Real frozen-Anima SigLIP2 smoke training lives in `training/siglip_real_smoke.py` with small helpers under `training/siglip_smoke_*.py`.
- The upstream Wenaka `TransformerEncoderLayer(memory=...)` bug is avoided by a real cross-attention fusion layer.
- The upstream loader constructor mismatch is avoided by detecting checkpoint tensor shapes and constructing `IPAdapterSigLIP` with explicit dimensions.
- PE-Core checkpoints are rejected by the SigLIP loader with a clear message instead of being partially loaded as SigLIP.
- `AnimaSigLIPIPAdapterLoader` uses the ComfyUI `ipadapter` model selector, not a raw filesystem path.
- `AnimaSigLIPIPAdapterApply` clones the ComfyUI `MODEL` and registers an `attn2_patch`, matching ComfyUI `ModelPatcher.set_model_attn2_patch` semantics.
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

## What blocks real full training

- No high-quality trained SigLIP2 TimeResampler/IPCrossAttn checkpoint is present in this repo. The current SigLIP checkpoint is a one-step smoke artifact only.
- The Hugging Face dataset is about 36.5 GiB; I did not download it without explicit approval.
- The public dataset appears to contain image tar shards only. Wenaka's training script expects a paired `training_pairs_final2.jsonl` with `ref_id`, `tgt_id`, and `prompt`, which is not exposed in the dataset preview or file list.
- Full training still needs explicit runtime approval, a target output path that the current user can write, and a quality gate based on reference-control contact sheets.

## Full-training outline

1. Choose the approved dataset scope: local color-panel pairs, downloaded Wenaka shards, or both.
2. Generate paired metadata with `ref_id`, `tgt_id`, and `prompt`.
3. Run the dry-run proof and local manifest audit against the selected rows.
4. Run bounded real smoke through the frozen Anima DiT/VAE/text-encoder loss path.
5. Save a SigLIP checkpoint with `resampler.time_proj.*`, `resampler.layers.*`, `intermediate_encoder.*`, `ip_cross_attns.*`, and `ip_scales.*` keys.
6. Load it through `AnimaSigLIPIPAdapterLoader`; do not use the PE-Core checkpoint with this loader.
7. Scale to a pilot training run, then evaluate against no-IP and PE-Core baselines with contact sheets before calling it usable.
