# IP-Adapter Reference Research

Date: 2026-06-10

## Question

Can we download the public Wenaka image dataset, caption it ourselves, and use it
to train a high-quality Anima SigLIP IP-Adapter?

## Short Answer

Yes, downloading and captioning can build the missing text side of the dataset.
It is not enough by itself. For the Wenaka/Anima Stage-2 trainer, we also need a
`ref_id -> tgt_id` pairing rule.

The practical path is:

1. Download/extract the public image shards after approval.
2. Caption or tag the images locally.
3. Mine reference-target pairs from image similarity, source grouping, or visual
   clustering.
4. Validate a small JSONL through `training/siglip_proof.py`.
5. Run a short pilot training before full training.

## Tencent IP-Adapter

Source:

```text
https://github.com/tencent-ailab/IP-Adapter
commit 62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2
```

Findings:

- Training uses a JSON dataset with `image_file` and `text`.
- The CLIP text encoder, VAE, UNet, and CLIP vision encoder are frozen.
- Only the image projection model and adapter modules are trained.
- The image encoder feature is `CLIPVisionModelWithProjection.image_embeds`.
- Projected image tokens are concatenated to text hidden states.
- Loss is diffusion noise-prediction MSE.

Implication:

Captioning downloaded images is a valid part of IP-Adapter dataset construction,
but Tencent's trainer is CLIP/SD oriented. It does not remove the need for
Anima-specific ref/tgt pairing.

## ComfyUI IPAdapter Plus

Source:

```text
https://github.com/cubiq/ComfyUI_IPAdapter_plus
commit a0f451a5113cf9becb0847b92884cb10cbdec0ef
```

Findings:

- The standard ComfyUI graph shape is loader -> apply -> sampler.
- The implementation expects ComfyUI `CLIP_VISION`.
- It maps presets to specific clip_vision files, ipadapter checkpoints, and
  sometimes LoRA files.
- It has rich weight types for style/composition scheduling.

Implication:

We should copy the user experience pattern, not the implementation. Anima
SigLIP needs native encoder/checkpoint handling because the existing IPAdapter
Plus implementation is tied to SD/SDXL and CLIP-Vision checkpoint families.

## FaceID-Like Model Feasibility

Sources:

```text
https://huggingface.co/h94/IP-Adapter-FaceID
https://github.com/tencent-ailab/IP-Adapter/wiki/IP%E2%80%90Adapter%E2%80%90Face
https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter
https://arxiv.org/abs/2503.07091
```

Findings:

- IP-Adapter-FaceID replaces or augments normal CLIP image embeddings with face
  recognition embeddings.
- FaceID-Plus combines face ID embeddings for identity with CLIP image
  embeddings for face structure.
- FaceID-PlusV2 adds a controllable structure branch, so identity and structure
  strength can be tuned separately.
- Tencent's Face training notes use high-resolution single-face filtering,
  face crop/augmentation, InsightFace ArcFace normalized embeddings, and
  OpenCLIP ViT-H embeddings.
- Public FaceID-6M research shows that high-quality FaceID training is very
  data-driven and benefits from million-scale filtered face/text examples.

Implication:

Training a FaceID-PlusV2-like model is possible, but the hard part is not the
adapter layer. The hard part is building an anime/manhwa identity embedding
space that is reliable across expressions, angles, panel crops, hair changes,
and stylized line/color renderings. Off-the-shelf InsightFace is optimized for
real human faces, so it is not enough for a high-quality Anima character ID
model by itself.

Practical tiers:

1. Prototype: reuse real-face InsightFace or CLIP/SigLIP features and train only
   the Anima adapter. This is quick, but likely weak for manhwa characters.
2. Better pilot: mine same-character groups from the color dataset, train a
   small anime character metric model, then feed that embedding to the adapter.
3. Production target: train an Anima character-ID encoder plus a structure
   encoder and an adapter jointly or in staged phases.

## Anima-Native Embedding Checkpoint Feasibility

An Anima-native IPAdapter embedding/checkpoint is also possible. It should not
try to load as a standard SD/SDXL IPAdapter checkpoint. It should be a native
Anima checkpoint family with:

- image encoder: SigLIP2 or an anime-tuned SigLIP/character encoder;
- resampler/projector: TimeResampler or Perceiver-style projection into Anima
  token dimensions;
- injection: native IP cross-attention or model patching at Anima attention
  blocks;
- training rows: `ref_id`, `tgt_id`, `prompt`, plus optional identity/group
  labels when available;
- evaluation: reference similarity, prompt following, composition preservation,
  and style leakage tests.

Captioning downloaded images can supply prompts, but it cannot create identity
control alone. For strong reference control we need ref-target pair mining,
character grouping, or a separate identity-label source.

## Arca Posts

The supplied Arca links returned HTTP 403 Cloudflare challenge from this
environment:

```text
https://arca.live/b/aiart/156883379
https://arca.live/b/aiart/156734804
https://arca.live/b/aiart/157369400
```

They cannot be cited or summarized until accessible text, screenshots, or HTML
is available.

## Current Local Pilot

The local color-panel dataset is a useful pilot target:

```text
/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best
```

Measured by `tools/generate_pair_manifest.py`:

```json
{
  "directories": 269,
  "source_images": 1571,
  "rows": 1537,
  "skipped_singleton_directories": 34
}
```

Generated sample:

```text
evidence/color_panel_style_v5_best_sample_pairs_64.jsonl
```

Validation:

```json
{
  "checked_rows": 8,
  "missing_images": [],
  "full_training_started": false
}
```

## Recommendation

Proceed in two tracks:

- Local pilot: use the color-panel v5 sample pairs to validate the training loop
  and ComfyUI checkpoint loading quickly.
- Full dataset: download Wenaka shards only after explicit approval, then run
  captioning and pair mining. Captioning alone is not the final dataset.
