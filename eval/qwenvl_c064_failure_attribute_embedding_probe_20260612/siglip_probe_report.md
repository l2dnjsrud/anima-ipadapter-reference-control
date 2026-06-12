# c064 Failure-Attribute Embedding Probe

- Encoder: `google/siglip2-base-patch16-512`
- Manifest: `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/probe_manifest.jsonl`
- Cases: `3`
- Supported cases: `0`
- Support rate: `0.0`
- Decision: `encoder_side_checkpoint_required`

## Case Decisions

- `heldout01` `old-face/speech-bubble-context-side-profile`: `encoder_space_not_enough`; best=`no_ip`, uplift=`0.0`, top_margin=`0.02801680564880371`, c063_vs_blend_delta=`0.01782923936843872`
- `heldout05` `old-bearded-official-black-hat-upper-body-crop`: `encoder_space_not_enough`; best=`no_ip`, uplift=`0.0`, top_margin=`0.010737240314483643`, c063_vs_blend_delta=`-0.006295323371887207`
- `heldout07` `non-human-green-monster-side-profile-red-eye`: `encoder_space_not_enough`; best=`no_ip`, uplift=`0.0`, top_margin=`0.014181911945343018`, c063_vs_blend_delta=`-0.023101747035980225`

This probe measures feature-space separation only. It does not by itself prove generation quality.
