# c064 Failure-Attribute Embedding Probe

- Encoder: `Qwen/Qwen3-VL-Embedding-2B`
- Manifest: `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/probe_manifest.jsonl`
- Cases: `3`
- Supported cases: `1`
- Support rate: `0.3333333333333333`
- Decision: `encoder_space_has_partial_supervised_signal`

## Case Decisions

- `heldout01` `old-face/speech-bubble-context-side-profile`: `encoder_space_supports_supervised_signal`; best=`c063_calibrator_only_w14`, uplift=`0.09628546237945557`, top_margin=`0.0220983624458313`, c063_vs_blend_delta=`0.0220983624458313`
- `heldout05` `old-bearded-official-black-hat-upper-body-crop`: `encoder_space_not_enough`; best=`blend_species_face`, uplift=`0.016989946365356445`, top_margin=`0.016577184200286865`, c063_vs_blend_delta=`-0.016577184200286865`
- `heldout07` `non-human-green-monster-side-profile-red-eye`: `encoder_space_not_enough`; best=`no_ip`, uplift=`0.0`, top_margin=`0.05199939012527466`, c063_vs_blend_delta=`-0.0016797184944152832`

This probe measures feature-space separation only. It does not by itself prove generation quality.
