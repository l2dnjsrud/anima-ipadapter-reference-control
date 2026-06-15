# c064 Failure-Attribute Embedding Probe

- Encoder: `pe`
- Manifest: `eval/qwenvl_c064_failure_attribute_embedding_probe_20260612/probe_manifest.jsonl`
- Cases: `3`
- Supported cases: `1`
- Support rate: `0.3333333333333333`
- Decision: `encoder_space_has_partial_supervised_signal`

## Case Decisions

- `heldout01` `old-face/speech-bubble-context-side-profile`: `encoder_space_not_enough`; best=`blend_species_face`, uplift=`0.043997764587402344`, top_margin=`0.020951151847839355`, c063_vs_blend_delta=`-0.020951151847839355`
- `heldout05` `old-bearded-official-black-hat-upper-body-crop`: `encoder_space_supports_supervised_signal`; best=`blend_species_face`, uplift=`0.09341549873352051`, top_margin=`0.09341549873352051`, c063_vs_blend_delta=`-0.11002564430236816`
- `heldout07` `non-human-green-monster-side-profile-red-eye`: `encoder_space_not_enough`; best=`no_ip`, uplift=`0.0`, top_margin=`0.09558939933776855`, c063_vs_blend_delta=`-0.013889431953430176`

This probe measures feature-space separation only. It does not by itself prove generation quality.
