# SigLIP Contrastive c014 Weight Sweep 2026-06-11

## Decision

`weight_sweep_no_usable_quality`

- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c015_contrastive_weight_sweep/contact_sheet.jpg`
- Summary: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c015_contrastive_weight_sweep/summary.json`
- Checkpoint: `anima_siglip_ip_adapter_identity128_contrastive_0512_20260611.safetensors`

| Reference | In Train | Best Weight By dRef | Best dRef | Notes |
|---|---:|---:|---:|---|
| train000 | True | 1.8 | 53.3 | lower dRef, but washed-out crowd/sky scene instead of reference identity |
| train064 | True | 0.6 | 62.3 | mild reference color/layout pressure, still not faithful |
| train127 | True | 1.0 | 67.8 | wide room layout pressure, no original character/table recovery |
| heldout0112 | False | 1.8 | 80.3 | identity replaced by generic group portrait |
| heldout0560 | False | 1.8 | 70.0 | face/profile identity not preserved |
| heldout1008 | False | 1.0 | 66.2 | blue palette/scene pressure only, no barred-hands composition |

## Visual Review

Increasing weight strengthens reference pressure, but it does not produce a
usable high-quality control point. Higher weights often reduce the simple pixel
distance to the reference while making the image less faithful: speech bubbles
multiply, character count drifts, and train/held-out identities are replaced by
generic court or group scenes.

The best numeric weight is therefore not a pass. This sweep shows that the c014
contrastive checkpoint has tunable influence, but inference scale alone cannot
recover the missing identity/layout channel.
