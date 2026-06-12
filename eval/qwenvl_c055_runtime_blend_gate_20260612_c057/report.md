# QwenVL c057 Runtime Weight/Blend Gate

- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_c055_runtime_blend_gate_20260612_c057/contact_sheet.jpg`
- Previous retrieval checkpoint: `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- c055 mixed checkpoint: `anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors`
- Columns: reference / no_ip / prev_w14 / c055_w06 / c055_w08 / c055_w12 / blend_prev10_c05506 / blend_prev14_c05504.

Decision: `runtime_blend_prev14_c05504_best_so_far_larger_gate_required`

## Metric Summary

PE pooled metric:

| variant | mean uplift | improved rate |
| --- | ---: | ---: |
| `blend_prev14_c05504` | `+0.1064` | `0.875` |
| `prev_w14` | `+0.0983` | `0.875` |
| `blend_prev10_c05506` | `+0.0768` | `0.750` |
| `c055_w12` | `+0.0583` | `0.750` |
| `c055_w06` | `+0.0543` | `0.875` |
| `c055_w08` | `+0.0282` | `0.625` |

QwenVL pooled metric:

| variant | mean uplift | improved rate |
| --- | ---: | ---: |
| `prev_w14` | `+0.0377` | `0.875` |
| `blend_prev14_c05504` | `+0.0375` | `0.875` |
| `blend_prev10_c05506` | `+0.0357` | `0.875` |
| `c055_w12` | `+0.0289` | `0.750` |
| `c055_w08` | `+0.0136` | `0.750` |
| `c055_w06` | `+0.0096` | `0.500` |

## Visual Summary

`blend_prev14_c05504` is the strongest single runtime recipe so far. It beats the previous retrieval checkpoint on PE mean uplift (`+0.1064` vs `+0.0983`) and is effectively tied on QwenVL mean uplift (`+0.0375` vs `+0.0377`) with the same QwenVL improved rate (`0.875`).

This is not a final high-quality pass yet. The blend improves broad stability and several heldout rows, but exact pose/action, speech bubbles, and prop transfer remain inconsistent. `c055_w06` or `c055_w12` can beat the blend on special cases like `train14` or `train23`, which means the model family still lacks one universally reliable control recipe.

See:

- `pe_similarity_metrics.json`
- `qwenvl_similarity_metrics.json`
- `visual_audit.md`
- `visual_audit.json`

## Runtime Recipes

| variant | recipe |
| --- | --- |
| `no_ip` | `no adapter` |
| `prev_w14` | `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors@1.4` |
| `c055_w06` | `anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors@0.6` |
| `c055_w08` | `anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors@0.8` |
| `c055_w12` | `anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors@1.2` |
| `blend_prev10_c05506` | `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors@1 -> anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors@0.6` |
| `blend_prev14_c05504` | `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors@1.4 -> anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors@0.4` |

| sample | variant | prompt_id |
| --- | --- | --- |
| train00 | no_ip | `2e8d3947-047d-4260-be55-d983c44de7f2` |
| train00 | prev_w14 | `ac8e6760-630e-4414-b4b3-2e412a778712` |
| train00 | c055_w06 | `eda74880-44ac-41d3-9627-ace360f49b5f` |
| train00 | c055_w08 | `fb858bc5-8ab7-4d35-805a-45c8ae257971` |
| train00 | c055_w12 | `6bca3f85-7ed5-4c14-891f-2fa149fe0267` |
| train00 | blend_prev10_c05506 | `fe724200-75f6-476b-a55d-0b9b545cea80` |
| train00 | blend_prev14_c05504 | `1a533356-4417-4810-9611-99a03c4efa94` |
| train07 | no_ip | `69649691-af71-4306-b056-74009755552a` |
| train07 | prev_w14 | `cca85bce-d1bf-4014-a490-a484aab620b1` |
| train07 | c055_w06 | `42f692c8-9e97-4aad-9382-c3ccb6b8c4db` |
| train07 | c055_w08 | `cf5845a8-7933-4fe8-9906-f99a9f3134f2` |
| train07 | c055_w12 | `d4c89d0e-5731-4a41-b424-47f88376cf6b` |
| train07 | blend_prev10_c05506 | `b3c1cef2-f7cb-4585-a893-7d93153d8bb2` |
| train07 | blend_prev14_c05504 | `1a6d3d3f-576f-4754-bbb6-355c658b9804` |
| train14 | no_ip | `c593bb82-a4df-42c9-8cb7-c5d31a7049fb` |
| train14 | prev_w14 | `7547fe11-551a-41fc-b148-e40bcd1d13c5` |
| train14 | c055_w06 | `ddfebb9f-55ef-4fa8-ab3c-d57c8e9b05a8` |
| train14 | c055_w08 | `47f59542-4dd0-482e-a882-0a880f122ddb` |
| train14 | c055_w12 | `a81a9a81-36b4-4e7c-805d-1fb81ad6c7c2` |
| train14 | blend_prev10_c05506 | `be9425d4-03cd-437f-9cd4-b83e138d96fe` |
| train14 | blend_prev14_c05504 | `bf2524c3-9857-4a77-a996-4acbe5c1eff0` |
| train23 | no_ip | `bf3e1f39-09af-4b54-b36c-6859ff38af0b` |
| train23 | prev_w14 | `91295cfd-c377-40f6-a592-176fd5630ec1` |
| train23 | c055_w06 | `931ec449-10b6-4945-8d1c-88c9b6c931f1` |
| train23 | c055_w08 | `2a5735af-2adc-4d8e-ad9d-a507cdf020dc` |
| train23 | c055_w12 | `94fb74e9-96c7-4194-8e74-0d964c362b14` |
| train23 | blend_prev10_c05506 | `ae764a96-098f-4167-afc2-3bda8c84513f` |
| train23 | blend_prev14_c05504 | `a3ee4117-45ee-4dcc-b47f-787b24cb4c15` |
| heldout00 | no_ip | `3e7d06b7-a89d-4be2-a163-698a04dab95e` |
| heldout00 | prev_w14 | `b4362695-bdfc-4b9c-b633-203c8fd6d0a3` |
| heldout00 | c055_w06 | `7ac2576d-d4e9-4966-8e2d-532128da26a3` |
| heldout00 | c055_w08 | `dd11ceac-7741-4b69-9970-f3c07540ab9c` |
| heldout00 | c055_w12 | `dd04d2f2-17ac-493a-b090-747772be8a4b` |
| heldout00 | blend_prev10_c05506 | `2cd48bf8-9ae3-4265-a8c6-931a4a725580` |
| heldout00 | blend_prev14_c05504 | `e3ab685e-a4ac-4848-8aa1-652195989ee8` |
| heldout02 | no_ip | `771ae7db-2585-44c6-ae11-5534582787a0` |
| heldout02 | prev_w14 | `9d1b3d17-ce06-4413-beec-3f9da4df305d` |
| heldout02 | c055_w06 | `44bc09d1-c66e-4b16-a4c6-31d9276892c9` |
| heldout02 | c055_w08 | `ac8cb72b-e3e8-4a36-bec7-ddceaa8db813` |
| heldout02 | c055_w12 | `dc16c05e-f712-4088-934c-16c7faef52e3` |
| heldout02 | blend_prev10_c05506 | `fb0ba2c0-24a2-4f0a-b2be-00ff2570a6c1` |
| heldout02 | blend_prev14_c05504 | `795c9e16-1dee-4be9-91c2-4932799be5f2` |
| heldout05 | no_ip | `47a0a9bd-cc3d-401e-85be-50c31a018691` |
| heldout05 | prev_w14 | `03fe38e8-3213-4289-a220-58bd41e3835a` |
| heldout05 | c055_w06 | `bed65136-9051-4191-bda8-5c32d4fb1c20` |
| heldout05 | c055_w08 | `587b16e9-871d-4f9d-af10-abfdcac9e9dd` |
| heldout05 | c055_w12 | `1d776972-ad87-40a6-925a-638e154d1fe2` |
| heldout05 | blend_prev10_c05506 | `be40694c-a614-44ac-8673-bb74843371ed` |
| heldout05 | blend_prev14_c05504 | `d1be00ea-7558-4f8c-acd2-d45e24d6df3c` |
| heldout07 | no_ip | `566bd42a-9b6c-4714-8477-94bfad483cb6` |
| heldout07 | prev_w14 | `d63fd5fa-05d3-4c49-bac5-1caca9754b0d` |
| heldout07 | c055_w06 | `db8dbeaf-80ca-46bb-812e-02a73d1f9c5e` |
| heldout07 | c055_w08 | `c18001eb-7932-4e0e-9baf-1dd5d356fc43` |
| heldout07 | c055_w12 | `22b489df-bdc5-4cf4-ad58-e9d0928f3def` |
| heldout07 | blend_prev10_c05506 | `ae503ea8-acbd-4e81-94d3-692c3bdf1f52` |
| heldout07 | blend_prev14_c05504 | `f74e1921-0f14-4635-9047-439fdc5d5faf` |
