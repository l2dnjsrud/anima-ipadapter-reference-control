# QwenVL c054 c052 Generation Smoke Gate

- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_c052_generation_gate_20260612_c054/contact_sheet.jpg`
- Previous retrieval checkpoint: `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- c053 checkpoint: `anima_qwenvl_ip_adapter_c052_identity_retrieval_0064_20260612.safetensors`
- Columns: reference / no_ip / qwen_prev_retrieval_w14 / qwen_c052_w1 / qwen_c052_w14.

Decision: `qwen_c052_partial_visual_improvement_metric_regression_not_quality_pass`

## Metric Summary

PE pooled metric:

| variant | mean uplift | improved rate |
| --- | ---: | ---: |
| `qwen_prev_retrieval_w14` | `+0.0983` | `0.875` |
| `qwen_c052_w1` | `+0.0377` | `0.625` |
| `qwen_c052_w14` | `+0.0394` | `0.625` |

QwenVL pooled metric:

| variant | mean uplift | improved rate |
| --- | ---: | ---: |
| `qwen_prev_retrieval_w14` | `+0.0377` | `0.875` |
| `qwen_c052_w1` | `+0.0174` | `0.500` |
| `qwen_c052_w14` | `+0.0231` | `0.625` |

## Visual Summary

c053 is not a metric pass, but it has real visual wins. It improves or competes on `train23`, `heldout00`, `heldout02`, and `heldout07`. The strongest gains are the old monk row and green demon row, where reference-specific traits survive better than in earlier QwenVL runs. It loses or stays weak on `train14` and `heldout05`, and the aggregate PE/QwenVL metrics still prefer the previous retrieval checkpoint.

See:

- `visual_audit.md`
- `visual_audit.json`

| sample | variant | prompt_id |
| --- | --- | --- |
| train00 | no_ip | `52452700-7360-48ba-809d-0874757f3e01` |
| train00 | qwen_prev_retrieval_w14 | `f28aa182-2c65-47ba-a813-f825412cd4a4` |
| train00 | qwen_c052_w1 | `91a590df-43ae-48d5-8133-22ce46853199` |
| train00 | qwen_c052_w14 | `906c2987-3a22-4beb-8e7a-f6c6fd4aa930` |
| train07 | no_ip | `c139c93a-3211-4563-bf9d-e4f0848d1175` |
| train07 | qwen_prev_retrieval_w14 | `0e886dc0-e35a-4fd6-9a6f-60f79744414c` |
| train07 | qwen_c052_w1 | `968a7076-a466-4a9f-9791-b71689c75588` |
| train07 | qwen_c052_w14 | `4cbeb7e8-3f23-4c49-b773-febea9920bb2` |
| train14 | no_ip | `8c5cef4a-7dee-4d31-9fe7-4a9871dbb124` |
| train14 | qwen_prev_retrieval_w14 | `277eafb3-c067-46cf-8338-5e39670ab71a` |
| train14 | qwen_c052_w1 | `490d105f-0d53-421e-99f8-bc377b5100c3` |
| train14 | qwen_c052_w14 | `9363dc5f-63a9-4cc1-aef8-41b62cc47179` |
| train23 | no_ip | `589c77fa-13f1-424e-a0b8-58cf7650f129` |
| train23 | qwen_prev_retrieval_w14 | `d76d318f-32e6-4930-9826-f37131f904f5` |
| train23 | qwen_c052_w1 | `1b0756d2-c288-4cde-b0d3-e0a2d3ab5d9d` |
| train23 | qwen_c052_w14 | `a9a36119-dedb-42bf-aa8a-81c28a45e0df` |
| heldout00 | no_ip | `228b14c1-9b7c-4578-93fc-2a32f7eb2674` |
| heldout00 | qwen_prev_retrieval_w14 | `1014c2fe-958e-4970-b431-4286c28aa526` |
| heldout00 | qwen_c052_w1 | `8276bebe-f863-4865-b323-eb34f9c418bb` |
| heldout00 | qwen_c052_w14 | `975ed656-d315-4dea-a220-f1fd0a1b0bb5` |
| heldout02 | no_ip | `565e0f43-5720-41a9-af10-7bf6d8cb7655` |
| heldout02 | qwen_prev_retrieval_w14 | `c72d236e-7669-4052-9464-07f09d031075` |
| heldout02 | qwen_c052_w1 | `560ee408-34aa-4ef3-a67d-efcb94b5953d` |
| heldout02 | qwen_c052_w14 | `5569b476-f014-40d9-8ce5-40eb966bddb9` |
| heldout05 | no_ip | `eaaf5079-8d05-4a71-871a-89c0b421db69` |
| heldout05 | qwen_prev_retrieval_w14 | `d2c60ec5-c7a4-4444-a530-fbe1116043f1` |
| heldout05 | qwen_c052_w1 | `f5994fc9-b23b-41a0-a360-5046e5853ddb` |
| heldout05 | qwen_c052_w14 | `72479672-a346-4ef6-8187-fe8a8d659b84` |
| heldout07 | no_ip | `12a9c602-6982-4908-adc1-9743ebae14e0` |
| heldout07 | qwen_prev_retrieval_w14 | `ef5a6ea8-94cc-42c0-b942-a7e8d4dbcea6` |
| heldout07 | qwen_c052_w1 | `5a5001e4-d4d0-4f57-84fb-8e8cf36a5a2f` |
| heldout07 | qwen_c052_w14 | `5de6528e-bd54-487f-936d-f39933ececb8` |
