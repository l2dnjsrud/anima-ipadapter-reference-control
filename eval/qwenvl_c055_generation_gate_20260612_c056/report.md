# QwenVL c056 c055 Generation Gate

- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_c055_generation_gate_20260612_c056/contact_sheet.jpg`
- Previous retrieval checkpoint: `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- c053/c052-positive checkpoint: `anima_qwenvl_ip_adapter_c052_identity_retrieval_0064_20260612.safetensors`
- c055 mixed checkpoint: `anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors`
- Columns: reference / no_ip / qwen_prev_retrieval_w14 / qwen_c052_w14 / qwen_c055_w1 / qwen_c055_w14.

Decision: `qwen_c055_improves_c052_not_quality_pass_prev_retrieval_still_best`

## Metric Summary

PE pooled metric:

| variant | mean uplift | improved rate |
| --- | ---: | ---: |
| `qwen_prev_retrieval_w14` | `+0.0983` | `0.875` |
| `qwen_c052_w14` | `+0.0394` | `0.625` |
| `qwen_c055_w1` | `+0.0502` | `0.750` |
| `qwen_c055_w14` | `+0.0460` | `0.750` |

QwenVL pooled metric:

| variant | mean uplift | improved rate |
| --- | ---: | ---: |
| `qwen_prev_retrieval_w14` | `+0.0377` | `0.875` |
| `qwen_c052_w14` | `+0.0231` | `0.625` |
| `qwen_c055_w1` | `+0.0257` | `0.750` |
| `qwen_c055_w14` | `+0.0339` | `0.875` |

## Visual Summary

c055 is a real improvement over c052, but not a final quality pass. It restores or improves distinctive traits on `train14`, `heldout00`, `heldout05`, and `heldout07`, and remains competitive on `train23` and `heldout02`. It still loses visible reference fidelity on `train00` and `train07`, where the output is mainly prompt-driven and misses reference pose/action or page-specific framing.

The previous retrieval checkpoint still wins the aggregate PE metric by a wide margin and narrowly wins QwenVL mean uplift. c055 w1.4 reaches the same QwenVL improved rate as previous retrieval (`0.875`) but not the mean uplift, so this is a continuation improvement, not a deployable high-quality reference-control gate.

See:

- `pe_similarity_metrics.json`
- `qwenvl_similarity_metrics.json`
- `visual_audit.md`
- `visual_audit.json`

| sample | variant | prompt_id |
| --- | --- | --- |
| train00 | no_ip | `aa33933f-471b-4cd2-90ea-605107c5cb50` |
| train00 | qwen_prev_retrieval_w14 | `886ef2b5-4b1d-48d7-abb4-c3d659d3f472` |
| train00 | qwen_c052_w14 | `4eb99fd6-e7c4-42fe-b44c-c372f641754c` |
| train00 | qwen_c055_w1 | `62276477-5c16-4409-a8dd-87ea2c5e34ba` |
| train00 | qwen_c055_w14 | `f8c990bc-bb6d-4b4f-9593-3a9f5f3850fb` |
| train07 | no_ip | `ba467941-1af0-4756-a602-ddd0428e58aa` |
| train07 | qwen_prev_retrieval_w14 | `98e1e9cb-eadc-44d9-8293-5097d7526ea8` |
| train07 | qwen_c052_w14 | `7d8c63c6-e5ac-4168-bccc-6165ae818afe` |
| train07 | qwen_c055_w1 | `36d074e8-89c9-4316-b8b9-739ebb87ae8c` |
| train07 | qwen_c055_w14 | `04d14c61-4b34-47aa-b256-c6b1d5b9d3c1` |
| train14 | no_ip | `b62e03c6-afc9-4ec6-bf45-17f925e4d315` |
| train14 | qwen_prev_retrieval_w14 | `2c3d2292-fe15-43ef-80e9-5175bb117fe4` |
| train14 | qwen_c052_w14 | `ad8a5001-de3a-40f8-9be4-75ac4b440981` |
| train14 | qwen_c055_w1 | `58427db7-0f4a-4404-a9ee-fdc89cbd3e20` |
| train14 | qwen_c055_w14 | `862e64c4-48a9-4f43-a7a8-38d02bcdbd12` |
| train23 | no_ip | `034fb232-95e1-4629-a259-37fc6501dbf4` |
| train23 | qwen_prev_retrieval_w14 | `81b5def5-9be5-4136-b4ba-ed4a3d770cfe` |
| train23 | qwen_c052_w14 | `29a7b766-00a8-4e38-abee-acade5dbefc6` |
| train23 | qwen_c055_w1 | `fc101321-93e5-405b-8474-9e00e72831e7` |
| train23 | qwen_c055_w14 | `160f47bb-0738-4984-afec-8a64cc0bc31b` |
| heldout00 | no_ip | `811fed6a-6d6e-4f76-bfa9-084eace3e2dd` |
| heldout00 | qwen_prev_retrieval_w14 | `51862ca3-d13f-4eee-8623-e5081db36a05` |
| heldout00 | qwen_c052_w14 | `18ef8e10-1353-4c5c-9b9e-857d81f546c0` |
| heldout00 | qwen_c055_w1 | `3e223cb4-2ca8-44f7-8c76-91cdd9763236` |
| heldout00 | qwen_c055_w14 | `ee3e5aee-d257-4387-940e-5b2db18146be` |
| heldout02 | no_ip | `b9ce111e-1a4f-4424-a272-50de6e6f2b23` |
| heldout02 | qwen_prev_retrieval_w14 | `aa8bedc2-3699-4b8e-91f6-1f66755075d2` |
| heldout02 | qwen_c052_w14 | `ebae5135-d5b6-4bcc-8d0b-2baf2c86593f` |
| heldout02 | qwen_c055_w1 | `531395c8-1242-4f68-8508-5cc23ad284b8` |
| heldout02 | qwen_c055_w14 | `2188e5e0-339c-4ddf-a13a-61a6105227dc` |
| heldout05 | no_ip | `19bd6bb2-e346-472c-a1af-f00f4468af9d` |
| heldout05 | qwen_prev_retrieval_w14 | `03821475-37f7-4b04-a579-fa13823468e6` |
| heldout05 | qwen_c052_w14 | `d953c379-8f29-4c5a-b83a-10843f27c4b1` |
| heldout05 | qwen_c055_w1 | `44661ec3-2dec-445b-9a2e-1d72d044bca5` |
| heldout05 | qwen_c055_w14 | `ca6dba1a-62b1-4a11-ba6b-e469260bdeb6` |
| heldout07 | no_ip | `5b1f2eaa-d79f-495b-a582-5da25aa53311` |
| heldout07 | qwen_prev_retrieval_w14 | `b064bdf6-3129-4f35-b380-4e30f71950ae` |
| heldout07 | qwen_c052_w14 | `9cc4e600-af01-46af-8c91-0a13edee202a` |
| heldout07 | qwen_c055_w1 | `62fb13da-3ac4-496e-9872-cd70db8b03bb` |
| heldout07 | qwen_c055_w14 | `2b495728-e565-47b7-aea6-046649ec7226` |
