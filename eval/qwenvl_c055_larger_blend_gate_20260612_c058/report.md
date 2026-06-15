# QwenVL c058 Larger Runtime Blend Gate

- Train contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_c055_larger_blend_gate_20260612_c058/contact_sheet_train.jpg`
- Heldout contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_c055_larger_blend_gate_20260612_c058/contact_sheet_heldout.jpg`
- Previous retrieval checkpoint: `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- c055 mixed checkpoint: `anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors`
- Columns: reference / no_ip / prev_w14 / blend_prev14_c05504.

Decision: `best_runtime_candidate_not_final_quality_pass_distillation_or_training_next`

## Metric Summary

| encoder metric | variant | mean cosine | mean uplift vs no-IP | improved rate |
| --- | --- | ---: | ---: | ---: |
| PE | `blend_prev14_c05504` | `0.803729` | `+0.049596` | `0.725` |
| PE | `prev_w14` | `0.783374` | `+0.029240` | `0.750` |
| QwenVL | `blend_prev14_c05504` | `0.822666` | `+0.041589` | `0.800` |
| QwenVL | `prev_w14` | `0.817265` | `+0.036187` | `0.725` |

`blend_prev14_c05504` is the best runtime recipe on mean PE and QwenVL similarity across the 40-sample larger gate. It improves the QwenVL improved-rate from `0.725` to `0.800`, although PE improved-rate is slightly lower than `prev_w14`.

## Visual Audit Summary

See `visual_audit.md` and `visual_audit.json`.

The heldout contact sheet supports the metric direction but does not justify a final quality pass. The blend recipe improves broad character cues such as hair, costume, bald/old head, official headwear, and red-haired woman identity. However it still misses exact pose/crop, speech bubbles, panel/page layout, distinctive props, and non-human silhouette. `heldout07` is the clearest failure: the reference is a green demon side-head close-up, while `prev_w14` and `blend_prev14_c05504` drift toward full-body dark demon/assassin imagery.

## Decision

c058 promotes `blend_prev14_c05504` to the current best runtime candidate, but not to a production-ready high-quality Anima reference-control model.

Next loop should stop simple runtime weight-only searching and move to one of:

- distill `prev_w14 + c055_w04` into a single checkpoint, then rerun the same 40-sample gate;
- run failure-focused continuation/encoder adaptation with c058 failure classes emphasized: pose/crop, props, speech bubbles, and non-human silhouettes.

| sample | variant | prompt_id |
| --- | --- | --- |
| train00 | no_ip | `ee2d99ce-6687-4d1c-a22c-b9862deda291` |
| train00 | prev_w14 | `812dde3c-5c07-41ae-b9d3-5ef9f4591edb` |
| train00 | blend_prev14_c05504 | `f10f190e-4dff-45c5-877b-f98396fbf5d1` |
| train01 | no_ip | `5f2692a1-204c-44ee-9581-e5075a13f8f0` |
| train01 | prev_w14 | `34dcb538-637c-456a-ab4f-382594f08fd7` |
| train01 | blend_prev14_c05504 | `d992bf4a-6f4f-429d-a23d-819b5ab98ab8` |
| train02 | no_ip | `25608f89-344c-4df3-a206-f8037cec985a` |
| train02 | prev_w14 | `17cda925-e542-4737-bec2-ef13e5e433ca` |
| train02 | blend_prev14_c05504 | `25e88262-fe73-4621-bb97-854c8a43f22f` |
| train03 | no_ip | `a9d188c3-0cf3-4c32-a1d4-3ee72937dc7c` |
| train03 | prev_w14 | `694481cb-7bc5-4b61-801f-fc35de220a87` |
| train03 | blend_prev14_c05504 | `e8884ba0-f97c-4ae1-8b15-835c8b4fda3a` |
| train04 | no_ip | `1ac46114-b00f-4099-ac11-60a34dc03d82` |
| train04 | prev_w14 | `34a3397d-ca1d-4429-90a7-774c9659fd3f` |
| train04 | blend_prev14_c05504 | `9587f486-5ed2-45e1-b31f-0ef1f7262d5f` |
| train05 | no_ip | `635f06c2-2add-4ca5-8e06-6b8db81cbee8` |
| train05 | prev_w14 | `dee80179-5479-409d-97d9-ecadda3f50ec` |
| train05 | blend_prev14_c05504 | `981dbf3b-c71c-47d0-9120-fa264978c589` |
| train06 | no_ip | `ff6f4bdd-b393-4fa5-b825-9a42d7b03c06` |
| train06 | prev_w14 | `690b2f6f-f1da-4ba2-be40-fddfd369967f` |
| train06 | blend_prev14_c05504 | `a98c72e9-a4b7-43e7-a7ec-2f7c28c51960` |
| train07 | no_ip | `2d852a09-4deb-4cb0-af6a-b63f6182e925` |
| train07 | prev_w14 | `b293be87-4dd1-4d6b-b292-a971c4ec1201` |
| train07 | blend_prev14_c05504 | `20ea7b6b-f151-4ac9-859a-a9b42656f57b` |
| train08 | no_ip | `87908e21-214b-462f-9a53-965a3aee05d5` |
| train08 | prev_w14 | `1c659b26-6aa9-4d44-84b6-9e6c7e4509eb` |
| train08 | blend_prev14_c05504 | `aa6b21c8-09bd-4710-a592-c0ed15768986` |
| train09 | no_ip | `f7cd4f72-b9a0-48b2-a227-9ab675e466fc` |
| train09 | prev_w14 | `f687d6ad-75ff-42cb-8c8c-a70847a805b9` |
| train09 | blend_prev14_c05504 | `7022b8aa-b4ff-4e74-96a8-37fa6fedb052` |
| train10 | no_ip | `67a3766d-94a4-43ab-8652-31e42e6caa9b` |
| train10 | prev_w14 | `60bacc20-044d-4f0e-a13f-0e6aaf5786f7` |
| train10 | blend_prev14_c05504 | `bc3bccc6-73f4-49bf-b970-f9c7e298a142` |
| train11 | no_ip | `d3a381a0-04f1-4e65-b809-394a6825161b` |
| train11 | prev_w14 | `8fbf3b40-a9c8-4fe5-be54-c64fcf7458f2` |
| train11 | blend_prev14_c05504 | `df78ee91-3387-43bc-bafa-6429494ee2fa` |
| train12 | no_ip | `5f1a91ec-f38e-4ac6-a4b9-41b06f7ae69d` |
| train12 | prev_w14 | `4b67976b-87b4-446c-9004-de6b281b23fd` |
| train12 | blend_prev14_c05504 | `cc351e6a-4ff6-4740-bc4f-476d1ca7de87` |
| train13 | no_ip | `803dfffd-6967-4d05-a8d0-26dcb26fd9a9` |
| train13 | prev_w14 | `81a88c87-35c1-499d-ae9b-286854a57dc4` |
| train13 | blend_prev14_c05504 | `7d84af50-702b-4fa5-8a45-b3ca614b030c` |
| train14 | no_ip | `d1d5b13e-f1aa-4145-a8e6-0ece8f86c3cd` |
| train14 | prev_w14 | `fdfe2160-5625-4ffa-8e2b-02566e8c9c21` |
| train14 | blend_prev14_c05504 | `2b764ab7-84f2-4dc3-970e-fd2ae5688794` |
| train15 | no_ip | `85e4c61c-5445-4a2a-9d2a-169471db1e64` |
| train15 | prev_w14 | `d177509f-1917-44a8-8e17-638eb2465cc0` |
| train15 | blend_prev14_c05504 | `7c35cbfe-fb95-4b11-81b1-b2eda40ed533` |
| train16 | no_ip | `fcd06ada-480f-4560-ae37-416ee9bc0a74` |
| train16 | prev_w14 | `37e4c00a-5b27-442d-8e3c-ac6944a70369` |
| train16 | blend_prev14_c05504 | `fb331cef-b584-4cd4-b893-20b124c92d2b` |
| train17 | no_ip | `9f5689d1-5e44-42d5-8733-3e2bb9b69c35` |
| train17 | prev_w14 | `11bdff50-fbbc-45f9-9e1d-3e8ea7804bd9` |
| train17 | blend_prev14_c05504 | `eb4d72cc-963b-48d1-859e-3b3e443c72f4` |
| train18 | no_ip | `bfabb5bb-323e-4001-8aea-4d170f0ee47b` |
| train18 | prev_w14 | `f4614b85-7285-4647-9545-3828040408c6` |
| train18 | blend_prev14_c05504 | `4a701899-3368-4432-8023-55888c0db72e` |
| train19 | no_ip | `0f2bc0d1-71b9-487d-af96-0f0242d8ff40` |
| train19 | prev_w14 | `ff5a07c4-159d-4c57-ac8b-b653d82e787d` |
| train19 | blend_prev14_c05504 | `1c0215a9-c4b6-432b-83ab-efaf7c1fdcee` |
| train20 | no_ip | `befe2651-839e-4c34-8ad6-000b65f8c223` |
| train20 | prev_w14 | `ae84550a-3e31-4cd4-888b-0bf12df9cc9c` |
| train20 | blend_prev14_c05504 | `349f9228-b52f-40ae-a71e-42958ad8e9f1` |
| train21 | no_ip | `8bfde85c-760e-46b0-856a-0c44ff640239` |
| train21 | prev_w14 | `df359ada-816e-492d-9ca9-671d8d88edbd` |
| train21 | blend_prev14_c05504 | `b00935e4-e0ab-41ce-a8e4-65f6530d2867` |
| train22 | no_ip | `b8233e9b-689a-4ecf-aea8-5f75544098c0` |
| train22 | prev_w14 | `5d2f751e-d02b-4d26-9ab1-750c9d500022` |
| train22 | blend_prev14_c05504 | `98917d34-d982-4f19-b6f6-bf5167fad8de` |
| train23 | no_ip | `6b3c8538-63ac-4624-924b-b8a2f9eeb9aa` |
| train23 | prev_w14 | `92f9da12-5f02-403e-8896-49f68ce70950` |
| train23 | blend_prev14_c05504 | `581934d5-9d35-4d3c-8f68-305b5de3071c` |
| train24 | no_ip | `fee8bed3-5986-407d-bbaa-9705e26d6131` |
| train24 | prev_w14 | `0d58e0cd-64ee-46da-8072-a28a21a6b51e` |
| train24 | blend_prev14_c05504 | `17a652d7-f388-429c-8cac-94afe6beb2b3` |
| train25 | no_ip | `26d12754-59b1-4bb4-b6f4-14fb2b8837f3` |
| train25 | prev_w14 | `fd3d1e5f-162c-4a32-9cea-4f1d5ed052e8` |
| train25 | blend_prev14_c05504 | `f2ab6f37-d94c-4ab1-bcd4-3239a8ec7f1f` |
| train26 | no_ip | `e03b20e5-1061-4170-aa2c-a053aad18150` |
| train26 | prev_w14 | `cf875785-335c-456e-9fb8-d50c88e030d3` |
| train26 | blend_prev14_c05504 | `e99112bf-195f-46aa-a3de-2d7c5d21bc25` |
| train27 | no_ip | `9a8b60c5-b1c1-449a-a2d9-5e5e68364415` |
| train27 | prev_w14 | `91c1a946-4ac4-4685-9192-a1ab65f7418d` |
| train27 | blend_prev14_c05504 | `97db8b10-3c8e-4147-aa0c-d00af100b5f5` |
| train28 | no_ip | `cdbb9dfa-1784-420f-b35c-be7109056a7a` |
| train28 | prev_w14 | `6a1c2912-2236-4daf-a6c1-3adb75f2b409` |
| train28 | blend_prev14_c05504 | `d8f83f40-b96b-47a5-9d75-2d19bcc17f16` |
| train29 | no_ip | `c00a55f3-6618-414a-8bdd-990dbc545f9c` |
| train29 | prev_w14 | `e3e068f7-d336-4ee0-bd23-a48112b20021` |
| train29 | blend_prev14_c05504 | `537cb6cc-50e9-467b-89d3-ab41eb96906c` |
| train30 | no_ip | `a949dd32-8063-45f4-954b-89398622a2a6` |
| train30 | prev_w14 | `1905ac7b-81c0-4759-a8a5-57099e31fcca` |
| train30 | blend_prev14_c05504 | `8571cbc0-503d-492f-94e5-0588ed348a18` |
| train31 | no_ip | `bdbfce94-f5bd-4bb2-91da-f6e1bd070c55` |
| train31 | prev_w14 | `9cea9c7a-d68c-4ce9-88fc-4b3847ae2d8b` |
| train31 | blend_prev14_c05504 | `10a95549-a736-40d7-87e6-99369d095155` |
| heldout00 | no_ip | `e2e5640a-0b6f-4dda-acb7-1493f7024038` |
| heldout00 | prev_w14 | `b986a695-5379-41bc-bd20-bc2bd6d4a0db` |
| heldout00 | blend_prev14_c05504 | `ac379af3-dde3-4ebe-9806-d1f1ae680660` |
| heldout01 | no_ip | `41b58cbc-1c86-472d-b6fe-4d1838a19ec4` |
| heldout01 | prev_w14 | `4443ea0c-80a4-41ea-bad5-1e2d3232890d` |
| heldout01 | blend_prev14_c05504 | `e8c6e013-f844-4d6e-98da-012d3128f23e` |
| heldout02 | no_ip | `e3494422-2062-4d27-9e59-f6087238f746` |
| heldout02 | prev_w14 | `3ec6d77d-80a5-47bc-b52e-67f77057578b` |
| heldout02 | blend_prev14_c05504 | `2d517a15-3828-4b74-8f8e-78324f59242f` |
| heldout03 | no_ip | `afbae233-335e-4291-b60d-43970e24dab0` |
| heldout03 | prev_w14 | `6da2681d-7b3a-44a5-a804-4dd393cf43fa` |
| heldout03 | blend_prev14_c05504 | `08a7d3df-4ba0-4b30-8d4e-ae8e1e69ce2a` |
| heldout04 | no_ip | `d39755bd-c530-4d0d-8393-14838e3a26f2` |
| heldout04 | prev_w14 | `6cb74887-cd71-404a-be10-44436363a478` |
| heldout04 | blend_prev14_c05504 | `1043ac0e-25d3-4edf-b750-a245d75e8bd0` |
| heldout05 | no_ip | `28beace1-1790-40fe-8c09-2f4e1997c73d` |
| heldout05 | prev_w14 | `2aa90851-768f-4326-83c1-33c15790b8f0` |
| heldout05 | blend_prev14_c05504 | `8053efa4-d129-4410-a6f2-c0e4e635ec63` |
| heldout06 | no_ip | `ebf14d25-89d7-48e5-9e0c-87c37936980f` |
| heldout06 | prev_w14 | `579752cc-ec4a-4977-9fb4-cb9d1644f36a` |
| heldout06 | blend_prev14_c05504 | `5c9ce121-9e98-48ba-af16-3263e0aabfa5` |
| heldout07 | no_ip | `d12d0a80-80bf-405a-8627-fb2e702e1a8c` |
| heldout07 | prev_w14 | `81fcc745-173d-4db9-86ed-76193c028a43` |
| heldout07 | blend_prev14_c05504 | `91f54511-4496-44fe-bb9f-74aca583cce8` |
