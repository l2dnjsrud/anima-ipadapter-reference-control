# QwenVL c059 Checkpoint Merge Gate

- Train contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_c059_checkpoint_merge_gate_20260612/contact_sheet_train.jpg`
- Heldout contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_c059_checkpoint_merge_gate_20260612/contact_sheet_heldout.jpg`
- Previous retrieval checkpoint: `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- c055 mixed checkpoint: `anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors`
- Merge alpha 0.25 checkpoint: `anima_qwenvl_ip_adapter_c059_merge_prev_c055_a0250.safetensors`
- Merge alpha 0.40 checkpoint: `anima_qwenvl_ip_adapter_c059_merge_prev_c055_a0400.safetensors`
- Columns: reference / no_ip / prev_w14 / blend_prev14_c05504 / merge_a025_w14 / merge_a040_w14.

Decision: `single_checkpoint_merge_not_quality_pass_runtime_blend_remains_best`

## Gate Summary

- Samples: `40` (`clean32` train + heldout8)
- Variants: `5`
- Generated PNGs: `200`
- Blank outputs: `0`
- ComfyUI API surface: `AnimaQwenVLIPAdapterLoader`, `AnimaQwenVLIPAdapterApply`, and `AnimaQwenVLEncodeImage` loaded; both c059 merged checkpoints appeared in the loader model selection list.
- Runtime cleanup: isolated ComfyUI server was stopped and port `8116` was closed after generation.

## Metric Summary

| encoder metric | variant | mean cosine | mean uplift vs no-IP | improved rate |
| --- | --- | ---: | ---: | ---: |
| PE | `blend_prev14_c05504` | `0.803729` | `+0.049596` | `0.725` |
| PE | `prev_w14` | `0.783374` | `+0.029240` | `0.750` |
| PE | `merge_a025_w14` | `0.779777` | `+0.025643` | `0.600` |
| PE | `merge_a040_w14` | `0.779970` | `+0.025837` | `0.475` |
| QwenVL | `merge_a040_w14` | `0.822691` | `+0.041614` | `0.800` |
| QwenVL | `blend_prev14_c05504` | `0.822666` | `+0.041589` | `0.800` |
| QwenVL | `merge_a025_w14` | `0.819944` | `+0.038867` | `0.800` |
| QwenVL | `prev_w14` | `0.817265` | `+0.036187` | `0.725` |

`merge_a040_w14` ties the runtime blend in the QwenVL metric, but it does not reproduce the blend in PE similarity and its PE improved rate drops to `0.475`. This means a simple parameter-space merge is not a reliable distillation of `prev_w14 + c055_w04`.

## Visual Audit Summary

See `visual_audit.md` and `visual_audit.json`.

The merged checkpoints are usable and often preserve broad color/costume/face cues, but they do not solve the c058 failure classes. The heldout sheet still shows weak exact pose/crop control, speech bubble loss, hand/fan prop drift, and non-human silhouette collapse. `merge_a040_w14` can look cleaner on several human samples, but `heldout06` and `heldout07` remain structurally unreliable.

## Decision

c059 does not promote either merged checkpoint to the high-quality reference-control path. The runtime blend `blend_prev14_c05504` remains the best reference recipe from c058/c059, while `merge_a040_w14` is only a diagnostic: it proves parameter-space interpolation can make a loadable QwenVL adapter, but not a production-quality single checkpoint.

Next loop should stop simple checkpoint merging and move to failure-focused continuation or stronger encoder/feature adaptation with explicit training pressure on pose/crop, speech bubbles, props, and non-human silhouettes.

| sample | variant | prompt_id |
| --- | --- | --- |
| train00 | no_ip | `304f0142-9e21-4a6a-9d72-4e10eff44b2e` |
| train00 | prev_w14 | `64efdfe1-bfd7-4ca3-84f8-510841feb15f` |
| train00 | blend_prev14_c05504 | `379b2a37-a647-4fd3-a8e5-326a9ed40c72` |
| train00 | merge_a025_w14 | `9d433fa9-9443-408b-839e-b317d01c9453` |
| train00 | merge_a040_w14 | `f9d17d14-906d-4d11-80af-3984ff6963d9` |
| train01 | no_ip | `a45617da-f98f-43f8-beef-f692b8c365bc` |
| train01 | prev_w14 | `66afff38-b419-4756-b59c-9a4beef9b0a5` |
| train01 | blend_prev14_c05504 | `3af1a866-e2d0-4f7c-b105-ec510178b7ac` |
| train01 | merge_a025_w14 | `9e4124e7-df5c-4afa-beef-c7b30f8aee10` |
| train01 | merge_a040_w14 | `953149ea-b7b3-4515-8f65-cd5d2ae40ce6` |
| train02 | no_ip | `af1960d5-4a15-4213-a887-84366ef0041a` |
| train02 | prev_w14 | `d5db460f-5207-4239-95ea-256e10c916a5` |
| train02 | blend_prev14_c05504 | `5abea909-10c2-47e5-a3ae-a88328794b6f` |
| train02 | merge_a025_w14 | `46a160d6-f598-4f00-9892-24178012aa98` |
| train02 | merge_a040_w14 | `4c4b2484-e7d9-4e37-b2a0-892394395a2e` |
| train03 | no_ip | `9fff1abf-bb7a-4730-8251-569b1590a183` |
| train03 | prev_w14 | `9d68d66a-bf8a-445d-86b8-e613a7540121` |
| train03 | blend_prev14_c05504 | `19f14f76-3c7e-463c-a04b-8f50b2f6d868` |
| train03 | merge_a025_w14 | `d53589ab-937e-4292-af74-6abdc9dfa526` |
| train03 | merge_a040_w14 | `c3179966-fb16-46d9-9858-12589737523e` |
| train04 | no_ip | `1f01a6b8-0289-401f-886a-3de6aa449203` |
| train04 | prev_w14 | `10c7ed8d-856a-4ede-96f8-14325c834b2f` |
| train04 | blend_prev14_c05504 | `5a8ad7bd-23f0-42df-9afd-056f7eff1105` |
| train04 | merge_a025_w14 | `05e5c87f-a772-4f76-b1ba-21a6744bcd83` |
| train04 | merge_a040_w14 | `18e6304f-a1cd-413b-b31d-a8c33bb18311` |
| train05 | no_ip | `de9a5e00-a0e3-4576-b8c0-fee6b8aec095` |
| train05 | prev_w14 | `6079d5cf-10b5-4cae-942a-a912eed6e3f3` |
| train05 | blend_prev14_c05504 | `66c99853-71f2-445a-85ad-b8ae797ad13d` |
| train05 | merge_a025_w14 | `c719390e-41c7-4f9d-95af-3b3a03de49e6` |
| train05 | merge_a040_w14 | `459a098c-7933-4200-8a19-9daf8a87c3aa` |
| train06 | no_ip | `8bd089ba-5351-4788-9695-6464de309787` |
| train06 | prev_w14 | `236317bc-5cc2-4652-8818-6a82736143d1` |
| train06 | blend_prev14_c05504 | `3cb46800-2a11-4e60-be79-a0427053ab52` |
| train06 | merge_a025_w14 | `294b1997-acf8-405c-9569-b78652fe9b1c` |
| train06 | merge_a040_w14 | `3fa5264c-3856-483e-8ebe-fdfdb8bb848e` |
| train07 | no_ip | `70bb84d2-c758-498e-8795-8e94bbe0644b` |
| train07 | prev_w14 | `c1caa3a5-f981-44b3-966f-1bb652c96eb3` |
| train07 | blend_prev14_c05504 | `4624e4af-f1f0-4348-bdf7-a2b0ae7f6694` |
| train07 | merge_a025_w14 | `4891139e-31fa-4673-b1cf-020ce0d16996` |
| train07 | merge_a040_w14 | `e268e654-d786-4bdf-a107-9fbe6c0af1ed` |
| train08 | no_ip | `bde152e1-36d9-40c0-9e5a-00ab02f0db14` |
| train08 | prev_w14 | `3c55fa4b-9fa8-44c6-9257-a866f03bac70` |
| train08 | blend_prev14_c05504 | `c3aad731-28b1-4a86-b9a3-fcd2adaba59d` |
| train08 | merge_a025_w14 | `a6de6421-6d89-4abc-8635-34aee2b86960` |
| train08 | merge_a040_w14 | `e1c5bc80-7019-42a7-a004-40fc062e9b70` |
| train09 | no_ip | `b6cf43db-c4a2-4e26-8691-2ad412ebdc50` |
| train09 | prev_w14 | `257ba2eb-fa6e-444e-810b-04a44257adc3` |
| train09 | blend_prev14_c05504 | `9c280c4d-09a3-4ba8-9de9-ad1c9a6d498a` |
| train09 | merge_a025_w14 | `5d84de2e-56e0-43ae-87e4-a514ae6adbed` |
| train09 | merge_a040_w14 | `f383de21-24a7-4495-a4dc-c67d5e72cce9` |
| train10 | no_ip | `e9b6d4e4-5fad-47d2-a538-0414700fc60c` |
| train10 | prev_w14 | `5d2470e3-c6e1-4dc1-8106-903ba810d649` |
| train10 | blend_prev14_c05504 | `6e0eb922-b35f-4787-a1f2-3b1b7ff8228f` |
| train10 | merge_a025_w14 | `a9dea2a2-2969-40cb-a764-8efc19b5a86a` |
| train10 | merge_a040_w14 | `0b5a6c2a-c462-40e4-9c8d-2ffe94caad7c` |
| train11 | no_ip | `38d84cb5-4049-40a8-9c2e-b68077e0fadb` |
| train11 | prev_w14 | `bb0161f0-7373-4c7d-8cc3-711b41d056cb` |
| train11 | blend_prev14_c05504 | `e6f8d2f1-5a5d-49d4-9848-9ce93e968213` |
| train11 | merge_a025_w14 | `1a679747-9cf5-4295-ac8d-51c4f211f3de` |
| train11 | merge_a040_w14 | `5aaad7e3-59fa-4731-9802-4ab84b0eec9e` |
| train12 | no_ip | `5276afb0-145c-4e7e-b8e9-5ba1f645d166` |
| train12 | prev_w14 | `5c1a9f1c-2f44-4005-ad5a-e0b16cef7a4b` |
| train12 | blend_prev14_c05504 | `bef1a5bd-c715-4dc2-89fc-8cfa03e2f63b` |
| train12 | merge_a025_w14 | `28d3496d-0824-4bed-a58e-c80fb6365060` |
| train12 | merge_a040_w14 | `5c59e4d8-e417-41ef-b6d6-373b35ddd46e` |
| train13 | no_ip | `48e02de2-772e-4b8f-8cdf-1e554fa23798` |
| train13 | prev_w14 | `49478595-5b8a-4484-8462-a3e29ba2b279` |
| train13 | blend_prev14_c05504 | `7ea889ce-9c94-4b50-9408-7f06745a20a0` |
| train13 | merge_a025_w14 | `1e7bcd54-d079-486b-b0c2-37b090b2589d` |
| train13 | merge_a040_w14 | `379b947d-590c-429f-86f6-ed1b4b16a232` |
| train14 | no_ip | `f64564cf-32d1-4c9e-8e7f-1b219e513646` |
| train14 | prev_w14 | `9ecd5b5c-60a1-4d8a-9869-247948213322` |
| train14 | blend_prev14_c05504 | `15d7b142-b057-4acf-a023-d933fa9834a3` |
| train14 | merge_a025_w14 | `8e109789-6c8f-4593-9bd5-12587751e862` |
| train14 | merge_a040_w14 | `78214231-11a1-4833-8b65-d37c533e1bbf` |
| train15 | no_ip | `f2d1579e-66fd-4acc-9195-8967bbe84301` |
| train15 | prev_w14 | `6bf80608-8004-41ba-bfcb-4275fdad4baa` |
| train15 | blend_prev14_c05504 | `3dbdead1-9bbd-4d46-9588-fbb08ba94e13` |
| train15 | merge_a025_w14 | `11a30e0b-2d4d-44e2-9556-67d457d694b0` |
| train15 | merge_a040_w14 | `84af30f9-f414-4f0c-a1ff-fb70a46bd5cc` |
| train16 | no_ip | `6beda1e3-856c-420c-af1e-7947fe1c2d31` |
| train16 | prev_w14 | `74e5a575-c1a8-43ae-92aa-967f0fa95899` |
| train16 | blend_prev14_c05504 | `f4f52a9b-23af-40a2-a1f8-47014f5f5281` |
| train16 | merge_a025_w14 | `7d55cd19-0bc4-4442-b9c3-53dcfb05d428` |
| train16 | merge_a040_w14 | `0b78fafe-392d-43da-95d5-19c08a2c7997` |
| train17 | no_ip | `76d44cd4-a272-4591-a55a-21acc31120b2` |
| train17 | prev_w14 | `d13ec4ea-aa1f-4541-9290-e05fb23a3f0a` |
| train17 | blend_prev14_c05504 | `242f2a5d-f3b6-440a-8bfd-4680cd96bd68` |
| train17 | merge_a025_w14 | `682e9700-f8d0-4653-b83e-90af266f1384` |
| train17 | merge_a040_w14 | `e2ed7102-6339-4bdb-a750-64076c95d046` |
| train18 | no_ip | `81d1016e-6e22-4f87-a022-9561471c8fdc` |
| train18 | prev_w14 | `5838c582-e8ff-437c-8cae-9cd174feed22` |
| train18 | blend_prev14_c05504 | `4c05d594-b81a-4a74-bab7-4ce325f97a39` |
| train18 | merge_a025_w14 | `f544d667-843a-4a87-a529-1bcce9d1c4dd` |
| train18 | merge_a040_w14 | `ee81d707-0ca7-4568-92ca-7b951a54164f` |
| train19 | no_ip | `de4739ce-6777-4190-baa7-5394654b6bc9` |
| train19 | prev_w14 | `851ffebd-dd49-4c32-973a-87bcb5a212a2` |
| train19 | blend_prev14_c05504 | `a1a81b23-dc89-4e12-b18b-ed8d4e6da1a9` |
| train19 | merge_a025_w14 | `286147ca-061b-4c86-abde-07731a564fd8` |
| train19 | merge_a040_w14 | `3055045a-7c54-4f24-84ec-a622743c0a64` |
| train20 | no_ip | `1aa00755-6ba8-4bf1-8aa5-5005f6d27a89` |
| train20 | prev_w14 | `9decc412-34d2-4678-a8af-5bb7beb39aba` |
| train20 | blend_prev14_c05504 | `3ada98b3-3715-4ae3-88e7-da569ea77e7c` |
| train20 | merge_a025_w14 | `627956b8-7440-4142-b1ee-6e93472d3acc` |
| train20 | merge_a040_w14 | `8169f380-daab-4f77-9e03-507e2aa47e81` |
| train21 | no_ip | `b6d8b9f7-6cf6-4c01-b8fe-fad592ecd4b5` |
| train21 | prev_w14 | `8907ddab-67df-4086-a9ac-879f5ca1a004` |
| train21 | blend_prev14_c05504 | `d89696a2-a57f-4dfc-8ff3-73e91ed2e038` |
| train21 | merge_a025_w14 | `a5de6640-32ff-4de9-832c-41c44d09509f` |
| train21 | merge_a040_w14 | `31b616d8-642f-407c-a900-509a04a5570a` |
| train22 | no_ip | `a0538071-5de1-425f-af87-4030139b4c7b` |
| train22 | prev_w14 | `e03ad6c2-ea2e-482f-bc74-5213cb658588` |
| train22 | blend_prev14_c05504 | `ad81c495-38d8-461f-bb11-7724aaa9de6b` |
| train22 | merge_a025_w14 | `df9ea5cb-1f48-4ce0-b3bb-b8f6ab5753f0` |
| train22 | merge_a040_w14 | `16d60ef6-e096-446c-a62d-d369994b43ae` |
| train23 | no_ip | `6c87da36-945f-477d-b55c-ca85a6b6901a` |
| train23 | prev_w14 | `d238d3c6-525c-4384-a8a2-c9c765779fc7` |
| train23 | blend_prev14_c05504 | `bde63c17-67b0-4d60-8a35-222162a2f62f` |
| train23 | merge_a025_w14 | `ecdb0e9a-0bb2-4938-9082-562f09a11f95` |
| train23 | merge_a040_w14 | `5f61cfaf-dc2d-4440-8760-0d90718a27f2` |
| train24 | no_ip | `755d7572-6b4c-4755-a106-77a8527be92b` |
| train24 | prev_w14 | `d74acaeb-d5ee-4664-b2b1-84f1c26f35cc` |
| train24 | blend_prev14_c05504 | `df54c3bd-e2ce-4e67-a3b5-b680f6902006` |
| train24 | merge_a025_w14 | `43ac9234-1bba-43ef-81ee-a34985e8a16e` |
| train24 | merge_a040_w14 | `11674f63-f944-4539-a9a5-aebbf82958d8` |
| train25 | no_ip | `a7dda3b9-ea1d-4797-848f-9ba39c44f3f1` |
| train25 | prev_w14 | `d257d40a-aff6-41dc-957b-4cc1799f8e6f` |
| train25 | blend_prev14_c05504 | `48573e2f-0da8-4a09-a824-bd14d0f03334` |
| train25 | merge_a025_w14 | `399a7499-9ceb-49e0-9536-99a6e8d22f66` |
| train25 | merge_a040_w14 | `c9027974-573f-4da3-9129-c66f1caae63b` |
| train26 | no_ip | `446bdd30-6f19-4de7-acf5-d71fdb78164e` |
| train26 | prev_w14 | `d7923467-810b-49ef-92c9-80f0f8f3ce52` |
| train26 | blend_prev14_c05504 | `31600672-c802-4518-a61f-0df54c3bb260` |
| train26 | merge_a025_w14 | `b9672f3e-31a5-45b9-b6d4-9a1ccd54aaa9` |
| train26 | merge_a040_w14 | `77a524be-129d-4bda-a596-7d4dc8c201e8` |
| train27 | no_ip | `ef657871-43bd-4c9a-b221-c8b25371cd1b` |
| train27 | prev_w14 | `5956726a-756a-46b6-8881-ccddb6e37b05` |
| train27 | blend_prev14_c05504 | `d782630d-79a5-46a1-a15c-d6afc9a53530` |
| train27 | merge_a025_w14 | `fecd55a8-f51f-477e-8210-75c4aae438cc` |
| train27 | merge_a040_w14 | `0b3bcf6a-3de4-4912-8d92-e4b5256d0655` |
| train28 | no_ip | `0d43575b-0fbf-403d-b58f-dead2652cbe2` |
| train28 | prev_w14 | `4fc3080c-4630-4251-9b86-fb877e83ae5f` |
| train28 | blend_prev14_c05504 | `3d5af271-6c9d-4e81-9593-57dff33d40d1` |
| train28 | merge_a025_w14 | `c1794ba4-34b8-4a92-9664-2ea5dd4420df` |
| train28 | merge_a040_w14 | `33457d81-deea-4482-83e4-c18816021cea` |
| train29 | no_ip | `1b4cd9d7-7416-4b8e-9413-6c28f8aa4788` |
| train29 | prev_w14 | `bf315590-e83e-4ef0-b83a-f67dcc280f57` |
| train29 | blend_prev14_c05504 | `66f9da2b-6507-48e6-a7c8-14e794955f54` |
| train29 | merge_a025_w14 | `dbb7ad85-f73f-4ff4-a88b-6bcce7e6a855` |
| train29 | merge_a040_w14 | `01dad7be-3477-448e-a2aa-6fa11b8c07ae` |
| train30 | no_ip | `c1d18ff8-1046-4c31-9d82-19930df34633` |
| train30 | prev_w14 | `536f21c7-9694-44e3-9ccd-afdc2febbfa4` |
| train30 | blend_prev14_c05504 | `4a8f4b4f-72cb-4263-bb66-b320490b36d1` |
| train30 | merge_a025_w14 | `cb10e15d-cf5a-4184-ade9-064b93dd6a14` |
| train30 | merge_a040_w14 | `c0cab03c-bcc4-4766-b427-6d9710f7e6bd` |
| train31 | no_ip | `25fcd672-7fa3-4e20-96de-a07fabfd8704` |
| train31 | prev_w14 | `85f7f18f-8fe6-417f-9748-64631e7d7b88` |
| train31 | blend_prev14_c05504 | `fc35322d-b5aa-45ea-bf1a-c0e13b1f1524` |
| train31 | merge_a025_w14 | `e47f066b-3eed-438f-b78d-82a9a29c41c7` |
| train31 | merge_a040_w14 | `f4b23698-e031-4423-823f-2a4859560177` |
| heldout00 | no_ip | `19aaa70a-5995-4b0d-8afa-a29cbc592640` |
| heldout00 | prev_w14 | `4cc7ac9e-a139-4abc-ab56-daf8b4f52f97` |
| heldout00 | blend_prev14_c05504 | `e9db56f2-061b-4336-9e74-d4bcd4c948e4` |
| heldout00 | merge_a025_w14 | `03c3849b-702c-40e2-8745-d78a6f706e0e` |
| heldout00 | merge_a040_w14 | `18f503ef-218f-4338-bd67-05fc55db9934` |
| heldout01 | no_ip | `d02f8a9f-02ec-41ce-b03f-85477e4435d9` |
| heldout01 | prev_w14 | `bddcdedc-b651-475c-b494-5640759be89b` |
| heldout01 | blend_prev14_c05504 | `dcee697a-4a52-4495-acc9-4bcbb0abd673` |
| heldout01 | merge_a025_w14 | `c4c78496-7874-4a7b-9d89-9397461bc197` |
| heldout01 | merge_a040_w14 | `8d78ea08-9863-4e6b-bc23-68dd1771c0de` |
| heldout02 | no_ip | `47dd0f0f-f137-43c9-8b1c-ee55b004d24a` |
| heldout02 | prev_w14 | `379ef34a-ffe7-43d1-9f02-4fcb05f92ea3` |
| heldout02 | blend_prev14_c05504 | `2b5dcd35-75ba-4714-ba34-6275605daa14` |
| heldout02 | merge_a025_w14 | `0bffddd9-3ce7-420c-8752-dac3ea63f685` |
| heldout02 | merge_a040_w14 | `4e066c85-cabd-4678-bf31-6e0372744481` |
| heldout03 | no_ip | `1e2ab781-46a9-4ad5-b391-0974fa82cdec` |
| heldout03 | prev_w14 | `07109a91-1883-466f-8dda-575baeaf1bc6` |
| heldout03 | blend_prev14_c05504 | `66bebbb8-6c3e-42e0-a1ac-cd895f15399c` |
| heldout03 | merge_a025_w14 | `b6988eff-a2b5-4fdc-be3b-018eac73ed99` |
| heldout03 | merge_a040_w14 | `660c4688-88de-4a81-8724-1968ca84b1e9` |
| heldout04 | no_ip | `bfecf809-ba4b-4327-bd1e-154528eaac9c` |
| heldout04 | prev_w14 | `c69b6b94-66d0-44c0-bbf7-dfc6fcfb8e2e` |
| heldout04 | blend_prev14_c05504 | `98da3c98-d97a-448a-b806-43f55409f0ce` |
| heldout04 | merge_a025_w14 | `4436f45c-8e42-4eb9-b586-e8c9de27c7a8` |
| heldout04 | merge_a040_w14 | `60ff554c-a3d0-4dda-995d-80a33fe25dbc` |
| heldout05 | no_ip | `3619bb44-7cbc-4fba-9563-5b33e3e536cd` |
| heldout05 | prev_w14 | `91e05730-6245-4f71-8d7f-148b692dc3fc` |
| heldout05 | blend_prev14_c05504 | `418224c9-1484-4e17-945e-3590d9111e6f` |
| heldout05 | merge_a025_w14 | `60943f68-30a4-4948-8b76-6096110f3b12` |
| heldout05 | merge_a040_w14 | `0f9312c6-28c7-4923-be39-bd5271956924` |
| heldout06 | no_ip | `9b344654-2acb-4c94-ba0c-b5e83f6a3a75` |
| heldout06 | prev_w14 | `7d73bc04-2ab6-434a-8e61-bf9964c05ed7` |
| heldout06 | blend_prev14_c05504 | `824bd211-b089-4e5f-867f-5fa6f4f096a2` |
| heldout06 | merge_a025_w14 | `190329f6-e18a-43d8-996f-f6706fd4324a` |
| heldout06 | merge_a040_w14 | `57b657a1-a1f2-402e-93d8-a22021452548` |
| heldout07 | no_ip | `31dc36fa-1133-497c-b5c4-04b060a6d193` |
| heldout07 | prev_w14 | `e720c405-98f8-41f4-99e5-1e1328e5b9b3` |
| heldout07 | blend_prev14_c05504 | `12d7ca03-9b80-43e8-94b4-2e5c9c05c9c2` |
| heldout07 | merge_a025_w14 | `ba7808c8-9b9a-4d97-b8c9-d9122e3381cb` |
| heldout07 | merge_a040_w14 | `b4e91b4d-e53b-4f0c-9c08-aecb9b387b0d` |
