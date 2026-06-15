# QwenVL c061 Instruction Calibration Gate

- Train contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_c061_instruction_calibration_gate_20260612/contact_sheet_train.jpg`
- Heldout contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_c061_instruction_calibration_gate_20260612/contact_sheet_heldout.jpg`
- Previous retrieval checkpoint: `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- c055 mixed checkpoint: `anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors`
- Runtime recipe: previous retrieval `1.4` + c055 mixed `0.4`.
- Columns: reference / no_ip / blend_default / blend_identity_exact / blend_species_face.

Instruction variants:

- `no_ip`: no adapter, instruction unused.
- `blend_default`: Represent this manhwa/anime reference image for visual style, color palette, composition, character identity, and panel layout.
- `blend_identity_exact`: Represent the exact same manhwa/anime character identity for image reference control. Emphasize face shape, facial silhouette, age, species or non-human traits, skin color, eye color, hair, beard, hat, costume, palette, expression, camera angle, pose crop, props, speech bubbles, and drawing style. Preserve distinctive traits over generic beauty.
- `blend_species_face`: Represent this reference for strict visual identity retrieval in a manhwa panel. Prioritize non-human species, monster or demon traits, facial structure, profile silhouette, beard and headwear, skin tone, glowing eyes, hand props, fan or weapon cues, speech bubble context, pose crop, costume palette, and line/color style.

Decision: `instruction_calibration_species_face_best_preset_not_quality_pass`

Metric summary:

- PE mean uplift: `blend_species_face=0.060893`, `blend_identity_exact=0.054909`, `blend_default=0.049596`.
- QwenVL mean uplift: `blend_species_face=0.042190`, `blend_default=0.041589`, `blend_identity_exact=0.039557`.
- Heldout PE mean uplift: `blend_species_face=0.053534`, `blend_default=0.039142`, `blend_identity_exact=0.035494`.
- Heldout QwenVL mean uplift: `blend_species_face=0.026471`, `blend_default=0.022779`, `blend_identity_exact=0.016944`.
- Prompt guard: all 40 samples preserved the same seed, positive/negative prompt, checkpoint sequence, weights `1.4 + 0.4`, start/end range, and sample set; only `AnimaQwenVLEncodeImage.instruction` changed across calibrated adapter variants.
- Visual audit: `blend_species_face` is the best c061 preset candidate, but it does not pass the high-quality reference-control gate. The main non-human/profile failure remains visible on `heldout07`, and most adapter columns still collapse toward the same dark-villain template.

See `visual_audit.md` and `visual_audit.json`.

| sample | variant | prompt_id |
| --- | --- | --- |
| train00 | no_ip | `ee2d246c-d500-41af-975b-a5ce6e1fc9d1` |
| train00 | blend_default | `913c9ceb-6234-435e-baf5-eba487e39009` |
| train00 | blend_identity_exact | `63773204-2143-4758-8a4c-0c1a1685ec50` |
| train00 | blend_species_face | `51295b2e-bc6c-4b71-b8d7-d6e64c87ed8b` |
| train01 | no_ip | `8608e953-56c3-499c-b2fb-200be1587744` |
| train01 | blend_default | `69d81c65-9f8b-45cd-a049-2c5b7953bcbe` |
| train01 | blend_identity_exact | `ef3291c9-a990-48e7-bbf6-34ca8e175f89` |
| train01 | blend_species_face | `626098a4-11ec-4334-a94c-36a04eb90ba6` |
| train02 | no_ip | `9de74196-934f-4c3d-9aae-648c106defcc` |
| train02 | blend_default | `a7b7d3ff-3887-40bc-b319-a7ede90e233c` |
| train02 | blend_identity_exact | `3dc13743-d3ac-4474-be9f-e5d42c6987f5` |
| train02 | blend_species_face | `96f28b5a-4a3a-4108-baa0-ba702f3be07d` |
| train03 | no_ip | `6563a9f2-ec5d-4fe2-af81-0db205f5e04b` |
| train03 | blend_default | `63a2f179-4a0f-4331-bf89-9420507682dc` |
| train03 | blend_identity_exact | `34e8c614-95ea-42ea-a8cf-58b979c34602` |
| train03 | blend_species_face | `1d6d7aea-d283-4e98-aa81-ce0c1a9aef11` |
| train04 | no_ip | `eadf69c2-db84-47ba-8880-70cfe072b46f` |
| train04 | blend_default | `afe25dd8-822d-440f-961e-66e5555aae80` |
| train04 | blend_identity_exact | `137a1372-0e43-4908-a1b4-0a4d8c63bb0a` |
| train04 | blend_species_face | `d59e7e1d-6f56-4388-95c0-362b8aa426f6` |
| train05 | no_ip | `8734a92f-920f-43ea-8cd1-c86ed5498ea3` |
| train05 | blend_default | `0b199f22-15ee-4232-b676-92be9c091369` |
| train05 | blend_identity_exact | `12683390-badb-45e0-a46e-240e679ca0ed` |
| train05 | blend_species_face | `216cdb9a-842d-4d69-b005-af43b6f01e04` |
| train06 | no_ip | `3562886a-d48a-455e-b8a3-1f6a8aa91210` |
| train06 | blend_default | `1079387b-f35c-404a-ab0a-3929db354f0b` |
| train06 | blend_identity_exact | `8d3cc2dd-6c86-4832-a29e-9e6f31ef11a4` |
| train06 | blend_species_face | `3800bbd8-f9ab-407f-a69e-f9a56b80fc06` |
| train07 | no_ip | `7b2790f6-bfc4-4a19-afa3-8a1359de6208` |
| train07 | blend_default | `9a92bdbc-f111-4f19-b2ce-012a8656e6c1` |
| train07 | blend_identity_exact | `0cb11b07-d38e-464a-aca0-d29e46eb4c17` |
| train07 | blend_species_face | `12f2a155-6316-4e38-8a2e-6e27d9840a81` |
| train08 | no_ip | `33f405b9-6046-4cf6-aad6-e8ce0a9273d3` |
| train08 | blend_default | `645d5817-ab87-40a9-8fa2-bd6ea2442b01` |
| train08 | blend_identity_exact | `74cfafa1-c72b-4a1b-89c8-a779206a57d0` |
| train08 | blend_species_face | `5dc928f8-7cbe-4d3e-8865-602c12954cb7` |
| train09 | no_ip | `2077e3be-2a32-46ca-9f12-33dfee7f2d97` |
| train09 | blend_default | `7baadee8-b82b-40b1-968e-9b1f297164da` |
| train09 | blend_identity_exact | `e2f87e4d-0e29-4ac2-8ad6-01e770d740b1` |
| train09 | blend_species_face | `9da7da6f-8c79-4310-902d-425c30e93fd4` |
| train10 | no_ip | `26f7fbbd-05db-4047-a575-7f9c89847005` |
| train10 | blend_default | `9e29dc64-41f1-4121-b337-51fcac8d2c78` |
| train10 | blend_identity_exact | `d5afcf37-8dac-4765-905f-1129b0086f4e` |
| train10 | blend_species_face | `47a3ebb7-ea91-4efc-88c5-68013c5f93da` |
| train11 | no_ip | `b69a6e02-09b8-44e0-b4ac-b5620aeace25` |
| train11 | blend_default | `874cdf9c-7b6f-4aeb-8962-9cc198d74172` |
| train11 | blend_identity_exact | `38fbe98c-7d6f-46ba-b69b-dadd00fb265a` |
| train11 | blend_species_face | `27a17f67-121d-4dee-94bd-54e7f28e72e4` |
| train12 | no_ip | `3307113e-593d-407d-b305-155ecbccf80f` |
| train12 | blend_default | `e7264fb8-74d7-493c-bc0e-b526739fcf5a` |
| train12 | blend_identity_exact | `fb13d1c5-d551-4f03-81ff-f23001433b1c` |
| train12 | blend_species_face | `42fbd761-0cf5-4817-9e5a-6d662b342103` |
| train13 | no_ip | `bc4fa591-fa4c-4127-a7cc-847ebd6808d0` |
| train13 | blend_default | `816f4a7e-dd8d-410c-b8cc-c7c6b76a3cca` |
| train13 | blend_identity_exact | `ac76d3ea-06a4-4df7-b5f0-4dea74895f3e` |
| train13 | blend_species_face | `1f5b486e-0acd-40a3-b6bd-9446458be845` |
| train14 | no_ip | `037c1466-5d44-48f1-8870-c991c85fd81a` |
| train14 | blend_default | `095600d1-c167-4fb5-a4b5-7005087effa6` |
| train14 | blend_identity_exact | `4913752b-e213-4afa-a0fc-a63a6f327804` |
| train14 | blend_species_face | `9b14fb49-9f12-40d5-8bd7-a0c8bb2c7be8` |
| train15 | no_ip | `2f6d7ace-0ef0-4363-bc1e-83164e0477b4` |
| train15 | blend_default | `4b9cf45f-e21f-4065-b336-2ff84fbecf61` |
| train15 | blend_identity_exact | `cc023e24-8bcb-4f59-93b6-c4ffeae4e511` |
| train15 | blend_species_face | `895e039e-2032-4e4d-b9e8-9c958812c417` |
| train16 | no_ip | `9120187a-c9b8-4493-83fa-766ddf0399e7` |
| train16 | blend_default | `e40802a0-1fa0-43b2-9552-ea78fdd606dc` |
| train16 | blend_identity_exact | `149480da-5454-45e7-a076-52c83a06d55c` |
| train16 | blend_species_face | `44574cc7-4c05-4e4c-8fae-2fd095d7cb46` |
| train17 | no_ip | `749f86fd-dcaa-4d42-aa35-4208ebc6a0ec` |
| train17 | blend_default | `161840d8-bf75-44e0-a253-9df9028f1146` |
| train17 | blend_identity_exact | `81f47160-c55c-43b9-8731-07539faaef03` |
| train17 | blend_species_face | `fec0cd20-2d9c-44dc-8ccc-5d5cb244ef4d` |
| train18 | no_ip | `0b244abc-6581-49e4-a38d-64c1c51e3f92` |
| train18 | blend_default | `72658a53-1de0-42ed-b17a-92ff9a07f779` |
| train18 | blend_identity_exact | `c8ddf967-8eb2-4067-bb95-b20341265757` |
| train18 | blend_species_face | `26754496-043e-4295-8ca9-9a9ad000948f` |
| train19 | no_ip | `9cbce7a0-0320-40f7-b059-1a038b8e2eb5` |
| train19 | blend_default | `5790e44b-f3e2-407f-b2d8-b2868aaa07e2` |
| train19 | blend_identity_exact | `151dec63-215d-4c03-9732-488ba20ddf1f` |
| train19 | blend_species_face | `5a230042-7e65-48f0-84bd-e081aa1842f4` |
| train20 | no_ip | `694456e3-87f6-44a7-a67a-e8d68d41edc4` |
| train20 | blend_default | `25887f27-b8a7-4c2b-9762-e30f0025493c` |
| train20 | blend_identity_exact | `b3f3faad-4c7a-4f5c-aaa9-163a9a062ffa` |
| train20 | blend_species_face | `48bd8337-4ce2-4ed7-a510-6b2a80c810d2` |
| train21 | no_ip | `60f2ec21-f4ba-4cc6-988c-49b54010cfde` |
| train21 | blend_default | `c87c4963-0fff-401c-b5d2-cc8d7fd64f72` |
| train21 | blend_identity_exact | `54e83b7e-dcfb-4576-bacf-8a99626f408a` |
| train21 | blend_species_face | `c8a1f4e5-1abb-4179-94e2-9c7262152485` |
| train22 | no_ip | `2e7a4500-84af-488f-a61a-d6c935e62a4b` |
| train22 | blend_default | `6fd7ded8-26db-45ba-a58b-a792b414cf51` |
| train22 | blend_identity_exact | `0ceeb904-95f9-41d8-a0b4-3e7c1fc2f895` |
| train22 | blend_species_face | `333fca58-bafa-4f96-9115-7ddf0b52c593` |
| train23 | no_ip | `a7607447-f359-4f1e-92a1-f5f551570dee` |
| train23 | blend_default | `e2d87124-79be-4f36-8d32-be14fb05f14c` |
| train23 | blend_identity_exact | `9ffe7ba1-fc37-4611-aa01-798bf4d2d954` |
| train23 | blend_species_face | `9110a95a-3763-4632-a08b-dc776d3bf297` |
| train24 | no_ip | `1b65da5b-7899-4821-8fb7-832811a159e4` |
| train24 | blend_default | `cefdcecc-5c9c-454d-bf51-44e34dfb8dca` |
| train24 | blend_identity_exact | `185b33f3-8d57-441f-ab54-db80f504738e` |
| train24 | blend_species_face | `8cc75933-e829-4b6a-97f8-c87ca1062c49` |
| train25 | no_ip | `63d06924-50cc-460b-a416-827587fa74d7` |
| train25 | blend_default | `0c5618ac-e82d-4898-8f1c-574889a0778a` |
| train25 | blend_identity_exact | `2be342f2-dabe-4b9c-a310-2e04496fa539` |
| train25 | blend_species_face | `c3983bcf-9c4d-46b2-bd75-eda67598eb0c` |
| train26 | no_ip | `6bfd592f-26bb-45f5-8e50-c1cb9f13de97` |
| train26 | blend_default | `9b7858f0-373d-4e38-be58-13df74147ae3` |
| train26 | blend_identity_exact | `017e6102-067d-4a89-a61a-86b99517995a` |
| train26 | blend_species_face | `2a1e01f0-3b17-4785-beb2-2333c6c30d34` |
| train27 | no_ip | `426db1a6-5fe2-4dbe-b81a-494859a238bd` |
| train27 | blend_default | `0e61a792-17c4-4098-bc26-a5f83a078e54` |
| train27 | blend_identity_exact | `6ea570e3-fe16-4827-9389-f3da4724ff31` |
| train27 | blend_species_face | `277bd554-b989-4553-8015-1b8ef6f62bbd` |
| train28 | no_ip | `e07480e4-2e23-4bc0-b071-1ca2e9b6f027` |
| train28 | blend_default | `7ebb59c4-96ab-495e-ac69-4f8691ab3514` |
| train28 | blend_identity_exact | `9ac9455f-2b13-4944-99be-6a5b79b286e9` |
| train28 | blend_species_face | `4026cf2b-9ccd-45d1-9c8a-18c19aa87de6` |
| train29 | no_ip | `b6e3f711-b852-4ab7-a22d-ecebaaa7cd19` |
| train29 | blend_default | `b0581045-fdd8-42ff-a1fb-ee9264d51082` |
| train29 | blend_identity_exact | `7b1ce211-ef93-4f76-bcf8-3b7a8cbc02c9` |
| train29 | blend_species_face | `394b7988-6b55-49a3-a04d-3624dfba711f` |
| train30 | no_ip | `12d90b4b-97c8-4960-a720-44350c899671` |
| train30 | blend_default | `5cb18c24-1bd3-40f6-8456-b906e61f9224` |
| train30 | blend_identity_exact | `f2e4c321-b1f3-4025-b5a7-a4bdae5e08ea` |
| train30 | blend_species_face | `c0fd4969-988c-48cf-9dcf-31647f25e4a4` |
| train31 | no_ip | `4283b77b-81f4-41ce-84a7-4bbc6f5499ea` |
| train31 | blend_default | `fc02f712-2e3f-4ddf-8d23-fd7a7352d726` |
| train31 | blend_identity_exact | `75f559d7-4978-4852-9788-0de2636a3d96` |
| train31 | blend_species_face | `dbda786d-10c6-4ca3-a1af-debccc25e467` |
| heldout00 | no_ip | `930ae327-753e-4110-be53-377d543a13d7` |
| heldout00 | blend_default | `ee0f3ea3-c954-41ac-b654-0a6f01700062` |
| heldout00 | blend_identity_exact | `5881ce79-523c-4f3d-9a75-4ad2a5fb7f19` |
| heldout00 | blend_species_face | `c60b8f4f-c610-4580-a3c3-49f04bcfeb50` |
| heldout01 | no_ip | `350dda39-7760-4e02-84a8-11cc5d12f474` |
| heldout01 | blend_default | `421eb22b-3508-4f93-af7c-d02c7bdc680f` |
| heldout01 | blend_identity_exact | `0a8967a7-8f3d-4576-8cc0-ff300e1f06eb` |
| heldout01 | blend_species_face | `06186862-51cb-4482-9eaa-cebd541923d3` |
| heldout02 | no_ip | `60b69913-d7ce-440b-a799-d1e66885dd43` |
| heldout02 | blend_default | `0715e4df-2765-4658-b20a-c7d0a86690bf` |
| heldout02 | blend_identity_exact | `ce8abc0f-917a-4472-81c1-581222bfe02e` |
| heldout02 | blend_species_face | `b34e4f6c-6ac9-4afe-91f4-171c3c3ea847` |
| heldout03 | no_ip | `fe663452-1fc4-4721-8d6e-28ee3c23dc0f` |
| heldout03 | blend_default | `607d91ca-a781-40c2-a064-8a05c50f2a9b` |
| heldout03 | blend_identity_exact | `b2a5bff3-b0b5-4fb4-b9a3-8f52554ac3f3` |
| heldout03 | blend_species_face | `80185d43-2342-4942-a903-76e27bbdf198` |
| heldout04 | no_ip | `9c02d154-9a30-48f0-bfc6-64a0c2f5411c` |
| heldout04 | blend_default | `0746f463-ec4e-47c6-9bc4-587f310aa1c8` |
| heldout04 | blend_identity_exact | `2b746cdd-5f36-4963-a299-e90b5d7319af` |
| heldout04 | blend_species_face | `a37e65fb-1b1d-47f2-b308-811e72e74525` |
| heldout05 | no_ip | `6739ab79-af73-4cbb-8cec-2f73ccdf5ce0` |
| heldout05 | blend_default | `19e2e699-ccf5-450c-9002-56693c526c3f` |
| heldout05 | blend_identity_exact | `3e7ff6e7-ef62-4195-bdb7-bb8450986f15` |
| heldout05 | blend_species_face | `1e170d1c-4174-4442-a1c0-f7c6be64431c` |
| heldout06 | no_ip | `dcf8b9d6-b296-4a2b-8548-4cf7fac51d5e` |
| heldout06 | blend_default | `82708656-4dd7-41ee-ae51-6ee599937c4a` |
| heldout06 | blend_identity_exact | `6830f282-fdce-48d8-bf73-41ef7652c97e` |
| heldout06 | blend_species_face | `3a3dc288-05c9-4551-9524-436ab2260a0b` |
| heldout07 | no_ip | `6562d765-67b7-499a-b89e-7111d69d770d` |
| heldout07 | blend_default | `492429c3-41ee-417b-8b42-772d0d85d3e8` |
| heldout07 | blend_identity_exact | `87eecf62-ffd6-40a8-869a-ca46deec814e` |
| heldout07 | blend_species_face | `1086fa13-df46-4758-8142-1988feec8541` |
