# QwenVL c063 Calibrator-Only Gate

- Train contact sheet: `eval/qwenvl_c063_calibrator_only_gate_20260612/contact_sheet_train.jpg`
- Heldout contact sheet: `eval/qwenvl_c063_calibrator_only_gate_20260612/contact_sheet_heldout.jpg`
- Previous retrieval checkpoint: `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- c055 mixed checkpoint: `anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors`
- c063 checkpoint: `anima_qwenvl_ip_adapter_c063_calibrator_only_b128_0128_20260612.safetensors`
- Columns: reference / no_ip / blend_species_face / c063_calibrator_only_w14.

Decision: `not_promoted`

Metric summary:

- PE mean uplift: `blend_species_face=0.060893`, `c063_calibrator_only_w14=0.029465`.
- PE heldout uplift: `blend_species_face=0.053534`, `c063_calibrator_only_w14=0.005121`.
- QwenVL mean uplift: `blend_species_face=0.042190`, `c063_calibrator_only_w14=0.037178`.
- QwenVL heldout uplift: `blend_species_face=0.026471`, `c063_calibrator_only_w14=0.024371`.
- Runtime guard: generated `120` PNGs for `40` samples x `3` variants; blank count `0`; minimum pixel std `35.883`.
- ComfyUI guard: QwenVL loader/encode/apply nodes were visible through `/object_info`, c063 was selectable, and port `8116` was closed after cleanup.
- Visual audit: c063 is active but not better than `blend_species_face`; `heldout07` still collapses the green non-human side-profile reference into a human dark-villain body template.

See `visual_audit.md` and `visual_audit.json`.

| sample | variant | prompt_id |
| --- | --- | --- |
| train00 | no_ip | `e4ac90a5-9e5d-4eb3-984b-8ea290d6ace0` |
| train00 | blend_species_face | `57eea0b8-fcd1-4c5e-928d-f2ba48cd2aae` |
| train00 | c063_calibrator_only_w14 | `783da201-3424-4b2c-823e-ffb6a0b8bcad` |
| train01 | no_ip | `62042c08-ab94-4194-b6c0-5db3b1c8903b` |
| train01 | blend_species_face | `0fdd52b2-4488-4cac-8fee-9a1275d99721` |
| train01 | c063_calibrator_only_w14 | `415a0a37-d9da-4d03-8c87-82c29936043b` |
| train02 | no_ip | `448590e7-ecfe-4141-9775-1aac7d89aec4` |
| train02 | blend_species_face | `ac858c0d-2afd-421f-9bbf-3e1f062d640a` |
| train02 | c063_calibrator_only_w14 | `db9e2e1b-12b0-4f23-ba92-ae3d70db99c2` |
| train03 | no_ip | `8793f770-0f97-45a3-8a46-17b0cff6568f` |
| train03 | blend_species_face | `f046a663-0152-4c82-aa40-8a79f143e92b` |
| train03 | c063_calibrator_only_w14 | `a42ec1ef-f2f2-43e0-aabf-3546667a6825` |
| train04 | no_ip | `40cfccba-f8ec-4af3-81ec-0e662ac50bc7` |
| train04 | blend_species_face | `8d48bdf9-0731-4d68-8cf5-df60277938b3` |
| train04 | c063_calibrator_only_w14 | `1911e989-8d56-4585-9892-80ab8b532b55` |
| train05 | no_ip | `67a5adb2-ba83-4d63-b3bf-f49c05757ce3` |
| train05 | blend_species_face | `1fd0a404-8530-4f7e-84d9-61da5385bda4` |
| train05 | c063_calibrator_only_w14 | `ffb91243-edc0-4072-bbed-bf89fa759e6f` |
| train06 | no_ip | `e4bd888b-0421-4603-85c4-a36a0be69a87` |
| train06 | blend_species_face | `b03c72ac-5c9e-4d71-9c71-76156ab032d2` |
| train06 | c063_calibrator_only_w14 | `5331981d-fa16-4d26-8fc7-0aead69a8c29` |
| train07 | no_ip | `dc134581-ab6a-43d2-ad44-2b91bc0135f4` |
| train07 | blend_species_face | `e4c48e00-e06a-4fb0-85f2-3b60e0016db7` |
| train07 | c063_calibrator_only_w14 | `26afca96-0bf4-4f7f-9290-8d50a014fe33` |
| train08 | no_ip | `25d4b0a6-b1f2-44a5-8cbf-010610089391` |
| train08 | blend_species_face | `a95200d0-4d1b-44e2-92e1-9b7f9d9198e1` |
| train08 | c063_calibrator_only_w14 | `efc26c87-7730-4807-9bb5-51747368376e` |
| train09 | no_ip | `cf561293-524f-4ffe-b7cd-0d059d344f38` |
| train09 | blend_species_face | `8a955053-e369-428d-9762-9017fae5b169` |
| train09 | c063_calibrator_only_w14 | `5823c337-038c-4213-bc46-4f53f6d025d0` |
| train10 | no_ip | `0659c0c0-fd31-4bd6-99ba-d37fb2635e3a` |
| train10 | blend_species_face | `d5992b25-ba2b-4596-88a3-91953558c3de` |
| train10 | c063_calibrator_only_w14 | `99148252-5695-4e34-a8b3-7ccf9324d40e` |
| train11 | no_ip | `7d948ae4-63e5-4375-a40c-4869298c4e15` |
| train11 | blend_species_face | `2ea42ce9-3b8a-4213-8b3d-d19722d16ecb` |
| train11 | c063_calibrator_only_w14 | `118cdc15-58db-417e-9202-6b6dde2e1c78` |
| train12 | no_ip | `54651e45-c6e2-4adf-8056-0339bfceaadd` |
| train12 | blend_species_face | `8c9cb6fb-15e8-4027-b33f-70c34cbfe2df` |
| train12 | c063_calibrator_only_w14 | `87719266-d11a-4639-9fe8-2b8cc76b4139` |
| train13 | no_ip | `23f54183-661b-409e-8b92-34d99b9d0b06` |
| train13 | blend_species_face | `d1f277b2-df5a-44a1-b6ea-ab5b840e0c83` |
| train13 | c063_calibrator_only_w14 | `363c539c-8589-4082-8620-37e8e886a834` |
| train14 | no_ip | `d92011ad-addb-4ed9-b7e2-ab7eb62946bd` |
| train14 | blend_species_face | `e1a25634-c9bb-4a5a-8597-7b291225e027` |
| train14 | c063_calibrator_only_w14 | `8994ce4d-e6d0-4786-90ac-db7aaab6aab2` |
| train15 | no_ip | `be99f2bf-42c0-4f46-a986-7b214d3b2881` |
| train15 | blend_species_face | `006ca164-7f5d-4764-97d0-dff352d78524` |
| train15 | c063_calibrator_only_w14 | `c897bcb3-c7bc-4d5f-afda-b1fb153cf610` |
| train16 | no_ip | `ba632c6d-533e-42c0-ac8b-4fdcb3f189bf` |
| train16 | blend_species_face | `b8261cf1-3b9c-4bc8-8a48-1e0e736f27c3` |
| train16 | c063_calibrator_only_w14 | `3d11b1fc-bcf3-4763-974c-744d9148fbfa` |
| train17 | no_ip | `323898d8-d283-4787-9f4e-e5ee6134b412` |
| train17 | blend_species_face | `bf0d4019-1e25-43c1-943f-9b6fbf8d0cc9` |
| train17 | c063_calibrator_only_w14 | `737ef792-659f-49d2-af03-d919c3dfb289` |
| train18 | no_ip | `1948c3de-e5d7-4b00-997d-63e5185273b1` |
| train18 | blend_species_face | `648844a2-596e-4b2f-a31d-ea3cd19a757f` |
| train18 | c063_calibrator_only_w14 | `4dc7c203-92ab-4dcb-a4e2-509bdc152528` |
| train19 | no_ip | `04dc9550-4782-4f45-8ce5-fc2570b8a4fc` |
| train19 | blend_species_face | `0e1e2f35-0643-421c-8ffa-06675bce17a4` |
| train19 | c063_calibrator_only_w14 | `6eb78af1-f88e-42ca-8be9-d993b1e8eb72` |
| train20 | no_ip | `81020203-ed56-4df3-9822-3dad4fbd18ab` |
| train20 | blend_species_face | `1c678998-582a-43e7-aeae-d10c2470aba7` |
| train20 | c063_calibrator_only_w14 | `afc88188-ef5f-4635-bcd4-3be26faf280e` |
| train21 | no_ip | `f603bc8f-973a-47e2-b866-13b69d06989e` |
| train21 | blend_species_face | `b999cc51-5ae8-44e9-ac02-e8736964324f` |
| train21 | c063_calibrator_only_w14 | `23d9bbf9-3f63-47fc-8963-6d5d3360682b` |
| train22 | no_ip | `77796dd7-8138-4c46-98eb-1ad4ebdef483` |
| train22 | blend_species_face | `a25c8893-2d31-49f7-97b6-20a320d57c7e` |
| train22 | c063_calibrator_only_w14 | `7875ba5f-3954-4676-a8a8-57e2dd5eae3c` |
| train23 | no_ip | `9e32e057-1223-4c7f-9c35-1e9e5dba9c68` |
| train23 | blend_species_face | `6837e5c1-b9e3-4bef-a8e9-13e1a7b23cb9` |
| train23 | c063_calibrator_only_w14 | `eb89cd48-b855-416c-a4f2-64e1a9fe98f3` |
| train24 | no_ip | `9dbd6faa-f237-426f-8664-95b9267ab156` |
| train24 | blend_species_face | `b5dd605b-2caa-41b1-bb10-9a53977ed7ee` |
| train24 | c063_calibrator_only_w14 | `20ace872-b28e-4346-b355-69b4c00cfc26` |
| train25 | no_ip | `b784b01e-7baa-4738-b0c9-e83acd480788` |
| train25 | blend_species_face | `d0eae7b4-e3ed-4c9a-bfb7-fc614d203892` |
| train25 | c063_calibrator_only_w14 | `cd0ab5a8-9434-483d-a628-2a823ad2781c` |
| train26 | no_ip | `d9674b0c-2e10-45f2-b8a5-dbf5c996e0ae` |
| train26 | blend_species_face | `f6da3830-a37e-48b8-8481-19d3b64bdad9` |
| train26 | c063_calibrator_only_w14 | `abf7cd6e-89d8-420a-a4d1-7a641e9361f8` |
| train27 | no_ip | `7d1457d0-a62e-44a9-8882-1b0ec9e1c16a` |
| train27 | blend_species_face | `65bee6cf-06f0-4ab0-86f6-1145be606c8e` |
| train27 | c063_calibrator_only_w14 | `9d28af11-ab85-4e5c-97b8-89cb573990fc` |
| train28 | no_ip | `dc78b386-80a8-42cc-bb35-bf5f01b25cf0` |
| train28 | blend_species_face | `ee2566bf-93b7-434f-b399-9d9a2390a5e2` |
| train28 | c063_calibrator_only_w14 | `43058d0e-9d71-4ab5-b404-d89ee29de691` |
| train29 | no_ip | `55b663ad-33d7-4cb7-a37c-1b15797e9c38` |
| train29 | blend_species_face | `76bc6701-77e4-4c96-859a-7254f49a5be8` |
| train29 | c063_calibrator_only_w14 | `167b5caa-ec4d-49a2-8b0d-18bab82f0da4` |
| train30 | no_ip | `ddb00a07-4305-41da-b606-b8513a105261` |
| train30 | blend_species_face | `a8d8f548-e5e1-429a-94f0-b3ea6d2406b5` |
| train30 | c063_calibrator_only_w14 | `fa7c7d63-c943-45c1-9921-18c010651675` |
| train31 | no_ip | `f3192c65-85d0-4e32-b639-94d980a28dad` |
| train31 | blend_species_face | `cdac0c96-785f-4274-8b0e-77e0e782ad70` |
| train31 | c063_calibrator_only_w14 | `a810f37d-c965-4dab-8d14-fc2ac43c3723` |
| heldout00 | no_ip | `a7660eef-b6d9-4c5c-ab7d-cfd4f4a68358` |
| heldout00 | blend_species_face | `8fb0b848-fd2a-4262-b89d-b33e67d4fbc6` |
| heldout00 | c063_calibrator_only_w14 | `ebdd8701-f6b2-4f45-b657-9d70f3cdf2f5` |
| heldout01 | no_ip | `e013d623-3ae9-4258-8bf6-c1fc6a081dd5` |
| heldout01 | blend_species_face | `d9b16399-a4b7-44d5-a794-b2b7d6cf02f4` |
| heldout01 | c063_calibrator_only_w14 | `b0901ed8-7e64-4442-b5e3-fd5ff3e72e17` |
| heldout02 | no_ip | `10c189d3-b1e0-4b3b-95d3-82e5b0f5e528` |
| heldout02 | blend_species_face | `21909a16-edad-4187-a81f-8cc60a40387f` |
| heldout02 | c063_calibrator_only_w14 | `0fa1ffdb-cc93-4412-afe0-1eac16fd0cd1` |
| heldout03 | no_ip | `7fefb131-559d-4e2e-9a6d-ec8dbe8bb651` |
| heldout03 | blend_species_face | `e1edca51-ede4-4e95-ba55-0999b7f89624` |
| heldout03 | c063_calibrator_only_w14 | `47db09d5-2dec-46c2-911d-f645dcaa7d25` |
| heldout04 | no_ip | `d01cdddb-e2d4-4c08-8ce5-9af040ea15a5` |
| heldout04 | blend_species_face | `81b870dd-397b-4b60-8fcb-081d3f17c7fd` |
| heldout04 | c063_calibrator_only_w14 | `879159ba-c24f-4a58-acb6-75a615787d1e` |
| heldout05 | no_ip | `10c117a5-b96a-4cd2-b53e-113fe43c4f3e` |
| heldout05 | blend_species_face | `1771a284-a9c5-4f12-a7dd-7fd4ba4ae386` |
| heldout05 | c063_calibrator_only_w14 | `85570049-f860-43c6-b8b4-24bf46b3701b` |
| heldout06 | no_ip | `84754929-93c1-4bfa-a634-6a2f3d7263a7` |
| heldout06 | blend_species_face | `769ed10b-26d7-4b59-952b-c6a92eb9749c` |
| heldout06 | c063_calibrator_only_w14 | `4f25021d-50ef-4837-9611-b8c0acd6613c` |
| heldout07 | no_ip | `f46d0005-34d9-4a88-ac93-4a7e2b7046a0` |
| heldout07 | blend_species_face | `38841baf-d64c-461b-b637-640193cda8a1` |
| heldout07 | c063_calibrator_only_w14 | `97538a68-d31a-4246-9c35-f49fa0e3f6ce` |
