# QwenVL c062 Calibrator Distillation Gate

- Train contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_c062_calibrator_distillation_gate_20260612/contact_sheet_train.jpg`
- Heldout contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_c062_calibrator_distillation_gate_20260612/contact_sheet_heldout.jpg`
- Previous retrieval checkpoint: `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- c055 mixed checkpoint: `anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors`
- c062 checkpoint: `anima_qwenvl_ip_adapter_c062_calibrator_distill_b128_0096_20260612.safetensors`
- Columns: reference / no_ip / blend_species_face / c062_w14.

Decision: `not_promoted`

Metric summary:

- PE mean uplift: `blend_species_face=0.060893`, `c062_w14=0.013234`.
- PE heldout uplift: `blend_species_face=0.053534`, `c062_w14=-0.003277`.
- QwenVL mean uplift: `blend_species_face=0.042190`, `c062_w14=0.026588`.
- QwenVL heldout uplift: `blend_species_face=0.026471`, `c062_w14=0.001077`.
- Runtime guard: generated `120` PNGs for `40` samples x `3` variants; blank count `0`; minimum pixel std `35.883`.
- ComfyUI guard: QwenVL loader/encode/apply nodes were visible through `/object_info`, c062 was selectable, and port `8116` was closed after cleanup.
- Visual audit: c062 is active but not better than `blend_species_face`; `heldout07` still collapses the green non-human side-profile reference into a human dark-villain body template.

See `visual_audit.md` and `visual_audit.json`.

| sample | variant | prompt_id |
| --- | --- | --- |
| train00 | no_ip | `5ffec4bc-db08-4033-9141-63050756b6bb` |
| train00 | blend_species_face | `d7c0f069-5255-472d-a767-ee4200a27372` |
| train00 | c062_w14 | `20631c03-7fbd-4786-aa24-abe5c88767cf` |
| train01 | no_ip | `e548a840-a319-48c5-8459-3214c523433b` |
| train01 | blend_species_face | `90df8038-d091-4c86-9152-8cad16fc9bd4` |
| train01 | c062_w14 | `2efe23d9-4f30-4a5a-beb3-deaa56329f20` |
| train02 | no_ip | `7a2a46cd-8e7c-46fa-bff4-75c3ae94b90f` |
| train02 | blend_species_face | `af00f0e0-ee5d-4d00-b3b3-4a3e363577eb` |
| train02 | c062_w14 | `a44a278e-66af-4ff3-b2f7-7fc0482b62f3` |
| train03 | no_ip | `1723e8d3-1110-4a7e-84f1-f2e640f05d11` |
| train03 | blend_species_face | `6ea54337-994d-4fc4-991a-7059ce5d3a88` |
| train03 | c062_w14 | `6c159eac-c3d0-4518-9723-de06b0a7db99` |
| train04 | no_ip | `22f30cfd-4dfa-4ba1-9ecd-ee135d42115c` |
| train04 | blend_species_face | `86fe81e4-91d0-48a3-b050-45ecac758f32` |
| train04 | c062_w14 | `5198ef7a-cba8-4547-8d9b-8b92403dbb17` |
| train05 | no_ip | `39396cac-bdb7-484c-ab98-4ce97eb3cba7` |
| train05 | blend_species_face | `d1b174eb-51f3-45d9-9108-82c72b380d86` |
| train05 | c062_w14 | `f34b4527-1f93-4235-93dd-c74717ed12bb` |
| train06 | no_ip | `d3209ed8-fb81-428a-9fb4-56992e8b6198` |
| train06 | blend_species_face | `ec1955da-135e-481a-927c-afc1ea2fa35a` |
| train06 | c062_w14 | `026505d7-9a25-47ca-897a-f09f3b11a1e7` |
| train07 | no_ip | `f88c3de5-271f-4c65-a82d-ff31698e5ef5` |
| train07 | blend_species_face | `e72146e0-975b-464f-a616-f273da6b245f` |
| train07 | c062_w14 | `f2a869fb-e1fb-473d-8a9d-43b4e03dd52c` |
| train08 | no_ip | `9e6c1e88-761c-431c-b38d-bb6573ef1a76` |
| train08 | blend_species_face | `ea48aa82-d8bc-4c05-ac5e-a06ce4440aa8` |
| train08 | c062_w14 | `bce23f12-a93c-4ce4-8ac1-3e0847fe1480` |
| train09 | no_ip | `2019a7f1-1d25-4750-95bc-e854e835756a` |
| train09 | blend_species_face | `9f81d6ba-fc24-4f48-ab73-a34d934b5fe3` |
| train09 | c062_w14 | `3527d51f-bb56-4115-bd04-1538352aeabe` |
| train10 | no_ip | `bb4d503b-cc71-4b82-89ce-65cee66ea2a9` |
| train10 | blend_species_face | `d9a02ff5-9f13-4fd5-aabf-972deecb481b` |
| train10 | c062_w14 | `4575617e-dea7-4e5f-9a4f-23dead0f55e0` |
| train11 | no_ip | `e8aa3719-279b-4272-8f48-2581d37c08d8` |
| train11 | blend_species_face | `4bbb8ddd-d8da-45de-a219-e86bc091ca8a` |
| train11 | c062_w14 | `f6f16078-b4d3-4b5a-8f46-3b579b2831f3` |
| train12 | no_ip | `dd3f28e1-d48a-45ea-804d-4fc4b7f48b54` |
| train12 | blend_species_face | `42bbad03-ed8b-408e-ba9d-5673545052fa` |
| train12 | c062_w14 | `9ef71b4a-df0d-4172-9fb6-b5f5bafa1eb6` |
| train13 | no_ip | `f967abf7-bc67-40e8-8fd6-6fd4adfe934d` |
| train13 | blend_species_face | `f9d0b32c-8c44-4e04-932a-ca2e9256d73d` |
| train13 | c062_w14 | `71d88f3e-e172-48f8-acc8-546a7f84c785` |
| train14 | no_ip | `c4a26954-3245-42e9-afca-c473e50ebca8` |
| train14 | blend_species_face | `2645322e-2fc9-46f6-9279-01625f4a5f82` |
| train14 | c062_w14 | `b5d40e0c-b9d7-4a93-9dd6-5c617a04b201` |
| train15 | no_ip | `2940f0ce-084e-4fa8-82cf-181a6b86ee97` |
| train15 | blend_species_face | `692fbacf-5d97-4d10-a78b-ef18d3c9ca1f` |
| train15 | c062_w14 | `0a645d37-04f2-4c22-804f-951382d4f3a1` |
| train16 | no_ip | `8d26ec60-10bf-4257-aabd-8e366bdee08e` |
| train16 | blend_species_face | `f55484d1-cd50-4465-8b18-64ee71cdc9cb` |
| train16 | c062_w14 | `248e1283-a2ee-4f31-b54e-6c6d6b97819a` |
| train17 | no_ip | `9a429f33-7b4e-4e61-b5a4-7c04a7a5dc72` |
| train17 | blend_species_face | `3345c885-7eee-42db-bc46-eaea2e0d2d62` |
| train17 | c062_w14 | `85bf8a0f-8913-419b-95f4-7b756ea223e7` |
| train18 | no_ip | `31d90c8e-8785-4d53-8c12-9ef9afe8abe9` |
| train18 | blend_species_face | `e9eade9f-05d5-43e6-ba9b-35a9ff6fe83c` |
| train18 | c062_w14 | `442cbbcf-9985-4d8d-b7ca-f2763b5d4b0f` |
| train19 | no_ip | `41e665ff-d723-4657-958f-2796ff118655` |
| train19 | blend_species_face | `6b2d4419-8ad2-4a4e-8983-2a6644c298b3` |
| train19 | c062_w14 | `28cac092-ad16-4a20-93ae-8da005b29c86` |
| train20 | no_ip | `2b214f54-de17-47f8-ba91-e8531ea634f1` |
| train20 | blend_species_face | `1c471016-aad5-4b3e-92c3-b8cd779074cb` |
| train20 | c062_w14 | `021b21e3-b074-42be-8569-e16c63a9034d` |
| train21 | no_ip | `3f4b8286-823f-49bc-97c5-f50021eaa52f` |
| train21 | blend_species_face | `44779cce-ec58-4530-896a-f611d8793e76` |
| train21 | c062_w14 | `e928aa13-4bff-4c4d-89fb-e9279ec2da19` |
| train22 | no_ip | `a1b8562f-8d94-4da4-baa5-53157da91d47` |
| train22 | blend_species_face | `da1c77a9-38b7-4069-a6dc-e9d78a3e489a` |
| train22 | c062_w14 | `7dcbf585-04fb-4443-82e6-1e7a81ef4339` |
| train23 | no_ip | `5c8af1f5-dbf1-4220-9e83-13aa448c9c8a` |
| train23 | blend_species_face | `95619740-51d2-4715-95d5-dd06485b9dd4` |
| train23 | c062_w14 | `9d72af5e-eb32-40ca-ab25-9ea9d2a363f1` |
| train24 | no_ip | `cfb05a67-8496-4d49-ae7b-f5b5d235f64b` |
| train24 | blend_species_face | `3c5c96f3-0c44-43f2-8155-98746e9c4a45` |
| train24 | c062_w14 | `9aae94f5-60fc-4f53-a8a1-17d2b29345cb` |
| train25 | no_ip | `8b65d7a6-8f55-4fbd-9971-81a2303e32cc` |
| train25 | blend_species_face | `a031be68-6ba5-49c6-a601-c5f387ca57f3` |
| train25 | c062_w14 | `6b3d8a59-142e-48b0-b4a5-19b3844f70b2` |
| train26 | no_ip | `c39d5a84-91e1-477c-9151-ebdd50861654` |
| train26 | blend_species_face | `6ba8cfe7-000e-41cc-b6cf-191b7599018c` |
| train26 | c062_w14 | `e328ae7f-448f-4cec-888a-720794e25ce8` |
| train27 | no_ip | `672ef9c1-7e78-4708-83a3-b447933954fb` |
| train27 | blend_species_face | `c0640339-440a-4af8-a67c-8f2b268dad89` |
| train27 | c062_w14 | `76552fa9-34ea-4b8c-a5ad-c989b3347166` |
| train28 | no_ip | `4d08e38f-20ee-4905-92e9-6d88399800fb` |
| train28 | blend_species_face | `de729699-cabc-47d4-ba81-77f9bd9d9803` |
| train28 | c062_w14 | `694582c5-0097-4162-9001-6e40deefc99a` |
| train29 | no_ip | `182deef8-b02a-45b0-b3b6-c58abfbcf141` |
| train29 | blend_species_face | `877e676d-c733-436b-9fd5-49c99ed6cd7e` |
| train29 | c062_w14 | `a3f12afa-c7ad-4fd5-9bad-2ddad7f1e157` |
| train30 | no_ip | `8aa8e06a-52c2-4cc8-9caf-c1cd046bf355` |
| train30 | blend_species_face | `0efe9947-d32c-4a93-8aeb-19c93f8da69c` |
| train30 | c062_w14 | `e5871b07-937b-4b55-a1a2-dddadd9870a9` |
| train31 | no_ip | `468c8d1c-6d51-4f84-ba89-5babefe3766f` |
| train31 | blend_species_face | `042f7835-3a94-44e4-b062-33eb8724f086` |
| train31 | c062_w14 | `ffa45442-0392-426e-9b83-0398cbe660b3` |
| heldout00 | no_ip | `fe64f91f-3fe3-489e-b712-f398aae4d66b` |
| heldout00 | blend_species_face | `f6a46601-cd7e-48c8-82e2-300e1f7ffa3d` |
| heldout00 | c062_w14 | `bc32bd53-8852-4096-83bf-a576bca8f50e` |
| heldout01 | no_ip | `6bccc034-57a3-49b7-86a6-380692213011` |
| heldout01 | blend_species_face | `262caf9e-e854-4be7-93e4-b00c85224e02` |
| heldout01 | c062_w14 | `e48829d8-e817-4953-bcbc-f44c37290900` |
| heldout02 | no_ip | `fc22b933-1268-44bf-952e-7cb3bc27b765` |
| heldout02 | blend_species_face | `d553e2f9-ffcc-4eec-89e0-4e7669c0b31b` |
| heldout02 | c062_w14 | `49ad90e7-cb8c-4c64-8f60-7c22de03f61e` |
| heldout03 | no_ip | `99f2b418-08a9-4478-9946-8ad6dd92f6a8` |
| heldout03 | blend_species_face | `b27478cd-f98d-45fe-99a9-0f7ea3da0e03` |
| heldout03 | c062_w14 | `6c175b17-626d-44f0-8fa5-e672ad82b24a` |
| heldout04 | no_ip | `77bc875f-472e-46b7-bbf6-24cd26ed1334` |
| heldout04 | blend_species_face | `d179f9e2-5e79-4eb4-8cf3-66ac8ed06844` |
| heldout04 | c062_w14 | `a12ee88a-7d2c-4919-8142-483f870bec18` |
| heldout05 | no_ip | `148fe1f4-573e-4d57-9d90-fc295179a860` |
| heldout05 | blend_species_face | `31c590c0-63d9-4ab1-af07-205ddce7b08f` |
| heldout05 | c062_w14 | `be64ce37-2563-494b-af2f-bac4816ff0ee` |
| heldout06 | no_ip | `c90119e4-59c9-4e01-b709-101b9dd93f95` |
| heldout06 | blend_species_face | `62b85897-6321-48e6-938e-c045105178e1` |
| heldout06 | c062_w14 | `77d2893d-a162-4b04-a9c8-fe523823f3f5` |
| heldout07 | no_ip | `1cef14a5-ffba-4843-922e-f7a4ddcf6cd2` |
| heldout07 | blend_species_face | `3567f7ce-4a86-4412-ba02-4feef0d53777` |
| heldout07 | c062_w14 | `80740d2c-1d87-45b6-bdca-9b1f761c1a16` |
