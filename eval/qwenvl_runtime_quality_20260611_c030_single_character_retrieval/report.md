# QwenVL c030 Single-Character Retrieval Runtime Evaluation

- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_runtime_quality_20260611_c030_single_character_retrieval/contact_sheet.jpg`
- Base checkpoint: `anima_qwenvl_ip_adapter_identity128_calibrated_contrastive_0064_20260611.safetensors`
- Retrieval checkpoint: `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- Columns: reference / no_ip / qwen_base_w1 / qwen_base_w14 / qwen_retrieval_w1 / qwen_retrieval_w14.

Decision: `qwen_retrieval_single_character_not_quality_pass`

## Training Summary

The retrieval checkpoint continued from the calibrated QwenVL identity128
adapter with a token-retrieval margin loss between the adapter's trainable
tokens and QwenVL image embeddings.

- manifest: `training/manifests/local_color_single_character_clean32_20260611.jsonl`
- rows loaded: `32`
- steps: `128`
- first/final loss: `0.2704607844352722` / `0.3180232048034668`
- mean loss: `0.32662612583953887`
- mean base loss: `0.21053104143356904`
- mean contrastive loss: `0.04986031912267208`
- mean retrieval loss: `0.20726001169532537`
- finite loss: `true`

The retrieval loss stayed close to the configured margin, so this short run did
not prove that the QwenVL embedding branch learned a strong matching-vs-wrong
reference separation signal.

## PE Similarity Metrics

Metric file:
`eval/qwenvl_runtime_quality_20260611_c030_single_character_retrieval/pe_similarity_metrics.json`

| variant | cases | mean uplift vs no-IP | improved rate |
| --- | ---: | ---: | ---: |
| qwen_base_w1 | 8 | -0.0024 | 50.00% |
| qwen_base_w14 | 8 | 0.0496 | 62.50% |
| qwen_retrieval_w1 | 8 | 0.0056 | 62.50% |
| qwen_retrieval_w14 | 8 | -0.0077 | 50.00% |

## Runtime Summary

This sheet was generated through an isolated ComfyUI API server on
`127.0.0.1:8116`. The QwenVL model selector exposed both the calibrated base
checkpoint and the new retrieval checkpoint, and all 40 prompts completed.

The isolated ComfyUI run needed `sentence_transformers` on the ComfyUI Python
path. For this evaluation it was loaded from a temporary target directory under
`.tmp`; a normal installation should install the dependency into the active
ComfyUI Python environment.

## Visual Result

The QwenVL retrieval adapter clearly changes the no-IP baseline, especially in
red/dark palette and wuxia portrait styling. It still does not pass the
reference-control gate:

- `train14` is an old exaggerated bearded man, but outputs become young or
  middle-aged black-haired wuxia men.
- `train23` has glasses, a scholar hat, and a fan; outputs lose those props and
  collapse to a generic robed male.
- `heldout02` is bald, old, and heavily bearded; outputs become young
  black-haired men.
- `heldout05` is a screaming close crop; outputs become calm normal portraits.
- `heldout07` is a green demon face with a red eye; outputs become human male
  portraits.

Conclusion: the single-character test is the right gate, but this QwenVL
token-retrieval pilot is not enough. The adapter receives and uses a reference
signal, yet it does not preserve identity, props, non-human traits, or
expression. The next useful branch should train a stronger image-feature
calibrator/encoder or explicit identity/palette/prop token objective before
another denoising-focused run.

| sample | variant | prompt_id |
| --- | --- | --- |
| train00 | no_ip | `b4a8ff93-eb5c-47db-b673-894a2e58269d` |
| train00 | qwen_base_w1 | `c3020430-99aa-49e7-9c11-7e6dcc4267dd` |
| train00 | qwen_base_w14 | `d1c0c0a9-73f3-4b57-8270-3eb052e91e25` |
| train00 | qwen_retrieval_w1 | `f99d8d92-3f7c-40e9-9235-a694d454bb48` |
| train00 | qwen_retrieval_w14 | `a1c904b6-7c09-479a-84e4-ddfaaddd61cd` |
| train07 | no_ip | `1ff0d18f-d2ab-43a3-9fc0-35820cfc911c` |
| train07 | qwen_base_w1 | `f1ae0924-a92e-40db-85da-37d2fedbcb7e` |
| train07 | qwen_base_w14 | `b0788740-076a-4e1d-a8ab-c3d6d9c16c81` |
| train07 | qwen_retrieval_w1 | `e3e7e859-d632-4e19-909d-d77526239923` |
| train07 | qwen_retrieval_w14 | `fdc97926-b786-410d-be3b-5f53028e0d67` |
| train14 | no_ip | `0a04f1a6-e01d-40a4-9756-33d77fed0c82` |
| train14 | qwen_base_w1 | `28537dd2-74b3-4b68-86f7-e2b648eb81ec` |
| train14 | qwen_base_w14 | `bd531ea9-20f3-4dd0-8811-25cd85c3bef0` |
| train14 | qwen_retrieval_w1 | `df5eec2f-6063-4ac7-91c1-b3cc8f2eedb7` |
| train14 | qwen_retrieval_w14 | `dfa3b230-9831-4890-8034-e4f3ac7bfa9f` |
| train23 | no_ip | `ec207f04-b12d-4f3e-9474-a5f3e8f327a8` |
| train23 | qwen_base_w1 | `7a305297-373d-4eb0-ab8f-e68b1c0a01b7` |
| train23 | qwen_base_w14 | `af0277ca-1457-4c70-8e2e-24e68998c197` |
| train23 | qwen_retrieval_w1 | `a19bd215-d115-4ab8-acfa-b96f693089b8` |
| train23 | qwen_retrieval_w14 | `84869797-3b78-4ad0-974d-bcf2ecf61f29` |
| heldout00 | no_ip | `14814c2d-f8a5-45f0-987f-4a90cc64823f` |
| heldout00 | qwen_base_w1 | `5810c093-cf88-4d7e-98b9-41b65a120256` |
| heldout00 | qwen_base_w14 | `87b7f76c-9f99-458f-b595-bffbbc81b7c8` |
| heldout00 | qwen_retrieval_w1 | `482ffba1-b304-4d54-b909-bd2152b9c700` |
| heldout00 | qwen_retrieval_w14 | `2e7740c8-d234-4beb-a538-a401cc1dd824` |
| heldout02 | no_ip | `a8d5321d-f2fc-4ab3-ae87-dc92fec727d9` |
| heldout02 | qwen_base_w1 | `70845533-ee74-4a7e-bafb-2cfef215f71d` |
| heldout02 | qwen_base_w14 | `f4e30f4b-0331-4c30-a596-d790ace65f4c` |
| heldout02 | qwen_retrieval_w1 | `452b6e9f-e72f-4ffc-9a56-20fc23e508c6` |
| heldout02 | qwen_retrieval_w14 | `7f94ced6-44df-485a-bbe3-7c2b855e1812` |
| heldout05 | no_ip | `9bfcc215-2be8-43ee-8f4d-52b64ec4907f` |
| heldout05 | qwen_base_w1 | `bdb1b3a0-4891-4088-b53e-8a2fb6d1bf74` |
| heldout05 | qwen_base_w14 | `ff915751-a97f-49c3-8dab-e241fb713f1d` |
| heldout05 | qwen_retrieval_w1 | `6de296ae-950b-4e51-b855-4efdadc576c6` |
| heldout05 | qwen_retrieval_w14 | `1b441084-7d7e-4d85-987f-dca329781536` |
| heldout07 | no_ip | `cd526b3d-0f62-4e89-a867-ed61c027cc4d` |
| heldout07 | qwen_base_w1 | `cbda826d-f8a2-4911-b364-579bc86b84d9` |
| heldout07 | qwen_base_w14 | `a3a86573-e985-4b32-b906-b7e57c0e4e42` |
| heldout07 | qwen_retrieval_w1 | `76d652de-8969-479c-8cc5-afce15021715` |
| heldout07 | qwen_retrieval_w14 | `90cf25cc-9def-4405-8cae-5aa6d2218ca9` |
