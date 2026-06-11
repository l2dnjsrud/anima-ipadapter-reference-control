# QwenVL c031 Attribute-Prompt Runtime Evaluation

- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/qwenvl_runtime_quality_20260611_c031_attribute_prompt_runtime/contact_sheet.jpg`
- Base checkpoint: `anima_qwenvl_ip_adapter_identity128_calibrated_contrastive_0064_20260611.safetensors`
- Retrieval checkpoint: `anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- Columns: reference / no_ip / qwen_base_w14 / qwen_retrieval_w1 / qwen_retrieval_w14.

Decision: `qwenvl_attribute_prompt_promising_not_primary_pass`

## PE Similarity Metrics

Metric file:
`eval/qwenvl_runtime_quality_20260611_c031_attribute_prompt_runtime/pe_similarity_metrics.json`

| variant | cases | mean uplift vs no-IP | improved rate |
| --- | ---: | ---: | ---: |
| qwen_base_w14 | 8 | 0.0883 | 75.00% |
| qwen_retrieval_w1 | 8 | 0.0346 | 50.00% |
| qwen_retrieval_w14 | 8 | 0.0983 | 87.50% |

## Visual Result

The QwenVL adapter becomes much more useful when the prompt names the visible
attributes of the reference. It improves PE reference similarity over no-IP in
most cases and clearly helps with the scholar, elder, dark-villain, and
non-human face examples.

This is not the primary SigLIP goal pass because it uses the QwenVL path, and
because no-IP also improves strongly with attribute prompts. It is useful
evidence that the earlier generic-prompt failures were partly a caption/prompt
conditioning problem, not only an adapter wiring problem.

| sample | variant | prompt_id |
| --- | --- | --- |
| train00 | no_ip | `4bde3ef0-7d5c-4edc-9dec-590e3dcd8892` |
| train00 | qwen_base_w14 | `e6092e27-e916-46c7-9bc8-7dabcf32d180` |
| train00 | qwen_retrieval_w1 | `7834a2d3-b7a1-4921-b3b6-185496f48ee8` |
| train00 | qwen_retrieval_w14 | `2af111bc-a874-4688-a2a4-849a3a958f6d` |
| train07 | no_ip | `bc870776-3bca-42a9-94ae-15bcf72f5d35` |
| train07 | qwen_base_w14 | `d604ac08-0771-42f6-9815-07525567660a` |
| train07 | qwen_retrieval_w1 | `b0e3ba55-676f-4f77-bafb-a56a377a532e` |
| train07 | qwen_retrieval_w14 | `fe3c2443-5080-47d9-ab4b-f3ea5eaf5fac` |
| train14 | no_ip | `6228cc63-c3d3-4961-8e91-e5f2555eabc5` |
| train14 | qwen_base_w14 | `e3656b6a-28d2-4678-a9da-badb5a0eab90` |
| train14 | qwen_retrieval_w1 | `0a825cf0-aa46-4f2e-8bfc-5a456a8848f6` |
| train14 | qwen_retrieval_w14 | `25ae7948-43f2-4052-bc39-381c1c9848b9` |
| train23 | no_ip | `0007e39d-1ff5-498b-8635-d28676aeb5c1` |
| train23 | qwen_base_w14 | `f827347b-e78a-40e9-a5ab-c565d617acf8` |
| train23 | qwen_retrieval_w1 | `0c5e2bcf-d4e5-4d0a-9d28-9d345d18f935` |
| train23 | qwen_retrieval_w14 | `e0714c22-7e3b-48e9-90d5-2efe6c17accc` |
| heldout00 | no_ip | `5d5f8460-a240-4e18-9093-26c0b65518fd` |
| heldout00 | qwen_base_w14 | `75acd69a-1585-4beb-b091-9a24a9b6ae22` |
| heldout00 | qwen_retrieval_w1 | `ad624775-79fd-425c-93a5-d7bee96e4c49` |
| heldout00 | qwen_retrieval_w14 | `b7218624-4e91-46b2-ae97-7605d4056c0d` |
| heldout02 | no_ip | `ed14ba50-0275-4347-bda1-f23146170311` |
| heldout02 | qwen_base_w14 | `652cb6a7-c883-4a07-9a03-1b6c18cc4200` |
| heldout02 | qwen_retrieval_w1 | `0f2cee7f-ef08-4a87-82d2-184bda0acde7` |
| heldout02 | qwen_retrieval_w14 | `059834c2-81db-40a6-b4da-dbf14fd2cc59` |
| heldout05 | no_ip | `0ee9315c-ecb8-46d6-98bf-0952a8e2ea08` |
| heldout05 | qwen_base_w14 | `ddc085e1-a7ef-45e2-85fa-c09b742bb3b6` |
| heldout05 | qwen_retrieval_w1 | `a88cb77c-180d-43ad-9243-03c0e75c6c85` |
| heldout05 | qwen_retrieval_w14 | `ecb1eb6e-1c67-40e3-b2db-575c3adb38f2` |
| heldout07 | no_ip | `75d86c39-69c3-48bc-b3c3-78b17dfb072b` |
| heldout07 | qwen_base_w14 | `becf9017-6c26-4a6c-8050-6d3ef10c8209` |
| heldout07 | qwen_retrieval_w1 | `cfd13fff-c9b8-45b1-8702-6c3476a7fef2` |
| heldout07 | qwen_retrieval_w14 | `9eeb7812-afd1-45e0-a134-32fd6936514b` |
