# SigLIP c032 Attribute-Prompt Runtime Evaluation

- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260611_c032_attribute_prompt_runtime/contact_sheet.jpg`
- Clean32 checkpoint: `anima_siglip_ip_adapter_single_character_clean32_pe_query_patch_0512_20260611.safetensors`
- PE-space checkpoint: `anima_siglip_ip_adapter_single_character_clean32_pe_space_init_0512_20260611.safetensors`
- PE-retrieval checkpoint: `anima_siglip_ip_adapter_single_character_clean32_pe_retrieval_0128_20260611.safetensors`
- Columns: reference / no_ip / clean32_w14 / pe_space_w14 / pe_retrieval_w14.

Decision: `siglip_attribute_prompt_reference_control_pass`

## PE Similarity Metrics

Metric file:
`eval/siglip_runtime_quality_20260611_c032_attribute_prompt_runtime/pe_similarity_metrics.json`

| variant | cases | mean uplift vs no-IP | improved rate |
| --- | ---: | ---: | ---: |
| clean32_w14 | 8 | 0.0515 | 75.00% |
| pe_space_w14 | 8 | 0.0603 | 100.00% |
| pe_retrieval_w14 | 8 | 0.0670 | 87.50% |

## Visual Result

This is the first practical quality pass for the native SigLIP path under a
realistic prompting condition: the prompt names the visible character class and
important attributes, while the adapter receives the reference image.

Compared with the no-IP column, the SigLIP adapter columns visibly pull outputs
toward the reference in color, face framing, costume silhouette, and expression.
The strongest general-purpose variant in this sheet is `pe_space_w14`: it
improves PE similarity on all 8 cases and keeps good visual quality. The
`pe_retrieval_w14` variant has the highest mean uplift and gives especially
strong gains on angry close-up and held-out elder references, but has one
negative case on the old bearded train sample.

This does not prove reference-only generation from a generic prompt. It proves
the usable recipe for high-quality reference control is now: good
identity/palette/prop prompt or caption plus the native SigLIP adapter. The next
production step should be automatic attribute/caption extraction for training
and ComfyUI workflows so the user does not have to hand-author these prompts.

| sample | variant | prompt_id |
| --- | --- | --- |
| train00 | no_ip | `4644d71a-a985-443b-9fff-264dfc84cb9e` |
| train00 | clean32_w14 | `c3b9bcf2-33f0-4de6-bce6-498c6edaa6f6` |
| train00 | pe_space_w14 | `ce53bf53-d629-491d-a763-8e0aba43bf45` |
| train00 | pe_retrieval_w14 | `9626ac99-6da9-4cc3-baa3-f45dbd3756c8` |
| train07 | no_ip | `e6307111-2a0f-4405-809a-b6448f0b33fc` |
| train07 | clean32_w14 | `7feefb2e-7648-4a07-9ae8-62acc1f72b3d` |
| train07 | pe_space_w14 | `50b0e46b-cb38-4c14-a90c-4bcc650497ed` |
| train07 | pe_retrieval_w14 | `2d8d745b-11c8-41c0-8fff-bdf7d7e0c2ac` |
| train14 | no_ip | `f5c23094-6345-4d97-b86b-a460d021bbfd` |
| train14 | clean32_w14 | `f19ee6a2-a515-4131-95e7-27f91a7e4c76` |
| train14 | pe_space_w14 | `116d8902-96dd-4b72-95e2-a41c27287374` |
| train14 | pe_retrieval_w14 | `a4e7a68b-699e-4f41-9e6a-422b298a0826` |
| train23 | no_ip | `a5675a9b-2504-4f6d-b6df-7bd8041a5341` |
| train23 | clean32_w14 | `a283b441-5b3b-49e6-97b4-09127d38a9c2` |
| train23 | pe_space_w14 | `86eb0a3a-05e3-4466-8c35-b3279c8de645` |
| train23 | pe_retrieval_w14 | `05861927-36de-4fe8-85a9-fb1f46d0246d` |
| heldout00 | no_ip | `b82bd83e-b9bf-4f33-bd22-7fe97879fc64` |
| heldout00 | clean32_w14 | `874b8338-4d53-4f9b-8a81-261f59b10454` |
| heldout00 | pe_space_w14 | `2a1f348b-0556-40a7-9797-ce0ac3035478` |
| heldout00 | pe_retrieval_w14 | `36ae452f-11ad-4863-b3aa-c40e9a7fb101` |
| heldout02 | no_ip | `67297335-0531-40c5-a756-585828575aea` |
| heldout02 | clean32_w14 | `30bf0bd1-e9ca-48a4-9e29-78d9703848a1` |
| heldout02 | pe_space_w14 | `647f8ffd-44b1-4861-9a23-992115466d1a` |
| heldout02 | pe_retrieval_w14 | `1ba9addc-bc37-4bb7-a59b-10ec3c66f694` |
| heldout05 | no_ip | `d2a192e7-8c2e-488d-8814-1bf784e92e36` |
| heldout05 | clean32_w14 | `456eea27-5b4d-4eda-99fd-ee4a0abe8c28` |
| heldout05 | pe_space_w14 | `5531ace6-59f9-4dd8-b3e8-7445a6097de1` |
| heldout05 | pe_retrieval_w14 | `df3791a1-4bf0-41c8-bd28-25462edc4d94` |
| heldout07 | no_ip | `dfb61eaa-81c3-44fd-8900-94c9a3c345d5` |
| heldout07 | clean32_w14 | `e34b4f69-59c9-4c72-9817-6c867c15d166` |
| heldout07 | pe_space_w14 | `aeba4812-b139-4e8c-88a0-24c740c29aee` |
| heldout07 | pe_retrieval_w14 | `c0585a4a-0ce8-4ae0-8282-74e1da6ab076` |
