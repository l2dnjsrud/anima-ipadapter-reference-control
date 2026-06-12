# QwenVL Metric Probe c036

Date: 2026-06-12

## Question

Can `Qwen/Qwen3-VL-Embedding-2B` image embeddings serve as a stronger evaluation/training signal than the PE pooled-cosine metric for the c035 SigLIP attribute-reference run?

## Input

- Runtime summary: `eval/siglip_runtime_quality_20260612_c035_suite_v1/summary.json`
- Visual audit: `eval/siglip_runtime_quality_20260612_c035_suite_v1/visual_audit.json`
- Output metrics: `eval/qwenvl_metric_probe_20260612_c036_c035/qwenvl_similarity_metrics.json`
- Encoder: `Qwen/Qwen3-VL-Embedding-2B`

## Metric Result

| variant | mean cosine | no-IP baseline | mean uplift | improved rate |
| --- | ---: | ---: | ---: | ---: |
| `siglip_kv_init_w14` | 0.8310 | 0.7888 | +0.0422 | 0.84375 |
| `siglip_ref_retrieval_w14` | 0.8335 | 0.7888 | +0.0446 | 0.90625 |

QwenVL is more optimistic than PE pooled-cosine on c035. It sees the SigLIP outputs as improved over no-IP in most rows.

## Visual-Audit Alignment Check

| variant | identity-pass mean uplift | identity-fail mean uplift | note |
| --- | ---: | ---: | --- |
| `siglip_kv_init_w14` | +0.0365 | +0.0478 | identity-fail rows score higher |
| `siglip_ref_retrieval_w14` | +0.0368 | +0.0525 | identity-fail rows score higher |

This means the pooled QwenVL embedding similarity is not aligned enough with the stricter identity/distinctive-trait audit. It rewards broad style, composition, costume, and palette similarity, but it does not penalize the repeated black-haired wuxia template collapse strongly enough.

## Decision

Decision: `qwenvl_pooled_metric_auxiliary_only`

Use QwenVL pooled embeddings as an auxiliary metric, not as the primary quality gate and not as proof that QwenVL adapter-only training will solve identity control. The next stronger-encoder experiment should require one of these before long training:

1. visual-token or shallow/deep feature path, not only one pooled embedding;
2. identity-positive/negative supervision from same-character groups;
3. a metric objective that correlates with the c035 identity/distinctive-trait visual audit.

## Next Loop

The next experiment should build or mine identity-positive/negative pairs from the local color dataset, then test whether QwenVL/SigLIP/PE features separate same-character or distinctive-trait pairs better than they separate broad style. If that gate fails, pooled VL embeddings should not be used as the main training target.
