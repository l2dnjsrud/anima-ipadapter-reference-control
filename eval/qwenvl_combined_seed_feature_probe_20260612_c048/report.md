# c048 Combined Reviewed Seed QwenVL Feature Gate

Date: 2026-06-12

## Question

Does the c045 QwenVL pooled identity-proxy signal remain stable after adding
c047's QwenVL-ranked top20 positives to the existing c044 reviewed seed?

## Inputs

- c044 reviewed seed:
  `eval/reviewed_face_identity_candidates_20260612_c044/reviewed_candidate_pairs.jsonl`
- c047 reviewed top20:
  `eval/qwenvl_top20_reviewed_identity_20260612_c047/reviewed_candidate_pairs.jsonl`
- Combined reviewed seed:
  `combined_reviewed_candidate_pairs.jsonl`
- Pair probe manifest:
  `pair_probe_manifest.jsonl`

## Reviewed Seed Summary

| label | count |
|---|---:|
| combined reviewed rows | 44 |
| same_character | 25 |
| different_character | 15 |
| unclear | 4 |
| positive_usable | 18 |
| feature-probe output rows | 33 |

## QwenVL Pooled Result

| metric | value |
|---|---:|
| positive pairs | 18 |
| negative pairs | 15 |
| positive mean | 0.905330 |
| negative mean | 0.817701 |
| separation margin | 0.087629 |
| pairwise AUC | 0.907407 |
| midpoint accuracy | 0.818182 |
| decision | feature_separates_proxy_pairs |

The pass gate is margin `>= 0.05` and AUC `>= 0.70`.

## Decision

`qwenvl_pooled_identity_gate_stable_on_combined_seed`

QwenVL pooled now passes on a larger reviewed seed than c045. This is still an
identity-proxy feature gate, not a generation-quality result, but it is strong
enough to promote QwenVL pooled to the primary candidate ranking and gating
metric for the next mining loop.

## Next

Use QwenVL pooled to build a larger and more diverse reviewed identity set. Once
that set is large enough, train or calibrate an adapter/metric head against this
gate instead of relying on PE/SigLIP pooled similarity.
