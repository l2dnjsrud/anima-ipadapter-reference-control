# c052 Combined Diverse Reviewed Seed QwenVL Feature Gate

Date: 2026-06-12

## Question

Does QwenVL pooled remain a stable identity-proxy feature after adding the c051
diverse reviewed seed to the c050 combined rank40 seed?

## Inputs

- c050 combined seed:
  `eval/qwenvl_combined_rank40_feature_probe_20260612_c050/combined_reviewed_candidate_pairs.jsonl`
- c051 diverse reviewed rows:
  `eval/qwenvl_diverse_identity_candidates_20260612_c051/reviewed_candidate_pairs.jsonl`
- Combined reviewed seed:
  `combined_reviewed_candidate_pairs.jsonl`
- Pair probe manifest:
  `pair_probe_manifest.jsonl`
- QwenVL pooled detailed report:
  `qwenvl_pooled_report.md`

## Reviewed Seed Summary

| label | count |
|---|---:|
| combined reviewed rows | 84 |
| same_character | 47 |
| different_character | 29 |
| unclear | 8 |
| positive_usable | 29 |
| feature-probe output rows | 58 |

## QwenVL Pooled Result

| metric | value |
|---|---:|
| positive pairs | 29 |
| negative pairs | 29 |
| positive mean | 0.897714 |
| negative mean | 0.825511 |
| separation margin | 0.072203 |
| pairwise AUC | 0.913199 |
| midpoint accuracy | 0.862069 |
| decision | feature_separates_proxy_pairs |

The pass gate is margin `>= 0.05` and AUC `>= 0.70`.

## Decision

`qwenvl_pooled_identity_gate_stable_on_diverse_seed`

QwenVL pooled remains stable after adding the diverse c051 reviewed set. The
seed is now more balanced (`29` positive / `29` negative) and covers more new
SG pages than the c050 seed.

This is still a data/feature gate, not a generation-quality result. The next
step is to use this larger reviewed seed for a bounded adapter or metric-head
training pilot and evaluate it through the c035-style single-character
generation gate.
