# c050 Combined Rank40 Reviewed Seed QwenVL Feature Gate

Date: 2026-06-12

## Question

Does QwenVL pooled remain a stable identity-proxy feature after adding the
noisier c049 rank21-40 reviewed rows to the c048 combined seed?

## Inputs

- c048 combined seed:
  `eval/qwenvl_combined_seed_feature_probe_20260612_c048/combined_reviewed_candidate_pairs.jsonl`
- c049 rank21-40 reviewed rows:
  `eval/qwenvl_rank21_40_reviewed_identity_20260612_c049/reviewed_candidate_pairs.jsonl`
- Combined reviewed seed:
  `combined_reviewed_candidate_pairs.jsonl`
- Pair probe manifest:
  `pair_probe_manifest.jsonl`
- QwenVL pooled detailed report:
  `qwenvl_pooled_report.md`

## Reviewed Seed Summary

| label | count |
|---|---:|
| combined reviewed rows | 52 |
| same_character | 30 |
| different_character | 17 |
| unclear | 5 |
| positive_usable | 19 |
| feature-probe output rows | 36 |

## QwenVL Pooled Result

| metric | value |
|---|---:|
| positive pairs | 19 |
| negative pairs | 17 |
| positive mean | 0.903785 |
| negative mean | 0.822187 |
| separation margin | 0.081599 |
| pairwise AUC | 0.900929 |
| midpoint accuracy | 0.861111 |
| decision | feature_separates_proxy_pairs |

The pass gate is margin `>= 0.05` and AUC `>= 0.70`.

## Decision

`qwenvl_pooled_identity_gate_stable_on_rank40_combined_seed`

QwenVL pooled remains stable after adding rank21-40 noise. The margin is still
comfortably above the gate and AUC remains above `0.90`.

This is still not a generation-quality result. It means the next data loop can
use QwenVL pooled as the primary identity ranking/gating signal, while keeping
manual review for labels.

## Next

Build a larger, more diverse reviewed identity set with QwenVL ranking, focusing
on new SG pages and avoiding repeated protagonist-heavy duplicates. Only after
that larger set passes should adapter/metric-head training begin.
