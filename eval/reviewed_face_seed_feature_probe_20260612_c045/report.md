# c045 Reviewed Face Seed Feature Probe

Date: 2026-06-12

## Question

After c044 expanded the reviewed seed to 8 usable positives and 15 hard
negatives, do QwenVL, SigLIP, PE, or SigLIP token features separate
same-character pairs from different-character pairs?

## Inputs

- Reviewed seed:
  `eval/reviewed_face_identity_candidates_20260612_c044/reviewed_candidate_pairs.jsonl`
- Pair probe manifest:
  `eval/reviewed_face_seed_feature_probe_20260612_c045/pair_probe_manifest.jsonl`
- Positive rows: `8`
- Negative rows: `15`
- Excluded rows: `7` same-but-noisy or unclear rows

## Results

| feature | positive mean | negative mean | margin | AUC | midpoint acc | decision |
|---|---:|---:|---:|---:|---:|---|
| QwenVL pooled | 0.883909 | 0.817701 | 0.066209 | 0.791667 | 0.652174 | pass |
| SigLIP pooled | 0.923141 | 0.907268 | 0.015873 | 0.650000 | 0.608696 | fail |
| PE pooled | 0.925392 | 0.880619 | 0.044773 | 0.783333 | 0.695652 | fail |
| SigLIP layer -6 pooled | 0.632742 | 0.624929 | 0.007813 | 0.491667 | 0.521739 | fail |
| SigLIP layer -6 mean_max_token | 0.765671 | 0.736944 | 0.028728 | 0.708333 | 0.695652 | fail |
| SigLIP layer -6 topk_token | 0.994048 | 0.996207 | -0.002159 | 0.291667 | 0.304348 | fail |

The pass gate is margin `>= 0.05` and AUC `>= 0.70`.

## Interpretation

QwenVL pooled is the first raw feature to pass the reviewed identity proxy gate.
This does not prove generation quality, but it is a useful decision change:
QwenVL pooled should be used as the priority metric for candidate ranking and
larger reviewed identity mining.

SigLIP layer -6 `mean_max_token` did not reproduce the c042 promising signal on
the larger seed. PE pooled is close on margin but misses the `0.05` threshold.

## Decision

`qwenvl_pooled_passes_small_reviewed_identity_proxy`

c045 is still small and positive rows are partly concentrated around recurring
characters. It is enough to justify a QwenVL-ranked expansion loop, not enough
to start IP-Adapter K/V training.

## Next

1. Use QwenVL pooled similarity to rank a broader same-page/near-page candidate
   pool.
2. Select high-confidence positives and hard negatives for a larger manual
   review sheet.
3. Repeat the feature gate on a larger reviewed set before training any
   adapter or metric head.
