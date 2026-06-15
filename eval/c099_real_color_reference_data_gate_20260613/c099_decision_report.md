# C099 real-color reference data gate

- decision: `c100_blocked_needs_annotation_or_teacher`
- candidate_rows: `276`
- missing_path_count: `0`
- heldout_leakage_count: `0`
- omitted_heldout_rows: `0`
- real_local_rows: `210`
- real_local_direct_green_confirmed_rows: `0`
- external_direct_green_positive_rows: `10`
- synthetic_hard_shape_rows: `56`

## 판단

real local-color direct-green/non-human character positive is still missing; use annotation or a stronger attribute teacher before C100 training.

## 인벤토리 핵심

{
  "clean32_train_rows": 32,
  "clean32_heldout_rows": 8,
  "c052_positive_pairs": 29,
  "c066_heldout_rows_used": 0,
  "c066_direct_green_positive_count": 0,
  "c097_selected_rows": 56
}
