# C101 local positive annotation / teacher rerank gate

- decision: `c102_blocked_needs_manual_annotation_or_teacher`
- input_candidate_rows: `64`
- reviewed_rows: `64`
- reviewed_local_positive_count: `0`
- local_negative_count: `25`
- unclear_count: `39`
- review_required_count: `0`
- teacher_only_positive_count: `0`
- heldout_leakage_count: `0`
- missing_path_count: `0`

## label counts

{
  "local_negative": 25,
  "unclear": 39
}

## prior review source counts

{
  "prior_visual_review": 36,
  "conservative_auto": 28
}

## 판단

C101 found no sufficient reviewed local direct-green/non-human positives; C102 training remains blocked until manual annotation or stronger VLM teacher confirms at least 8 positives.

## inventory

{
  "c100_decision": "c101_blocked_needs_manual_annotation_or_teacher",
  "c100_candidate_rows": 64,
  "c100_review_sheet_size": "880x4384",
  "heldout_count": 8,
  "min_reviewed_positive": 8
}
