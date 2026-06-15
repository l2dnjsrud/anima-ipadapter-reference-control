# C102 stronger VLM QA teacher gate

- decision: `c103_blocked_needs_manual_annotation_or_external_teacher`
- selected_teacher_status: `local_qwen3vl_8b_instruct_runnable`
- candidate_rows: `64`
- covered_rows: `64`
- qa_positive_candidate_count: `0`
- confirmed_local_positive_count: `0`
- teacher_only_positive_count: `0`
- local_negative_count: `63`
- unclear_count: `1`
- heldout_leakage_count: `0`
- missing_path_count: `0`

## QA label counts

{
  "green_background_or_object": 12,
  "human_character": 50,
  "proxy_only": 2
}

## final label counts

{
  "local_negative": 63,
  "unclear": 1
}

## 판단

C102 QA package did not reach 8 conflict-free confirmed local positives; confirmed=0.

## inventory

{
  "c101_decision": "c102_blocked_needs_manual_annotation_or_teacher",
  "c101_reviewed_local_positive_count": 0,
  "c100_candidate_rows": 64,
  "heldout_count": 8,
  "min_confirmed_positive": 8
}
