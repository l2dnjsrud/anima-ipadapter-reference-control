# C100 local real-color direct-green positive acquisition

- decision: `c101_blocked_needs_manual_annotation_or_teacher`
- candidate_rows: `64`
- local_real_candidate_rows: `64`
- reviewed_local_positive_count: `0`
- review_required_count: `64`
- heldout_leakage_count: `0`
- missing_path_count: `0`

## source buckets

{
  "direct_green_pixel_candidate": 40,
  "pale_non_human_proxy": 11,
  "fang_profile_proxy": 13
}

## 판단

candidate sheet is ready, but local real-color positives still need manual review or a stronger attribute teacher before C101 training.

## inventory

{
  "c099_decision": "c100_blocked_needs_annotation_or_teacher",
  "c066_direct_green_positive_count": 0,
  "c066_total_candidates": 120,
  "c066_source_buckets": {
    "direct_green_pixel_candidate": 40,
    "fang_profile_proxy": 17,
    "human_negative": 20,
    "old_headwear_negative": 22,
    "pale_non_human_proxy": 11,
    "red_eye_proxy": 10
  },
  "heldout_count": 8,
  "min_reviewed_positive": 8
}
