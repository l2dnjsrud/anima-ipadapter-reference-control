# C107 Qwen Teacher Distillation Plan

C106에서 QwenVL teacher가 hard-shape positive/negative를 강하게 분리했으므로, 그 판단을 `neg_id` hard-negative contrastive 학습 manifest로 보존한다.

- Source manifest: `training/manifests/c097_siglip_hard_shape_expanded_pairs_20260613.jsonl`
- Teacher score: `eval/c106_qwen_teacher_feature_distillation_20260613/qwenvl_pair_scores.json`
- Output manifest: `training/manifests/c107_qwen_teacher_distillation_20260613.jsonl`
- Rows / explicit negatives: `56` / `56`
- Heldout rows used: `0`
- C106 teacher margin: `0.21772855200937813`
- C107 mean row margin: `0.21772855200937816`
- Decision: `c107_manifest_ready_for_bounded_training`
