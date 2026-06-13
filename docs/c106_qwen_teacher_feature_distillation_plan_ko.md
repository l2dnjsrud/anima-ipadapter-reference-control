# C106 Qwen Teacher Feature Distillation Plan

C105에서 선택한 `qwen_teacher_distillation` 경로의 첫 번째 문턱 실험으로, C097 hard-shape positive/explicit-negative 쌍을 QwenVL pair feature probe로 변환한다.

- Source / manifest / output: `C097/C087/C105` / `training/manifests/c097_siglip_hard_shape_expanded_pairs_20260613.jsonl` / `eval/c106_qwen_teacher_feature_distillation_20260613/probe_manifest.jsonl`
- Selected / pair-probe / heldout rows: `56` / `112` / `0`
- Gate: margin >= `0.05`, AUC >= `0.85`, C104 margin `0.01997534079211105` 초과
- Branches: pass `c107_bounded_qwen_teacher_distillation_training`, stop `manual_external_annotation_or_stronger_teacher_checkpoint`
