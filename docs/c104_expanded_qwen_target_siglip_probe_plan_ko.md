# C104 expanded Qwen-target SigLIP hard-shape probe 계획

## 목적

C103에서 C102 local-real positive QA가 0건이고 C098 SigLIP encoder LoRA가 not promoted로 끝난 것을 확인했다. 따라서 C104는 바로 긴 학습으로 들어가지 않고, C097 hard-shape 56쌍 전체에서 SigLIP feature 공간이 positive target과 explicit negative를 충분히 분리하는지 먼저 확인한다.

## 입력 데이터

- 원본 manifest: `training/manifests/c097_siglip_hard_shape_expanded_pairs_20260613.jsonl`
- 이미지 root: `.tmp/c097_siglip_hard_shape_expanded_root`
- C097 요약: selected_rows 56, explicit_negative_rows 56, heldout_rows_used 0
- 비교 기준:
  - C098 best mean uplift: `0.0865313863`
  - C087 Qwen baseline mean uplift: `0.1089544056`

## 실험 방식

1. C097 각 row를 positive token-probe row와 negative token-probe row로 확장한다.
2. 기존 `tools/score_siglip_token_pair_probe.py`를 사용해 SigLIP2 hidden-token similarity를 계산한다.
3. pooled, mean_max_token, topk_token 중 가장 큰 separation margin과 pairwise AUC를 본다.
4. best margin이 Qwen baseline 기준과 C098 기준을 넘고 AUC가 0.85 이상일 때만 다음 C105 소규모 학습으로 진행한다.

## 중단 조건

- referenced image path가 하나라도 없으면 `blocked_missing_required_inputs`로 취급한다.
- finite metric이 아니거나 best margin이 threshold를 넘지 못하면 C104 branch는 학습으로 진행하지 않는다.
- C104 결과는 연구 일지에 기록하고, 커밋/푸시 후 다음 루프를 결정한다.
