# C097 hard-shape data expansion plan

## 목적

C096은 SigLIP encoder-LoRA가 ComfyUI에서 실제로 적용된다는 점은 증명했지만, 학습 데이터가
`10`개 hard-shape row에 그쳐 frog/chibi/mascot/non-human reference가 green humanoid bust로
수렴하는 문제를 깨지 못했다. C097의 목적은 같은 얕은 학습을 반복하지 않고, 다음 deeper
encoder adaptation에 넣을 수 있는 더 큰 no-heldout hard-shape pair set을 먼저 만드는 것이다.

## 입력 데이터

- source manifest: `training/manifests/c087_expanded_crop_pairs_20260613.jsonl`
- source summary: `training/manifests/c087_expanded_crop_pairs_20260613.summary.json`
- source image root: `.tmp/c087_expanded_crop_pairs_root`

C087 원본은 `224`개 crop-pair, `4`개 shape group, `heldout_rows_used=0`으로 확인됐다.

## 생성 방식

1. C087의 `ref_id`, `tgt_id`, `prompt`를 읽는다.
2. image id에서 `c082_*` shape group과 pose pair를 파싱한다.
3. 같은 shape group의 positive pair만 사용한다.
4. 학습 붕괴 방지를 위해 각 row마다 다른 shape group의 `neg_id`를 붙인다.
5. source-pose 편향을 줄이기 위해 group당 최대 `16`, pose-pair당 최대 `8`개로 제한한다.
6. `.tmp/c097_siglip_hard_shape_expanded_root`에 ref/target/negative 이미지를 materialize한다.
7. `pair_review_sheet.jpg`로 사람이 빠르게 훑을 수 있는 3열 검토 시트를 만든다.

## 통과 기준

- selected rows `>= 48`
- explicit negative rows == selected rows
- heldout rows used == `0`
- shape group 수 `>= 4`
- negative shape group이 positive shape group과 다름
- manifest에 적힌 모든 ref/target/negative 이미지가 존재함
- blank-like source image가 없음

## 이번 결과

C097 데이터 게이트는 통과했다.

- selected rows: `56`
- explicit negative rows: `56`
- materialized images: `168`
- heldout rows used: `0`
- heldout strings in manifest/root: `0`
- blank-like images: `0`
- shape groups:
  - `c082_frog_yokai_guard`: `16`
  - `c082_goblin_mage`: `16`
  - `c082_green_oni_scout`: `16`
  - `c082_jade_lizard_monk`: `8`

## 다음 결정

C097은 품질 승격 실험이 아니라 다음 학습 입력을 확보하는 데이터 게이트다. 바로 다음 루프는
C097 manifest를 사용해 C096보다 더 깊은 SigLIP encoder-LoRA 파일럿을 진행한다. 최소 다음
조건은 `layer_count=4`, `rank=8` 또는 `rank=16`, explicit negative row 전체 사용, heldout
누수 0, 그리고 C094/C095/C096/C087 baseline과 같은 hard-shape generation gate 비교다.
