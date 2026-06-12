# IP-Adapter 다음 방향 결정

작성일: 2026-06-12
범위: `/home/wktwin/anima-ipadapter-reference-control`

## 결론

SELECTED: `agentic_reference_control_loop` -> `train_stronger_encoder`

현재 SigLIP attribute-reference recipe는 ComfyUI에서 실행 가능하고, 넓은 의미의 색감/의상/표정/무협 스타일 신호는 가져온다. 하지만 c035 32-case 단일 캐릭터 검증에서 metric gate와 identity/distinctive-trait gate를 통과하지 못했다. 따라서 지금 체크포인트를 “고퀄 reference-control 모델”로 패키징하는 것은 이르다.

반대로 `stop_siglip_branch`로 완전히 중단하기도 이르다. 한 장 overfit은 성공했고, PE-style query patch, PE-space 초기화, retrieval loss, 자동 속성 프롬프트를 거치며 실제 개선 신호가 있었다. 다음 방향은 현재 recipe를 실험용으로 보존하면서, 더 강한 이미지 인코더/정렬 학습을 준비하는 것이다.

중요: generic prompt reference-only is not solved.

2026-06-12에 확인한 `zhengdian1/InterleaveThinker`와 `zlab-princeton/i1`는 이 결론을 보강한다. InterleaveThinker는 planner/critic 방식의 외부 제어 루프 근거로 유용하고, i1은 open T2I data/training recipe 근거로 유용하다. 다만 둘 다 Anima IP-Adapter checkpoint 구조의 직접 대체재는 아니다.

공식 확인 출처:

- `https://github.com/zhengdian1/InterleaveThinker`, 확인 SHA `440c1b879cd4913b0382761f7bfa8297a32dc7d6`
- `https://github.com/zlab-princeton/i1`, 확인 SHA `cd6a34fd8e7fa7a0b7de36ff4602363e607f8a72`

## 선택지 판단

| 선택지 | 판단 | 이유 |
| --- | --- | --- |
| `package_current_recipe` | rejected | c035 best uplift `+0.0577`, improved rate `0.65625`로 metric gate 실패. identity/distinctive trait `16/32`로 visual gate 실패. |
| `agentic_reference_control_loop` | chosen first | InterleaveThinker식 planner/critic 판단을 직접 모델 구조가 아니라 평가/수정 루프로 가져온다. c035 실패 케이스마다 prompt, reference, output, visual audit를 묶어 다음 route를 결정한다. |
| `train_stronger_encoder` | chosen after audit loop | broad style/palette/costume 신호는 있으나, frozen SigLIP2 adapter-only로 identity/props/non-human trait가 부족하다. 더 강한 encoder 또는 image-encoder adaptation이 필요하다. |
| `stop_siglip_branch` | rejected | one-image overfit, c034 small-suite pass, c035 broad-style pass가 있어 구조적으로 불가능하다고 볼 근거는 부족하다. |

## 검증 요약

| 경로 | 목표 | 결과 | 현재 상태 |
| --- | --- | --- | --- |
| PE-Core baseline | 작동 가능한 reference-control baseline 확보 | ComfyUI contact-sheet에서 best scale `1.0`, mean uplift `+0.0937`, improved rate `87.5%` | baseline으로 유지 |
| SigLIP c034 | 8-case 자동 속성 프롬프트 검증 | `siglip_ref_retrieval_w14` uplift `+0.1452`, 7/8 개선. PE metric caveat 있음 | 가능성 확인용 |
| SigLIP c035 | 32-case single-character suite 검증 | best uplift `+0.0577`, improved rate `0.65625`, identity gate `16/32` | not ready |
| QwenVL c036 metric probe | Qwen3-VL pooled image embedding이 c035 판단과 맞는지 확인 | `siglip_ref_retrieval_w14` uplift `+0.0446`, improved rate `0.90625`, 그러나 identity-fail row uplift가 identity-pass보다 높음 | auxiliary metric only |
| Identity feature c037 | PE/QwenVL/SigLIP2 pooled feature가 약한 identity-positive/negative pair를 분리하는지 확인 | 세 encoder 모두 AUC `< 0.60`, margin `< 0.05` | pooled identity feature not ready |
| Strict panel c038 | duplicate panel sanity control에서 feature pipeline 확인 | QwenVL/SigLIP2/PE pooled와 SigLIP `mean_max_token` 모두 duplicate panel을 분리 | sanity pass, identity unsolved |
| Candidate review c039 | duplicate 제외 same-page 후보가 true same-character 후보로 충분한지 확인 | 후보 sheet는 생성 가능하지만 다른 인물/배경/소품 노이즈가 많음 | needs character filtering |
| Character filter c040 | QwenVL image-text score로 character-centered 후보만 남기는지 확인 | 24개 중 14개 유지, 노이즈 감소. 그러나 다른 인물/몸통 crop이 남음 | needs reviewed labels |
| Reviewed identity c041 | c040 후보에 수동 시각 라벨을 붙여 true identity seed가 충분한지 확인 | 14개 중 same 6, different 3, unclear 5, usable positive 4 | seed only, not training gate |
| Reviewed seed feature c042 | c041 seed에서 feature가 same/different character를 분리하는지 확인 | QwenVL pooled margin 0.024/AUC 0.667, SigLIP -6 mean_max margin 0.043/AUC 0.917 | gate not passed |
| Broad face filter c043 | same-page 후보를 160개로 넓히고 QwenVL face/upper-body filter로 리뷰 후보 확장 | threshold 0.08 기준 30개 유지, 22개 SG page 분산. identity label 자동 확정은 아님 | needs manual labels |
| Reviewed face seed c044 | c043 후보 30쌍을 수동 identity 라벨링 | same 12, different 15, unclear 3, usable positive 8 | seed expanded but small |
| Reviewed face feature c045 | c044 seed에서 feature separation 재검증 | QwenVL pooled margin 0.066/AUC 0.792로 pass, 다른 feature는 fail | use QwenVL for ranking |
| QwenVL-ranked candidates c046 | 전체 same-page 후보 372쌍을 face-filter 후 QwenVL pooled로 ranking | face-filtered 65쌍, top40은 27개 SG page 분산, top20까지 precision 좋음 | label top20 next |
| QwenVL top20 review c047 | c046 top20 수동 라벨링 | same 18, unclear 2, usable positive 14 | high positive yield |
| Combined seed gate c048 | c044 hard negatives + c047 positives로 QwenVL gate 반복 | 18 positive / 15 negative, margin 0.088, AUC 0.907 | QwenVL gate stable |
| QwenVL adapter-only | QwenVL embedding을 adapter에 직접 연결 | 출력 변화는 있으나 generic wuxia/interior collapse | 현재 방식 보류 |
| line-art colorization | IP-Adapter 단독 선화 채색 | 색/스타일 압력은 있으나 구조 보존 실패. EasyControl 결합 필요 | 별도 spatial-control track |
| InterleaveThinker | agentic interleaved generation 연구 | planner/critic loop가 출력 편차를 찾고 지시를 수정한다 | reference-control audit loop 참고 |
| i1 | fully open T2I recipe | data processing, recaptioning, 1024 T2I 학습/추론 구조 공개 | data/backbone recipe 참고 |

## c035 근거

산출물:

- `eval/siglip_runtime_quality_20260612_c035_suite_v1/contact_sheet.jpg`
- `eval/siglip_runtime_quality_20260612_c035_suite_v1/report.md`
- `eval/siglip_runtime_quality_20260612_c035_suite_v1/pe_similarity_metrics.json`
- `eval/siglip_runtime_quality_20260612_c035_suite_v1/visual_audit.json`

수치:

| variant | mean uplift | improved rate | decision |
| --- | ---: | ---: | --- |
| `siglip_kv_init_w14` | +0.0292 | 0.65625 | fail |
| `siglip_ref_retrieval_w14` | +0.0577 | 0.65625 | fail |

시각 감사:

- palette/costume/expression/framing: `31/32`
- identity/distinctive trait: `16/32`
- non-human/special trait: `0/1`
- 주요 실패: 검은 장발 무협 캐릭터, 보라색/밤 궁전 배경, 붉은 눈, generic official/elder template으로 수렴

## 다음 학습 방향

지금 바로 장기 학습을 시작하지 않는다. 먼저 아래 prerequisite를 만족해야 한다.

1. Agentic audit loop
   - c035처럼 이미 생성된 no-IP/IP output, reference image, prompt, metric, visual audit를 하나의 manifest로 묶는다.
   - critic 출력은 누락 identity, 누락 palette, 누락 prop, non-human trait, prompt patch, model route를 구조화한다.
   - 이 결과가 `prompt_patch` 문제인지, pair mining 문제인지, encoder adaptation 문제인지 먼저 나눈다.

2. 데이터
   - color dataset 기반 단일 캐릭터 검증 suite를 더 확장한다.
   - 같은 캐릭터 그룹 또는 identity-positive/negative pair를 mining한다.
   - i1식 data processing/recaptioning recipe를 참고해 이미지-캡션 품질과 long prompt coverage를 올린다.
   - Wenaka dataset을 쓸 경우, 다운로드 전 명시 승인과 ref/tgt pairing rule이 필요하다.

3. 모델
   - frozen SigLIP2 adapter-only를 반복하지 않는다.
   - 후보는 anime/manhwa 특화 SigLIP/PE-like encoder, QwenVL feature calibrator, 또는 image encoder LoRA/adaptation이다.
   - c036/c037 기준 QwenVL/SigLIP2/PE pooled embedding은 identity gate와 충분히 맞지 않으므로, pooled cosine 하나를 주 loss나 pass/fail gate로 쓰지 않는다.
   - c038 strict duplicate sanity는 통과했으므로 feature pipeline은 유지하되, true same-character pair에서 다시 검증한다.
   - FaceID-like 방향은 실사 InsightFace를 그대로 믿지 말고 manhwa character metric model을 먼저 검증한다.

4. 평가
   - c035와 같은 32-case 이상 single-character suite를 기본 gate로 둔다.
   - gate는 metric과 visual audit를 같이 본다.
   - 통과 기준은 최소 best uplift `>= +0.10`, improved rate `>= 0.75`, identity/distinctive trait `>= 18/32`이다.

## c036 QwenVL metric probe

산출물:

- `tools/score_auto_caption_qwenvl_metrics.py`
- `tests/test_score_auto_caption_qwenvl_metrics.py`
- `eval/qwenvl_metric_probe_20260612_c036_c035/qwenvl_similarity_metrics.json`
- `eval/qwenvl_metric_probe_20260612_c036_c035/report.md`

수치:

| variant | mean uplift | improved rate |
| --- | ---: | ---: |
| `siglip_kv_init_w14` | +0.0422 | 0.84375 |
| `siglip_ref_retrieval_w14` | +0.0446 | 0.90625 |

결론은 `qwenvl_pooled_metric_auxiliary_only`다. QwenVL pooled embedding은 broad style/palette/composition similarity에는 더 낙관적으로 반응하지만, c035 visual audit의 identity/distinctive-trait pass/fail과는 맞지 않았다. 다음 stronger-encoder 실험은 identity-positive/negative pair를 먼저 만들고, QwenVL/SigLIP/PE feature가 같은 캐릭터와 다른 캐릭터를 실제로 분리하는지 확인한 뒤 진행한다.

## c037 Identity feature probe

산출물:

- `tools/build_identity_pair_probe_manifest.py`
- `tools/image_feature_embedders.py`
- `tools/score_identity_pair_probe.py`
- `tests/test_identity_feature_probe.py`
- `eval/identity_feature_probe_20260612_c037/report.md`

현재 color dataset에서 같은 SG 폴더를 positive proxy, 다음 SG 폴더를 negative proxy로 삼아 64/64 pair를 만들고 PE, Qwen3-VL, SigLIP2 pooled image feature를 비교했다.

| encoder | positive mean | negative mean | margin | pairwise AUC | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| PE | 0.8560 | 0.8404 | 0.0156 | 0.5806 | `feature_not_sufficiently_separated` |
| Qwen/Qwen3-VL-Embedding-2B | 0.7893 | 0.7567 | 0.0326 | 0.5913 | `feature_not_sufficiently_separated` |
| SigLIP2 base patch16 512 | 0.8932 | 0.8800 | 0.0132 | 0.5759 | `feature_not_sufficiently_separated` |

결론은 `pooled_identity_feature_not_ready`다. pooled cosine 하나를 primary identity loss나 pass/fail gate로 쓰면 broad style 유사도에 속을 가능성이 크다. 다음 루프는 더 엄격한 same-character pair mining, token/layer feature probe, 또는 작은 metric head/calibrator 학습을 먼저 진행한다.

## c038 Strict panel feature probe

산출물:

- `tools/build_strict_panel_pair_probe_manifest.py`
- `tools/score_siglip_token_pair_probe.py`
- `tools/token_pair_probe_metrics.py`
- `tests/test_strict_identity_probe.py`
- `eval/strict_identity_feature_probe_20260612_c038/report.md`

positive는 같은 panel key의 v4/v5 duplicate crop이고, negative는 같은 `SG-*` 폴더 안의 다른 panel key다. 이 probe는 캐릭터 identity benchmark가 아니라 feature pipeline sanity control이다.

| encoder/metric | margin | pairwise AUC | decision |
| --- | ---: | ---: | --- |
| Qwen3-VL pooled | +0.2061 | 1.0000 | pass |
| SigLIP2 pooled | +0.1058 | 1.0000 | pass |
| PE pooled | +0.1404 | 0.9998 | pass |
| SigLIP2 `mean_max_token` | +0.3170 | 1.0000 | pass |
| SigLIP2 layer `-6` pooled | +0.4739 | 0.9998 | pass |

결론은 `strict_duplicate_feature_sanity_pass_identity_unsolved`다. feature pipeline은 near-duplicate crop을 분리할 수 있지만, 이것은 true character identity 제어가 아니다. 다음 루프는 duplicate crop을 제외한 true same-character positive와 같은 장면/스타일 hard negative를 만들고, SigLIP layer `-6` pooled 및 `mean_max_token` 후보를 다시 검증해야 한다.

## c039 True identity candidate review

산출물:

- `tools/build_true_identity_candidate_review.py`
- `tests/test_true_identity_candidate_review.py`
- `eval/true_identity_candidate_review_20260612_c039/candidate_pairs.jsonl`
- `eval/true_identity_candidate_review_20260612_c039/candidate_sheet.jpg`
- `eval/true_identity_candidate_review_20260612_c039/report.md`

같은 `SG-page` 안의 non-duplicate panel pair 24개를 뽑아 sheet로 확인했다. 같은 장면 후보를 빠르게 모으는 데는 유용하지만, 다른 인물, 배경, 소품, 다인물 panel이 많이 섞여 true same-character positive로 자동 확정하기 어렵다.

결론은 `same_page_candidates_need_character_filtering`이다. 다음 루프는 후보 양쪽이 모두 캐릭터 중심인지 먼저 거르고, 그 뒤 same-character 여부를 라벨링 가능한 sheet로 유지하는 것이다.

## c040 Character-filtered identity candidates

산출물:

- `tools/filter_character_candidate_pairs.py`
- `tests/test_character_candidate_filter.py`
- `eval/character_filtered_identity_candidates_20260612_c040/scored_candidate_pairs.jsonl`
- `eval/character_filtered_identity_candidates_20260612_c040/kept_candidate_pairs.jsonl`
- `eval/character_filtered_identity_candidates_20260612_c040/kept_candidate_sheet.jpg`
- `eval/character_filtered_identity_candidates_20260612_c040/report.md`

Qwen3-VL image-text retrieval로 character-centered score를 계산했다. 양쪽 crop의 score가 모두 `>= 0.15`인 pair만 유지했을 때 24개 중 14개가 남았다.

결론은 `character_filter_reduces_noise_not_identity_labels`다. 배경/소품 노이즈를 줄이는 데는 도움이 되지만, 남은 후보도 다른 인물이나 몸통 crop이 섞여 true same-character positive로 자동 승격할 수 없다. 다음 loop는 kept sheet에 `same_character`, `different_character`, `unclear` 라벨을 붙이는 reviewed manifest를 만드는 것이다.

## c041 Reviewed identity candidates

산출물:

- `tools/build_reviewed_identity_manifest.py`
- `tests/test_reviewed_identity_manifest.py`
- `eval/reviewed_identity_candidates_20260612_c041/manual_visual_labels.jsonl`
- `eval/reviewed_identity_candidates_20260612_c041/reviewed_candidate_pairs.jsonl`
- `eval/reviewed_identity_candidates_20260612_c041/usable_positive_pairs.jsonl`
- `eval/reviewed_identity_candidates_20260612_c041/different_character_pairs.jsonl`
- `eval/reviewed_identity_candidates_20260612_c041/reviewed_candidate_sheet.jpg`
- `eval/reviewed_identity_candidates_20260612_c041/report.md`

c040 kept 14개를 보수적으로 라벨링했다. 결과는 `same_character=6`, `different_character=3`, `unclear=5`, `positive_usable=4`다.

결론은 `reviewed_seed_too_small_for_training_gate`다. 이 seed는 feature sanity probe에는 쓸 수 있지만 학습을 시작하기에는 너무 작다. 다음 loop는 c041 seed로 SigLIP/QwenVL/PE feature separation을 확인하면서, 더 넓은 face/upper-body mining으로 usable positive를 늘리는 것이다.

## c042 Reviewed seed feature probe

산출물:

- `tools/build_reviewed_pair_probe_manifest.py`
- `tests/test_reviewed_pair_probe_manifest.py`
- `eval/reviewed_seed_feature_probe_20260612_c042/pair_probe_manifest.jsonl`
- `eval/reviewed_seed_feature_probe_20260612_c042/siglip_pooled_report.md`
- `eval/reviewed_seed_feature_probe_20260612_c042/siglip_layer_m6_token_report.md`
- `eval/reviewed_seed_feature_probe_20260612_c042/qwenvl_pooled_report.md`
- `eval/reviewed_seed_feature_probe_20260612_c042/pe_pooled_report.md`
- `eval/reviewed_seed_feature_probe_20260612_c042/report.md`

결론은 `reviewed_seed_feature_gate_not_passed`다. c041의 4 positive / 3 negative seed에서 기존 pass 기준 margin `>= 0.05`, AUC `>= 0.70`을 동시에 만족한 feature는 없다.

가장 볼 만한 후보는 SigLIP layer `-6` `mean_max_token`이다. AUC `0.916667`이지만 margin `0.043225`로 기준 미달이고 seed가 너무 작으므로 통과로 보지 않는다. 다음 loop는 reviewed positive를 수십 개로 늘린 뒤 같은 metric을 다시 확인하는 것이다.

## c043 Broad face/upper-body identity candidate mining

산출물:

- `tools/filter_face_upper_body_candidates.py`
- `tests/test_face_upper_body_candidate_filter.py`
- `eval/broad_identity_candidate_mining_20260612_c043/raw_candidate_pairs.jsonl`
- `eval/broad_identity_candidate_mining_20260612_c043/scored_face_candidates.jsonl`
- `eval/broad_identity_candidate_mining_20260612_c043/kept_face_candidates.jsonl`
- `eval/broad_identity_candidate_mining_20260612_c043/kept_face_candidate_sheet.jpg`
- `eval/broad_identity_candidate_mining_20260612_c043/report.md`

결론은 `face_upper_body_filter_expands_review_pool_not_identity_labels`다. raw same-page 후보 `160`개 중 QwenVL face/upper-body score threshold `0.08`로 `30`개를 남겼고, kept set은 `22`개 SG page에 분산됐다.

이 단계는 c041의 4개 usable positive seed를 바로 학습에 쓰지 않기 위한 데이터 확장이다. contact sheet 기준 얼굴/상반신 crop 비율은 좋아졌지만 동일 인물/다른 인물 라벨은 아직 자동 확정할 수 없다. 다음 loop는 이 30쌍을 reviewed manifest로 바꾸고, 그 뒤 c042 feature probe를 더 큰 seed에서 반복하는 것이다.

## c044-c045 Reviewed face seed and feature gate

산출물:

- `eval/reviewed_face_identity_candidates_20260612_c044/report.md`
- `eval/reviewed_face_seed_feature_probe_20260612_c045/report.md`

c044에서 c043 kept 30쌍을 보수적으로 라벨링했다. 결과는 `same_character=12`, `different_character=15`, `unclear=3`, `positive_usable=8`이다.

c045에서 c042 feature probe를 반복했다. QwenVL pooled가 margin `0.066209`, AUC `0.791667`로 처음 reviewed identity proxy gate를 통과했다. SigLIP pooled, PE pooled, SigLIP layer -6 token metric은 통과하지 못했다.

결론은 `qwenvl_pooled_passes_small_reviewed_identity_proxy`다. 이 결과는 QwenVL pooled를 더 큰 후보 ranking metric으로 쓸 수 있다는 뜻이지, IP-Adapter 생성 품질이 해결됐다는 뜻은 아니다. 다음 loop는 QwenVL pooled로 broader candidate pool을 rank하고, 더 큰 reviewed identity manifest를 만든다.

## c046 QwenVL-ranked identity candidate mining

산출물:

- `tools/rank_identity_candidate_pairs.py`
- `tests/test_rank_identity_candidate_pairs.py`
- `eval/qwenvl_ranked_identity_candidates_20260612_c046/report.md`

전체 same-page non-duplicate 후보 `372`개를 만들고, c043 face/upper-body filter threshold `0.08`로 `65`개를 남긴 뒤 QwenVL pooled similarity로 ranking했다. top40은 `27`개 SG page에 분산됐고, top10 similarity range는 `0.9155` to `0.9632`, top20 lower bound는 `0.8801`이다.

결론은 `qwenvl_ranking_improves_candidate_precision_top20`이다. top10은 대부분 clean same-character pair였고 top20까지 리뷰 효율이 좋다. rank 21 이후부터는 다른 인물, group panel, back/partial crop 노이즈가 늘어난다. 다음 loop는 top20을 먼저 라벨링한다.

## c047-c048 QwenVL reviewed top20 and combined gate

산출물:

- `eval/qwenvl_top20_reviewed_identity_20260612_c047/report.md`
- `eval/qwenvl_combined_seed_feature_probe_20260612_c048/report.md`

c047에서 c046 top20을 라벨링했다. 결과는 `same_character=18`, `different_character=0`, `unclear=2`, `positive_usable=14`다.

c048에서 c044 hard negatives와 c047 positives를 결합했다. combined seed는 reviewed rows `44`, positive pairs `18`, negative pairs `15`다. QwenVL pooled는 margin `0.087629`, AUC `0.907407`, midpoint accuracy `0.818182`로 통과했다.

결론은 `qwenvl_pooled_identity_gate_stable_on_combined_seed`다. 이제 QwenVL pooled는 reviewed identity 후보 ranking/gating metric으로 채택한다. 아직 generation 품질 통과가 아니라 데이터/metric gate 통과이므로, 다음 단계는 더 큰 reviewed identity set 구축이다.

## 실행 명령

현재 단계에서는 full training command를 실행하지 않는다. 다음에 할 일은 데이터/encoder 준비 검증이다.

새 agentic audit loop의 상세 계획은 `docs/ipadapter_agentic_reference_control_plan_ko.md`에 둔다.

```bash
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python -m pytest \
  tests/test_reference_prompting.py \
  tests/test_siglip_auto_caption_eval.py \
  tests/test_comfyui_workflows.py \
  tests/test_native_siglip.py -q
```

```bash
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python tools/validate_reference_suite.py \
  eval/siglip_reference_suite_v1_20260612/reference_suite_v1.jsonl \
  --data-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best
```

## 현재 사용 가능한 recipe

실험용으로는 다음 recipe를 유지한다.

- node family: `AnimaSigLIPIPAdapterLoader` -> `AnimaSigLIPEncodeImage` -> `AnimaSigLIPIPAdapterApply`
- 기본 checkpoint: `anima_siglip_ip_adapter_single_character_clean32_pe_retrieval_0128_20260611.safetensors`
- fallback checkpoint: `anima_siglip_ip_adapter_single_character_clean32_pe_space_init_0512_20260611.safetensors`
- runtime label: `siglip_ref_retrieval_w14`
- weight: `1.4`
- start/end: `0.0 / 0.85`
- 필수 조건: 자동 visible-attribute prompt를 같이 사용

이 recipe는 “품질 좋은 실험 경로”이지, 아직 “바로 믿고 쓰는 완성 모델”은 아니다.
