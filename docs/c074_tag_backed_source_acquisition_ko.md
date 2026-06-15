# c074 tag-backed direct-green source acquisition

## 목적

c073에서 외부 metadata 후보 12장을 실제 이미지로 검수했지만 `target_positive`가 0개였다. 따라서 c074의 목적은 학습을 강행하는 것이 아니라, tag/caption 근거가 더 강한 공개 소스에서 실제 이미지가 있는 direct-green/non-human 후보를 확보하는 것이다.

## 조사한 소스

- `CyberHarem/neeko_leagueoflegends`: MIT로 표시되어 있고, core tags에 `green_skin`, `colored_skin`, `tail`, `monster_girl`가 명시되어 있다. Dataset Viewer는 500으로 실패했지만 repo에 sample PNG가 직접 존재한다. `not-for-all-audiences` 표시와 원천 이미지 저작권 caveat가 있으므로 재배포/공개 학습 전 권리 검토가 필요하다.
- `OneIG-Bench/OneIG-Bench`: `green skin`, `lizard tail` prompt는 검색되지만 prompt-only라 이미지 학습 후보로 쓰지 않았다.
- `CaptionEmporium/anime-caption-danbooru-2021-sfw-5m-hq`: tag/caption은 풍부하지만 row가 image asset을 노출하지 않고 검색이 불안정했다.
- `mrzjy/splash-art-gacha-collection-10k`: image+caption source이지만 green query 검색이 timeout되어 이번 후보 package에는 넣지 않았다.

## 실행 방식

도구는 `tools/c074_tag_backed_source_acquisition.py`다. 성적/노출 태그가 있는 Neeko cluster 0은 제외했고, cluster 1/2 sample PNG 10장만 `.tmp/c074_tag_backed_direct_green_source_acquisition/`에 소량 다운로드했다. 다운로드 이미지는 커밋하지 않고, 커밋 대상은 `eval/c074_tag_backed_direct_green_source_acquisition_20260612/` 아래의 JSONL/CSV/JSON/MD 텍스트 산출물만이다.

## 결과

- inspected sources: `4`
- candidate rows: `10`
- downloaded rows: `10`
- reviewed rows: `10`
- target_positive_confirmed_count: `10`
- minimum_target_positive_required: `4`
- heldout_rows_used: `0`
- large_downloads_performed: `false`
- committed_external_image_count: `0`
- decision: `ready_for_encoder_training`

## 다음 결정

c074는 처음으로 direct-green/non-human target-positive gate를 통과했다. 다음 루프는 c075로 넘긴다. c075는 c074 positives 10장을 바로 무작정 학습하지 말고, 기존 c071/c073 guard negatives와 섞은 작은 training manifest를 만들고, license/NFA caveat를 문서에 남긴 상태에서 bounded encoder-feature training을 진행해야 한다.
