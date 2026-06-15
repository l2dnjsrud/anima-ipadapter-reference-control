# c076 paired direct-green source expansion plan

## 목적

c075는 런타임과 ComfyUI 노드가 정상 작동했지만, `blend_species_face`보다 약했고 direct-green focus에서도 PE/QwenVL uplift가 음수 또는 낮았다. 그래서 같은 c074 10장과 같은 calibrator-only 신호를 반복하지 않고, 다음 훈련 전에 **더 넓은 paired/direct-green/non-human target-positive 데이터가 실제로 있는지**를 먼저 검증한다.

## 후보 소스

- `CyberHarem/neeko_leagueoflegends`: c074에서 이미 10장 target-positive로 수동 확인된 seed. 단, NFA/source-rights caution 때문에 raw 이미지는 commit하지 않는다.
- `Wenaka/anima-ip-adapter-dataset`: 공개 dataset이지만 row probe가 HTTP 500이었고 라이선스/스키마가 불명확하다. source manifest에는 blocked/unknown으로 기록한다.
- `mrzjy/AniGamePersonaCaps`: 캐릭터 metadata와 cached image row가 있어 bounded row probe 대상으로 둔다.
- `mrzjy/AnimeMangaCharacters-247K`: 이미지 URL과 캐릭터 metadata가 있어 direct-green/non-human keyword 후보를 찾는다.
- `alfredplpl/anime-with-caption-cc0`: CC0 caption/image row라 안전성이 상대적으로 좋지만, row probe에서 실제 green non-human 후보가 충분히 나오는지 확인해야 한다.
- `CaptionEmporium/furry-e621-safe-llama3.2-11b`: non-human caption 밀도는 높지만 row에 직접 image URL이 없을 수 있어 metadata-only source로 기록한다.

## 경계

- heldout manifest는 후보 ID 차단용으로만 읽고, 학습/negative pair로 사용하지 않는다.
- 외부 원본 이미지는 `.tmp/c076_paired_direct_green_source_expansion/`에만 둔다.
- commit 대상은 source manifest, candidate metadata, download/review manifest, summary/report/contact-sheet 경로 기록, 테스트, 문서뿐이다.
- 대용량 다운로드는 금지하고 candidate당 4 MiB cap, 최대 8개 신규 다운로드로 제한한다.

## 라벨 스키마

- `target_positive`: 이미 수동 검수된 direct-green/non-human target-positive. c076에서 새 source가 이 라벨을 받으려면 사람이 contact sheet를 보고 확인해야 한다.
- `useful_proxy_non_human`: metadata나 색/형상 proxy는 있지만 target-positive 학습에는 아직 부족한 후보.
- `guard_false_positive_human`
- `guard_false_positive_background_object`
- `reject_unclear`

## 다음 훈련 gate

c076이 `ready_for_c077_training`이 되려면 기존 c074 seed를 제외하고 새 target-positive가 최소 12개 이상, 전체 unique target-positive가 24개 이상이어야 한다. 그 전에는 c055-init full adapter continuation이나 SigLIP/QwenVL feature checkpoint 학습을 시작하지 않고, source expansion/manual labeling을 계속한다.
