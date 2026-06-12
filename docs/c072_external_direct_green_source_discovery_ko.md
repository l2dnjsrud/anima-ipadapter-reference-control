# c072 외부 Direct-Green / Non-Human 후보 소스 탐색

## 목적

c071에서 내부 데이터셋 자동 탐색만으로는 direct-green/non-human `target_positive`를 4장 이상 확보하지 못했다. c072의 목적은 바로 학습을 재개하는 것이 아니라, 외부 공개 데이터셋에서 다음 학습으로 넘길 수 있는 후보 소스가 있는지 확인하는 것이다.

중요한 안전 조건은 세 가지다.

- 대용량 데이터셋을 무작정 다운로드하지 않는다.
- clean32 heldout 8장은 절대 후보/학습에 섞지 않는다.
- 메타데이터 기반 후보를 자동으로 `target_positive`로 승격하지 않는다.

## 조사한 소스

조사는 Hugging Face Dataset API와 Dataset Viewer API를 사용했다. 각 소스는 repo metadata와 train split의 작은 row probe만 읽었다.

| 소스 | 라이선스 메모 | 접근/스키마 판단 | c072 판단 |
| --- | --- | --- | --- |
| [Wenaka/anima-ip-adapter-dataset](https://huggingface.co/datasets/Wenaka/anima-ip-adapter-dataset) | unknown | 공개 repo지만 card/license가 없고, 이번 live rows probe는 HTTP 500으로 실패 | 원본 anima IP-Adapter 쪽 데이터로 보이나 라이선스와 스키마 불명이라 학습 후보로 쓰지 않음 |
| [mrzjy/AniGamePersonaCaps](https://huggingface.co/datasets/mrzjy/AniGamePersonaCaps) | cc-by-sa-4.0 | `image`, `title`, `description`, `image_url`, `caption` 필드 확인 | non-human/green 관련 메타데이터 후보가 많음. 다만 자동 캡션 hallucination 가능성이 있어 시각 검수 필요 |
| [mrzjy/AnimeMangaCharacters-247K](https://huggingface.co/datasets/mrzjy/AnimeMangaCharacters-247K) | cc-by-4.0 | Fandom 캐릭터 metadata + image URL 확인 | 캐릭터 단위 후보 소스로 좋지만 direct-green 확정은 metadata만으로 부족 |
| [alfredplpl/anime-with-caption-cc0](https://huggingface.co/datasets/alfredplpl/anime-with-caption-cc0) | cc0-1.0 | image + prompt/caption 확인 | green skin, furry/anthro 계열 prompt 후보가 있음. synthetic 성격이라 reference-control 학습에는 별도 검수 필요 |
| [CaptionEmporium/furry-e621-safe-llama3.2-11b](https://huggingface.co/datasets/CaptionEmporium/furry-e621-safe-llama3.2-11b) | cc-by-sa-4.0 | caption/tag metadata는 확인되지만 viewer row에 직접 image URL이 없음 | non-human 밀도는 높지만 c071 호환 이미지 후보 패키지에는 바로 쓰지 않음 |

## 산출물

- `eval/c072_external_direct_green_source_discovery_20260612/source_manifest.jsonl`
- `eval/c072_external_direct_green_source_discovery_20260612/external_candidates.jsonl`
- `eval/c072_external_direct_green_source_discovery_20260612/external_candidate_template.csv`
- `eval/c072_external_direct_green_source_discovery_20260612/summary.json`
- `eval/c072_external_direct_green_source_discovery_20260612/report.md`

`external_candidates.jsonl`은 c071 importer가 요구하던 핵심 필드와 호환되도록 만들었다. 단, `image_path`는 아직 로컬 파일 경로가 아니라 외부 URL이고 `path_exists=false`다. 즉 지금 상태는 학습 manifest가 아니라 수동/시각 검수용 metadata package다.

## 결과

live probe 결과:

- inspected source count: `5`
- large downloads performed: `false`
- heldout rows used: `0`
- source candidate counts:
  - Wenaka/anima-ip-adapter-dataset: `0`
  - mrzjy/AniGamePersonaCaps: `63`
  - mrzjy/AnimeMangaCharacters-247K: `7`
  - alfredplpl/anime-with-caption-cc0: `21`
  - CaptionEmporium/furry-e621-safe-llama3.2-11b: `0`
- packaged metadata candidates: `12`
- confirmed `target_positive`: `0`

## 판단

c072 decision은 `external_candidates_found_manual_confirmation_required`다.

즉 외부 소스에서 direct-green/non-human 후보를 찾는 것은 가능해 보인다. 하지만 현재 후보는 metadata scoring으로 선별된 것이며, 일부는 green clothing/background, hallucinated caption, 캐릭터가 아닌 객체/로봇 후보가 섞일 수 있다. 따라서 이 결과만으로 encoder-side 학습을 재개하면 안 된다.

다음 루프는 c073으로 분리한다.

- c072 상위 후보 이미지를 소량만 로컬 scratch에 다운로드한다.
- review/contact sheet를 만든다.
- 실제 이미지 기준으로 `target_positive`, `useful_proxy_non_human`, guard label을 수동 확정한다.
- unique `target_positive >= 4`가 확인될 때만 c071 importer 또는 c073 importer를 통해 `ready_for_encoder_training` 상태로 넘긴다.
