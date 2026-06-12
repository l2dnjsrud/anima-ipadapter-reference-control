# c077 direct-green target-positive acquisition plan

## 목적

c076은 새 metadata 후보 3개를 확인했지만 신규 `target_positive`가 0개였다. c075에서 c074 seed 10장만으로 학습한 결과도 `blend_species_face`보다 약했기 때문에, 같은 10장을 반복하는 학습은 중단하고 **새 direct-green/non-human target-positive를 실제로 확보할 수 있는지**를 먼저 확인한다.

## 접근

- c074의 Neeko seed 10장은 이전 수동 검수 기준으로 carry-forward한다.
- Hugging Face `CyberHarem/*` sample asset tree를 bounded source로 사용한다.
- 후보는 `.tmp/c077_direct_green_target_positive_acquisition/` 아래에만 다운로드한다.
- 레포에는 source/candidate/download/review metadata, summary/report, manual label CSV만 남기고 raw 외부 이미지는 커밋하지 않는다.

## 후보 소스

- `CyberHarem/green_heart_azurlane`: green-named source. 실제로는 초록 머리/의상 인간형 false-positive guard일 가능성이 높다.
- `CyberHarem/poppy_leagueoflegends`: non-human/yordle proxy. direct-green skin target은 아닐 가능성이 높다.
- `CyberHarem/tristana_leagueoflegends`: non-human/yordle proxy.
- `CyberHarem/lulu_leagueoflegends`: non-human/yordle proxy.
- `CyberHarem/soraka_leagueoflegends`: horned non-human proxy. 보라/청록 계열일 가능성이 높다.
- `CyberHarem/nami_leagueoflegends`: mermaid/non-human proxy. blue/teal 계열일 가능성이 높다.
- `CyberHarem/vex_leagueoflegends`: non-human/yordle proxy. 어두운 blue/purple 계열일 가능성이 높다.

## 승격 조건

다음 학습 manifest로 넘어가려면 둘 다 만족해야 한다.

- 전체 unique `target_positive` >= 24
- c074 seed를 제외한 신규 `target_positive` >= 12

조건을 만족하지 못하면 checkpoint training을 하지 않고 `manual_needed_more_target_positives` 또는 `source_blocked_manual_needed`로 닫는다.

## 검증

- `tests/test_c077_target_positive_acquisition.py`
- `py_compile` for c077 modules
- c077 artifact consistency check
- `git diff --check`
- 연구일지 업데이트, 커밋, 푸시
