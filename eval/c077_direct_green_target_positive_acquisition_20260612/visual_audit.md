# c077 visual audit

## 판정

`manual_needed_more_target_positives`

c077은 c074 seed 10장과 신규 CyberHarem sample asset 후보 36장을 contact sheet로 검수했다. 신규 후보는 다운로드/시트 생성에는 성공했지만, direct-green/non-human target-positive로 승격할 이미지는 없었다.

## 라벨 요약

- `target_positive`: 10
- `guard_false_positive_human`: 8
- `useful_proxy_non_human`: 28
- 신규 `target_positive`: 0

## 시각 판단

- `CyberHarem/green_heart_azurlane`: 초록 머리/의상 중심의 인간형 캐릭터다. green-name source지만 green skin/non-human trait가 아니므로 guard false positive로 둔다.
- `CyberHarem/poppy_leagueoflegends`, `tristana_leagueoflegends`, `lulu_leagueoflegends`: yordle/non-human proxy로는 유용하지만 direct-green skin target-positive는 아니다.
- `CyberHarem/soraka_leagueoflegends`: horned non-human proxy지만 피부/색상 조건이 direct-green target-positive 기준을 만족하지 않는다.
- `CyberHarem/nami_leagueoflegends`: mermaid/non-human proxy지만 blue/teal 계열이며 target-positive로 쓰기에는 목표와 다르다.

## 결론

c077은 "후보 source가 아예 없는 상태"는 벗어났지만, 고품질 direct-green reference-control 학습에 필요한 신규 양성 샘플 수를 채우지 못했다. 다음 루프는 checkpoint training이 아니라 더 강한 source acquisition, 사용자 제공 샘플, 또는 synthetic/direct-green bootstrap source를 검토해야 한다.
