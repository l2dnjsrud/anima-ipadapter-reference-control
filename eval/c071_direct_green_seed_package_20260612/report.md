# c071 Direct-Green Manual Seed Package

c071은 학습 실험이 아니라 data gate 실험이다. c068-c070의 proxy/guard 후보를 합쳐 수동/외부 라벨링 패키지로 만들고, 사람이 확정한 `target_positive`만 다음 encoder training으로 보낼 수 있게 했다.

## Result

- Source rows: c068 `48`, c069 `48`, c070 `36`
- Raw candidate rows: `132`
- Unique candidates: `84`
- Heldout rows used: `0`
- Missing paths: `0`
- Label schema: `target_positive`, `useful_proxy_non_human`, `guard_false_positive_human`, `guard_false_positive_background_object`, `reject_unclear`
- Suggested labels: `useful_proxy_non_human=29`, `guard_false_positive_background_object=40`, `guard_false_positive_human=15`
- Example import target positives: `0`
- Decision: `external_manual_data_required`

## Decision

Do not train yet. The importer promotes the package only when a manual label file contains at least 4 unique `target_positive` rows and no heldout/unknown/duplicate-positive violations.
