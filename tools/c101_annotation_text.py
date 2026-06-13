from __future__ import annotations

import json

from tools.siglip_auto_caption_types import JsonObject


def build_plan(inventory: JsonObject) -> str:
    metrics = inventory["key_metrics"]
    return "\n".join(
        (
            "# C101 local positive annotation / teacher rerank plan",
            "",
            "C100은 local-real 후보 64개를 만들었지만 reviewed local positive가 `0`개라 C101 학습을 막았다. C101은 학습이 아니라 C102 학습 가능 여부를 결정하는 보수적 라벨/teacher proposal 게이트다.",
            "",
            "## 실행 경계",
            "",
            "- adapter 학습 없음",
            "- ComfyUI 생성 없음",
            "- checkpoint 생성 없음",
            "- C100 local-real 후보만 C102 greenlight 증거로 사용",
            "- clean32 heldout id 제외",
            "- label schema: `local_positive`, `local_negative`, `unclear`",
            "",
            "## 입력 상태",
            "",
            f"- C100 decision: `{metrics['c100_decision']}`",
            f"- C100 candidate rows: `{metrics['c100_candidate_rows']}`",
            f"- C100 review sheet size: `{metrics['c100_review_sheet_size']}`",
            "",
            "## C102 greenlight 기준",
            "",
            f"- reviewed local positives >= `{metrics['min_reviewed_positive']}`",
            "- reviewed rows = input candidate rows",
            "- review required count = `0`",
            "- teacher-only positive count = `0`",
            "- heldout leakage = `0`",
            "- missing paths = `0`",
            "- positive label은 direct-green/non-human prior visual review 또는 수동 확인 evidence가 있어야 한다.",
            "",
        )
    )


def build_report(summary: JsonObject, inventory: JsonObject) -> str:
    return "\n".join(
        (
            "# C101 local positive annotation / teacher rerank gate",
            "",
            f"- decision: `{summary['decision']}`",
            f"- input_candidate_rows: `{summary['input_candidate_rows']}`",
            f"- reviewed_rows: `{summary['reviewed_rows']}`",
            f"- reviewed_local_positive_count: `{summary['reviewed_local_positive_count']}`",
            f"- local_negative_count: `{summary['local_negative_count']}`",
            f"- unclear_count: `{summary['unclear_count']}`",
            f"- review_required_count: `{summary['review_required_count']}`",
            f"- teacher_only_positive_count: `{summary['teacher_only_positive_count']}`",
            f"- heldout_leakage_count: `{summary['heldout_leakage_count']}`",
            f"- missing_path_count: `{summary['missing_path_count']}`",
            "",
            "## label counts",
            "",
            json.dumps(summary["label_counts"], ensure_ascii=False, indent=2),
            "",
            "## prior review source counts",
            "",
            json.dumps(summary["review_source_counts"], ensure_ascii=False, indent=2),
            "",
            "## 판단",
            "",
            str(summary["blocker_reason"] or summary["next_c102_command_surface"]),
            "",
            "## inventory",
            "",
            json.dumps(inventory["key_metrics"], ensure_ascii=False, indent=2),
            "",
        )
    )
