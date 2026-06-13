from __future__ import annotations

import json

from tools.siglip_auto_caption_types import JsonObject


def build_plan(inventory: JsonObject) -> str:
    metrics = inventory["key_metrics"]
    return "\n".join(
        (
            "# C100 local real-color direct-green positive acquisition plan",
            "",
            "C099에서 local real-color direct-green/non-human confirmed positive가 `0`개로 확인되었다. C100은 학습 루프가 아니라 C101 학습을 시작해도 되는지 가르는 후보 확보/검수 게이트다.",
            "",
            "## 실행 경계",
            "",
            "- 학습 실행 없음",
            "- ComfyUI 생성 없음",
            "- checkpoint 생성 없음",
            "- clean32 heldout id 후보 제외",
            "- c074 external source와 c097 synthetic source는 local-real greenlight 증거로 세지 않는다.",
            "",
            "## 입력 상태",
            "",
            f"- C099 decision: `{metrics['c099_decision']}`",
            f"- c066 direct-green confirmed positives: `{metrics['c066_direct_green_positive_count']}`",
            f"- c066 total candidates: `{metrics['c066_total_candidates']}`",
            "",
            "## C101 greenlight 기준",
            "",
            f"- reviewed local positives >= `{metrics['min_reviewed_positive']}`",
            "- heldout leakage = `0`",
            "- missing paths = `0`",
            "- 후보 review sheet와 manifest가 존재해야 한다.",
            "",
        )
    )


def build_report(summary: JsonObject, inventory: JsonObject) -> str:
    return "\n".join(
        (
            "# C100 local real-color direct-green positive acquisition",
            "",
            f"- decision: `{summary['decision']}`",
            f"- candidate_rows: `{summary['candidate_rows']}`",
            f"- local_real_candidate_rows: `{summary['local_real_candidate_rows']}`",
            f"- reviewed_local_positive_count: `{summary['reviewed_local_positive_count']}`",
            f"- review_required_count: `{summary['review_required_count']}`",
            f"- heldout_leakage_count: `{summary['heldout_leakage_count']}`",
            f"- missing_path_count: `{summary['missing_path_count']}`",
            "",
            "## source buckets",
            "",
            json.dumps(summary["source_bucket_counts"], ensure_ascii=False, indent=2),
            "",
            "## 판단",
            "",
            str(summary["blocker_reason"] or summary["next_c101_command_surface"]),
            "",
            "## inventory",
            "",
            json.dumps(inventory["key_metrics"], ensure_ascii=False, indent=2),
            "",
        )
    )
