from __future__ import annotations

import json

from tools.siglip_auto_caption_types import JsonObject


def build_plan(inventory: JsonObject) -> str:
    metrics = inventory["key_metrics"]
    return "\n".join(
        (
            "# C102 stronger VLM QA teacher gate plan",
            "",
            "C101은 64개 local-real 후보를 모두 보수적으로 라벨링했지만 `local_positive=0`이었다. C102는 학습이 아니라 로컬 생성형 VLM QA teacher로 후보를 다시 판정하는 획득 게이트다.",
            "",
            "## 실행 경계",
            "",
            "- adapter 학습 없음",
            "- ComfyUI 생성 없음",
            "- checkpoint 생성 없음",
            "- C100/C101 local-real 후보 64개만 C103 greenlight 증거로 사용",
            "- 기존 C101 `local_negative`는 VLM 단독 판단으로 뒤집지 않음",
            "",
            "## 확인한 VLM/teacher 표면",
            "",
            json.dumps(inventory["vlm_surfaces_checked"], ensure_ascii=False, indent=2),
            "",
            "## 선택한 teacher",
            "",
            f"- status: `{inventory['selected_teacher']['status']}`",
            f"- model_path: `{inventory['selected_teacher']['model_path']}`",
            f"- reason: {inventory['selected_teacher']['reason']}",
            "",
            "## 입력 상태",
            "",
            f"- C101 decision: `{metrics['c101_decision']}`",
            f"- C101 reviewed local positive: `{metrics['c101_reviewed_local_positive_count']}`",
            f"- C100 candidate rows: `{metrics['c100_candidate_rows']}`",
            f"- heldout count: `{metrics['heldout_count']}`",
            "",
            "## C103 greenlight 기준",
            "",
            "- 64 candidates covered",
            f"- QA/manual confirmed local positives >= `{metrics['min_confirmed_positive']}`",
            "- teacher_only_positive_count = `0`",
            "- heldout_leakage_count = `0`",
            "- missing_path_count = `0`",
            "",
        )
    )


def build_report(summary: JsonObject, inventory: JsonObject) -> str:
    return "\n".join(
        (
            "# C102 stronger VLM QA teacher gate",
            "",
            f"- decision: `{summary['decision']}`",
            f"- selected_teacher_status: `{summary['selected_teacher_status']}`",
            f"- candidate_rows: `{summary['candidate_rows']}`",
            f"- covered_rows: `{summary['covered_rows']}`",
            f"- qa_positive_candidate_count: `{summary['qa_positive_candidate_count']}`",
            f"- confirmed_local_positive_count: `{summary['confirmed_local_positive_count']}`",
            f"- teacher_only_positive_count: `{summary['teacher_only_positive_count']}`",
            f"- local_negative_count: `{summary['local_negative_count']}`",
            f"- unclear_count: `{summary['unclear_count']}`",
            f"- heldout_leakage_count: `{summary['heldout_leakage_count']}`",
            f"- missing_path_count: `{summary['missing_path_count']}`",
            "",
            "## QA label counts",
            "",
            json.dumps(summary["qa_label_counts"], ensure_ascii=False, indent=2),
            "",
            "## final label counts",
            "",
            json.dumps(summary["final_label_counts"], ensure_ascii=False, indent=2),
            "",
            "## 판단",
            "",
            str(summary["blocker_reason"] or "C103 training can proceed with the reviewed C102 manifest."),
            "",
            "## inventory",
            "",
            json.dumps(inventory["key_metrics"], ensure_ascii=False, indent=2),
            "",
        )
    )
