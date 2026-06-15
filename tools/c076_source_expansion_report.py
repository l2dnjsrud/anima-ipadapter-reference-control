from __future__ import annotations

import json

from tools.siglip_auto_caption_types import JsonObject


def feature_boundary_metrics(reviewed: tuple[JsonObject, ...]) -> JsonObject:
    only_prior_targets = all(
        row.get("review_source") == "prior_c074_manual"
        for row in reviewed
        if row.get("manual_label") == "target_positive"
    )
    return {
        "status": "skipped_no_new_reviewed_target_positives" if only_prior_targets else "ready_for_pair_probe",
        "c075_direct_green_pe_blend_uplift": 0.03799172639846802,
        "c075_direct_green_pe_candidate_uplift": -0.02068805992603302,
        "c075_direct_green_qwenvl_blend_uplift": -0.012108683586120605,
        "c075_direct_green_qwenvl_candidate_uplift": -0.014385020732879639,
        "interpretation": "prior c074 positives alone already failed c075; train only after new visually confirmed positives expand the target set",
    }


def next_action(decision: str) -> str:
    match decision:
        case "ready_for_c077_training":
            return "build c077 manifest from c074 seeds plus new reviewed positives and run a bounded QwenVL/SigLIP feature gate"
        case "more_data_required":
            return "continue data acquisition/manual labeling before another checkpoint training run"
        case "source_blocked":
            return "find a new source or provide manually approved direct-green non-human positives"
        case unreachable:
            raise ValueError(f"unexpected c076 decision: {unreachable}")


def report(summary: JsonObject) -> str:
    return "\n".join(
        [
            "# c076 paired direct-green source expansion",
            "",
            f"- decision: `{summary['decision']}`",
            f"- source_probe_decision: `{summary['source_probe_decision']}`",
            f"- inspected_source_count: {summary['inspected_source_count']}",
            f"- candidate_count: {summary['candidate_count']}",
            f"- downloaded_count: {summary['downloaded_count']}",
            f"- reviewed_rows: {summary['reviewed_rows']}",
            f"- target_positive_confirmed_count: {summary['target_positive_confirmed_count']}",
            f"- new_target_positive_confirmed_count: {summary['new_target_positive_confirmed_count']}",
            f"- feature_boundary_status: `{summary['feature_boundary_status']}`",
            f"- contact_sheet_path: `{summary['contact_sheet_path']}`",
            f"- next_training_or_data_action: {summary['next_training_or_data_action']}",
            "",
        ]
    )


def blocked_report(summary: JsonObject) -> str:
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    return f"# c076 source blocked\n\nNo safe bounded source candidate was found.\n\n```json\n{rendered}\n```\n"
