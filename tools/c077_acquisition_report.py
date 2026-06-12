from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from tools.siglip_auto_caption_types import JsonObject

MINIMUM_TARGET_POSITIVES: Final = 24
MINIMUM_NEW_TARGET_POSITIVES: Final = 12


def summarize_acquisition(
    source_rows: tuple[JsonObject, ...],
    downloads: tuple[JsonObject, ...],
    reviewed: tuple[JsonObject, ...],
    *,
    sheet_path: Path,
    sheet_written: bool,
) -> JsonObject:
    target_ids = {str(row["image_id"]) for row in reviewed if row["manual_label"] == "target_positive"}
    new_targets = [row for row in reviewed if row["manual_label"] == "target_positive" and row.get("review_source") != "prior_c074_manual"]
    new_candidates = [row for row in downloads if row.get("review_source") != "prior_c074_manual"]
    candidate_count = sum(int(row["potential_candidate_count"]) for row in source_rows)
    decision = _decision(len(new_candidates), len(target_ids), len(new_targets))
    return {
        "source": "c077_direct_green_target_positive_acquisition",
        "inspected_source_count": len(source_rows),
        "candidate_count": candidate_count,
        "download_candidate_count": len(downloads),
        "new_candidate_count": len(new_candidates),
        "downloaded_count": sum(1 for row in downloads if row["download_status"] in {"downloaded", "copied_prior_c074"}),
        "new_downloaded_count": sum(1 for row in new_candidates if row["download_status"] == "downloaded"),
        "reviewed_rows": len(reviewed),
        "target_positive_confirmed_count": len(target_ids),
        "new_target_positive_confirmed_count": len({str(row["image_id"]) for row in new_targets}),
        "unique_target_positive_count": len(target_ids),
        "minimum_target_positive_required": MINIMUM_TARGET_POSITIVES,
        "minimum_new_target_positive_required": MINIMUM_NEW_TARGET_POSITIVES,
        "heldout_rows_used": sum(1 for row in downloads if bool(row.get("heldout_excluded", False))),
        "large_downloads_performed": False,
        "raw_external_images_committed": False,
        "committed_external_image_count": 0,
        "contact_sheet_path": str(sheet_path),
        "contact_sheet_written": sheet_written,
        "source_probe_decision": "source_blocked" if len(new_candidates) == 0 else "source_probe_ready_for_review",
        "candidate_review_decision": decision,
        "decision": decision,
        "next_training_or_data_action": next_action(decision),
    }


def write_decision_report(out_dir: Path, summary: JsonObject) -> None:
    if summary["decision"] == "source_blocked_manual_needed":
        out_dir.joinpath("source_blocked_report.md").write_text(blocked_report(summary), encoding="utf-8")
    if summary["decision"] == "manual_needed_more_target_positives":
        out_dir.joinpath("manual_needed_report.md").write_text(blocked_report(summary), encoding="utf-8")


def acquisition_report(summary: JsonObject) -> str:
    return "\n".join(
        [
            "# c077 direct-green target-positive acquisition",
            "",
            f"- decision: `{summary['decision']}`",
            f"- inspected_source_count: {summary['inspected_source_count']}",
            f"- candidate_count: {summary['candidate_count']}",
            f"- download_candidate_count: {summary['download_candidate_count']}",
            f"- new_candidate_count: {summary['new_candidate_count']}",
            f"- downloaded_count: {summary['downloaded_count']}",
            f"- target_positive_confirmed_count: {summary['target_positive_confirmed_count']}",
            f"- new_target_positive_confirmed_count: {summary['new_target_positive_confirmed_count']}",
            f"- contact_sheet_path: `{summary['contact_sheet_path']}`",
            f"- next_training_or_data_action: {summary['next_training_or_data_action']}",
            "",
        ]
    )


def next_action(decision: str) -> str:
    match decision:
        case "ready_for_c077_training_manifest":
            return "build the next training manifest from c074 seeds plus c077 reviewed target positives"
        case "manual_needed_more_target_positives":
            return "use the contact sheet/manual labels to find at least 12 new direct-green target positives before training"
        case "source_blocked_manual_needed":
            return "provide or locate a new direct-green non-human source; current bounded source set is insufficient"
        case unreachable:
            raise ValueError(f"unexpected c077 decision: {unreachable}")


def blocked_report(summary: JsonObject) -> str:
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    return f"# c077 acquisition threshold not met\n\n```json\n{rendered}\n```\n"


def _decision(new_candidates: int, total_targets: int, new_targets: int) -> str:
    if new_candidates == 0:
        return "source_blocked_manual_needed"
    if total_targets >= MINIMUM_TARGET_POSITIVES and new_targets >= MINIMUM_NEW_TARGET_POSITIVES:
        return "ready_for_c077_training_manifest"
    return "manual_needed_more_target_positives"
