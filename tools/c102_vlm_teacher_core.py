from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

from tools.siglip_auto_caption_types import JsonObject

LABELS: Final = (
    "direct_green_non_human",
    "human_character",
    "green_background_or_object",
    "proxy_only",
    "unclear",
)


def build_prompt(row: JsonObject) -> str:
    return "\n".join(
        (
            "Choose exactly one label and output only two lines:",
            "LABEL: direct_green_non_human | human_character | green_background_or_object | proxy_only | unclear",
            "EVIDENCE: short visual phrase",
            "",
            "Definitions:",
            "direct_green_non_human = main narrative character is person-like but clearly non-human/monster/creature with green skin or a green face.",
            "human_character = ordinary human main figure.",
            "green_background_or_object = green is on leaves/cups/food/scenery/props/UI or non-character animals/insects.",
            "proxy_only = fangs/red eyes/side profile/beard/headwear but no direct green non-human character.",
            "unclear = not enough evidence.",
            "",
            f"Prior local label: {row.get('manual_label', '')}",
            f"Prior evidence: {row.get('label_evidence', '')}",
        )
    )


def parse_teacher_label(response: str) -> str:
    lowered = response.lower()
    for line in lowered.splitlines()[:4]:
        stripped = line.replace("label:", "").strip()
        if stripped in LABELS:
            return stripped
    for label in LABELS:
        if label in lowered:
            return label
    if "not enough" in lowered or "unclear" in lowered:
        return "unclear"
    return "unclear"


def build_review_rows(candidates: tuple[JsonObject, ...], responses: dict[str, str]) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for candidate in candidates:
        image_id = _str(candidate, "image_id")
        raw_response = responses.get(image_id, "")
        qa_label = parse_teacher_label(raw_response)
        final_label, review_source = _final_label(_str(candidate, "manual_label"), qa_label)
        row = dict(candidate)
        row.update(
            {
                "qa_label": qa_label,
                "qa_raw_response": raw_response,
                "qa_evidence": _evidence(raw_response),
                "final_label": final_label,
                "final_review_source": review_source,
            }
        )
        rows.append(row)
    return tuple(rows)


def build_summary(
    rows: tuple[JsonObject, ...],
    *,
    input_rows: int,
    heldout_leakage: int,
    min_confirmed_positive: int,
    selected_teacher_status: str,
) -> JsonObject:
    missing = sum(1 for row in rows if not _path_ok(row))
    covered = sum(1 for row in rows if bool(_str(row, "qa_raw_response")))
    qa_positive = _field_count(rows, "qa_label", "direct_green_non_human")
    confirmed_positive = _field_count(rows, "final_label", "local_positive")
    teacher_only_positive = sum(1 for row in rows if _is_teacher_only_positive(row))
    can_train = (
        input_rows == covered
        and input_rows == len(rows)
        and confirmed_positive >= min_confirmed_positive
        and teacher_only_positive == 0
        and missing == 0
        and heldout_leakage == 0
    )
    decision = "c103_training_greenlit" if can_train else "c103_blocked_needs_manual_annotation_or_external_teacher"
    return {
        "candidate_rows": input_rows,
        "covered_rows": covered,
        "reviewed_rows": len(rows),
        "heldout_leakage_count": heldout_leakage,
        "missing_path_count": missing,
        "teacher_only_positive_count": teacher_only_positive,
        "qa_positive_candidate_count": qa_positive,
        "confirmed_local_positive_count": confirmed_positive,
        "local_negative_count": _field_count(rows, "final_label", "local_negative"),
        "unclear_count": _field_count(rows, "final_label", "unclear"),
        "min_confirmed_positive": min_confirmed_positive,
        "qa_label_counts": _counts(rows, "qa_label"),
        "final_label_counts": _counts(rows, "final_label"),
        "source_bucket_counts": _counts(rows, "source_bucket"),
        "selected_teacher_status": selected_teacher_status,
        "decision": decision,
        "blocker_reason": "" if can_train else _blocker_reason(selected_teacher_status, confirmed_positive),
    }


def _final_label(prior_label: str, qa_label: str) -> tuple[str, str]:
    if prior_label == "local_negative":
        return ("local_negative", "c101_negative_preserved")
    match qa_label:
        case "direct_green_non_human":
            return ("local_positive", "qwen3vl_qa_confirmed")
        case "human_character" | "green_background_or_object":
            return ("local_negative", "qwen3vl_qa_negative")
        case "proxy_only" | "unclear":
            return ("unclear", "qwen3vl_qa_unclear")
        case _:
            return ("unclear", "qwen3vl_qa_unclear")


def _is_teacher_only_positive(row: JsonObject) -> bool:
    return row.get("qa_label") == "direct_green_non_human" and row.get("final_label") != "local_positive"


def _evidence(response: str) -> str:
    for line in response.splitlines():
        lowered = line.lower().strip()
        if lowered.startswith("evidence:"):
            return line.split(":", 1)[1].strip()
    return response.strip().splitlines()[0][:160] if response.strip() else "missing response"


def _blocker_reason(selected_teacher_status: str, confirmed_positive: int) -> str:
    if selected_teacher_status.startswith("blocked"):
        return "No runnable local generative VLM QA teacher was available."
    return f"C102 QA package did not reach 8 conflict-free confirmed local positives; confirmed={confirmed_positive}."


def _field_count(rows: tuple[JsonObject, ...], field: str, value: str) -> int:
    return sum(1 for row in rows if row.get(field) == value)


def _counts(rows: tuple[JsonObject, ...], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field, "unknown")) for row in rows))


def _str(row: JsonObject, key: str) -> str:
    value = row.get(key)
    return value if isinstance(value, str) else ""


def _path_ok(row: JsonObject) -> bool:
    value = row.get("paths_ok")
    if isinstance(value, bool):
        return value
    return Path(_str(row, "image_path")).is_file()
