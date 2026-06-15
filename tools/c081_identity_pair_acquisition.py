from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tools.c071_seed_package import LABEL_SCHEMA
from tools.c076_source_expansion_io import read_label_map, write_jsonl, write_sheet
from tools.siglip_auto_caption_types import JsonObject, JsonValue

OUT_DIR: Final = Path("eval/c081_identity_preserving_pair_acquisition_20260613")
SCRATCH: Final = Path(".tmp/c081_identity_preserving_pair_acquisition")
NEGATIVE: Final = (
    "low quality, blurry, text, watermark, logo, multiple characters, duplicate "
    "characters, normal human skin, inconsistent character design, nude, nsfw"
)
MIN_APPROVED_GROUPS: Final = 4
MIN_APPROVED_PAIRS: Final = 24

IDENTITIES: Final = (
    ("c081_green_oni_scout", "young green oni scout girl, short ivory horns, yellow eyes, dark bob hair, red scarf, leather shoulder guard"),
    ("c081_jade_lizard_monk", "jade lizardfolk monk, long tail, scale cheek marks, amber eyes, blue sash, sleeveless robe"),
    ("c081_goblin_mage", "small emerald goblin mage, big pointed ears, round goggles, purple hood, tiny tusks"),
    ("c081_frog_yokai_guard", "frog yokai guard, round green face, webbed hands, straw hat, teal robe"),
    ("c081_plant_dryad", "plant dryad girl, green skin, leaf hair, vine crown, gold eyes, bark-pattern dress"),
    ("c081_serpent_dancer", "serpent folk dancer, green scales, long tail, black hair, bronze ornaments"),
)
VIEWS: Final = (
    ("front", "front-facing upper-body portrait, clear face and costume"),
    ("three_quarter", "three-quarter full-body pose, same face, same costume and colors"),
    ("profile", "side-profile bust, same horns ears hair and facial markings"),
    ("action", "single-character action pose, same identity, same palette and outfit"),
)


@dataclass(frozen=True, slots=True)
class C081Config:
    out_dir: Path = OUT_DIR
    scratch_dir: Path = SCRATCH
    labels_path: Path | None = None


def build_c081_prompt_package(config: C081Config) -> JsonObject:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _prompt_rows()
    write_jsonl(config.out_dir / "prompt_manifest.jsonl", rows)
    summary = {
        "source": "c081_identity_preserving_pair_acquisition",
        "identity_group_count": len(IDENTITIES),
        "view_count_per_group": len(VIEWS),
        "prompt_count": len(rows),
        "heldout_rows_used": 0,
        "training_started": False,
        "minimum_approved_groups": MIN_APPROVED_GROUPS,
        "minimum_approved_pairs": MIN_APPROVED_PAIRS,
        "decision": "prompt_package_ready",
    }
    _write_summary(config.out_dir / "summary.json", summary)
    (config.out_dir / "report.md").write_text(_prompt_report(summary), encoding="utf-8")
    return summary


def review_c081_generation(config: C081Config, *, generation_manifest_path: Path | None = None) -> JsonObject:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = generation_manifest_path or config.out_dir / "generation_manifest.jsonl"
    generated = _read_jsonl(manifest_path) if manifest_path.is_file() else ()
    labels_path = config.labels_path or config.out_dir / "manual_visual_labels.csv"
    labels = read_label_map(labels_path) if labels_path.is_file() else {}
    reviewed = tuple(_reviewed_row(row, labels.get(str(row["candidate_id"]), {})) for row in generated if row.get("status") == "generated")
    write_jsonl(config.out_dir / "reviewed_pair_labels.jsonl", reviewed)
    _write_visual_template(config.out_dir / "visual_label_template.csv", generated)
    pairs = _approved_pairs(reviewed)
    write_jsonl(config.out_dir / "approved_pair_manifest.jsonl", pairs)
    sheet_path = config.scratch_dir / "contact_sheet.jpg"
    sheet_written = write_sheet(reviewed, sheet_path)
    summary = _review_summary(generated, reviewed, pairs, sheet_path=sheet_path, sheet_written=sheet_written)
    _write_summary(config.out_dir / "summary.json", summary)
    (config.out_dir / "report.md").write_text(_review_report(summary), encoding="utf-8")
    return summary


def _prompt_rows() -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for group_index, (group_id, identity) in enumerate(IDENTITIES):
        for view_index, (view_id, view_prompt) in enumerate(VIEWS):
            rows.append(
                {
                    "candidate_id": f"{group_id}_{view_id}",
                    "group_id": group_id,
                    "view_id": view_id,
                    "seed": 20260810 + group_index * 100 + view_index,
                    "prompt": _prompt(identity, view_prompt),
                    "negative": NEGATIVE,
                    "source_bucket": "synthetic_identity_preserving_direct_green_pair",
                    "source_labels": ["direct_green", "non_human", "identity_group", "same_character_views"],
                    "heldout_excluded": True,
                }
            )
    return tuple(rows)


def _prompt(identity: str, view_prompt: str) -> str:
    return (
        "masterpiece, best quality, clean anime manhwa character design sheet crop, "
        f"single non-human character, {identity}, clearly visible green skin, {view_prompt}, "
        "consistent identity, same costume, same palette, no other characters, simple pale background"
    )


def _reviewed_row(row: JsonObject, label_row: JsonObject) -> JsonObject:
    label = str(label_row.get("manual_label") or "useful_proxy_non_human")
    if label not in LABEL_SCHEMA:
        raise ValueError(f"unknown c081 manual label: {label}")
    return dict(row) | {
        "manual_label": label,
        "manual_note": str(label_row.get("manual_note") or "requires identity-pair visual review"),
        "visual_confirmation": label == "target_positive",
    }


def _approved_pairs(reviewed: tuple[JsonObject, ...]) -> tuple[JsonObject, ...]:
    by_group: dict[str, list[JsonObject]] = {}
    for row in reviewed:
        if row["manual_label"] == "target_positive":
            by_group.setdefault(str(row["group_id"]), []).append(row)
    pairs: list[JsonObject] = []
    for group_id, rows in sorted(by_group.items()):
        if len(rows) < 2:
            continue
        for ref in rows:
            for tgt in rows:
                if ref["candidate_id"] == tgt["candidate_id"]:
                    continue
                pairs.append(
                    {
                        "group_id": group_id,
                        "ref_id": f"external/c081_identity_pairs/{ref['candidate_id']}",
                        "tgt_id": f"external/c081_identity_pairs/{tgt['candidate_id']}",
                        "prompt": "mrcolor_panel_style, same direct-green non-human character identity, clean color manhwa panel",
                    }
                )
    return tuple(pairs)


def _review_summary(
    generated: tuple[JsonObject, ...],
    reviewed: tuple[JsonObject, ...],
    pairs: tuple[JsonObject, ...],
    *,
    sheet_path: Path,
    sheet_written: bool,
) -> JsonObject:
    approved_groups = {str(row["group_id"]) for row in reviewed if row["manual_label"] == "target_positive"}
    decision = "ready_for_c082_paired_training_manifest" if len(approved_groups) >= MIN_APPROVED_GROUPS and len(pairs) >= MIN_APPROVED_PAIRS else "more_identity_pairs_required"
    return {
        "source": "c081_identity_preserving_pair_acquisition",
        "prompt_count": len(_prompt_rows()),
        "generated_count": sum(1 for row in generated if row.get("status") == "generated"),
        "blank_count": sum(1 for row in generated if row.get("blank") is True),
        "reviewed_rows": len(reviewed),
        "approved_group_count": len(approved_groups),
        "approved_pair_rows": len(pairs),
        "direct_self_pair_rows": sum(1 for row in pairs if row["ref_id"] == row["tgt_id"]),
        "minimum_approved_groups": MIN_APPROVED_GROUPS,
        "minimum_approved_pairs": MIN_APPROVED_PAIRS,
        "heldout_rows_used": 0,
        "training_started": False,
        "raw_generated_images_committed": False,
        "contact_sheet_path": str(sheet_path),
        "contact_sheet_written": sheet_written,
        "decision": decision,
    }


def _prompt_report(summary: JsonObject) -> str:
    return "\n".join(["# c081 identity-preserving pair acquisition", "", f"- decision: `{summary['decision']}`", f"- prompt_count: `{summary['prompt_count']}`", f"- identity_group_count: `{summary['identity_group_count']}`", "- training_started: false", ""])


def _review_report(summary: JsonObject) -> str:
    return "\n".join(["# c081 identity-preserving pair acquisition", "", f"- decision: `{summary['decision']}`", f"- generated_count: `{summary['generated_count']}`", f"- approved_group_count: `{summary['approved_group_count']}`", f"- approved_pair_rows: `{summary['approved_pair_rows']}`", f"- direct_self_pair_rows: `{summary['direct_self_pair_rows']}`", f"- contact_sheet_path: `{summary['contact_sheet_path']}`", ""])


def _write_visual_template(path: Path, rows: tuple[JsonObject, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("candidate_id", "group_id", "view_id", "status", "manual_label", "manual_note", "allowed_labels"), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"candidate_id": row["candidate_id"], "group_id": row.get("group_id", ""), "view_id": row.get("view_id", ""), "status": row.get("status", ""), "manual_label": row.get("manual_label", ""), "manual_note": "", "allowed_labels": "|".join(LABEL_SCHEMA)})


def _write_summary(path: Path, summary: JsonObject) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)


if __name__ == "__main__":
    build_c081_prompt_package(C081Config())
