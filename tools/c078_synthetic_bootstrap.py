from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tools.c071_seed_package import LABEL_SCHEMA
from tools.c076_source_expansion_io import read_label_map, write_jsonl, write_sheet
from tools.siglip_auto_caption_types import JsonObject, JsonValue

OUT_DIR: Final = Path("eval/c078_synthetic_direct_green_bootstrap_20260612")
SCRATCH: Final = Path(".tmp/c078_synthetic_direct_green_bootstrap")
MINIMUM_TARGET_POSITIVES: Final = 24
MINIMUM_NEW_TARGET_POSITIVES: Final = 12
NEGATIVE: Final = "low quality, blurry, text, watermark, logo, multiple characters, normal human skin, nude, nsfw"

ARCHETYPES: Final = (
    "emerald goblin swordswoman with small tusks and pointed ears",
    "jade lizardfolk monk with scales and a long tail",
    "green oni warrior with short horns and fangs",
    "mossy forest spirit girl with leaf-like hair and non-human eyes",
    "bright green slime knight with translucent skin and armor",
    "alien princess with emerald skin and antennae",
    "frog yokai martial artist with webbed hands",
    "dragonkin archer with green scales and horns",
    "orc herbalist with olive green skin and braided hair",
    "serpent folk dancer with green scales and tail",
    "goblin mage with glowing yellow eyes and green skin",
    "cactus dryad with green skin and thorn motifs",
    "swamp demon girl with moss green skin and curved horns",
    "turtle shell guardian with emerald skin and shell armor",
    "gecko rogue with lime green skin and spotted scales",
    "plant monster noble with vine hair and green face",
    "green kobold scout with fangs and hooded cloak",
    "leaf fairy knight with non-human ears and green skin",
    "jade gargoyle girl with small wings and stone horns",
    "toad alchemist with round green face and robe",
    "green insectoid princess with antennae and chitin accents",
    "forest troll child with big ears and emerald skin",
    "willow witch with green skin and branch-like horns",
    "emerald salamander warrior with tail and scale patterns",
)


@dataclass(frozen=True, slots=True)
class C078BootstrapConfig:
    out_dir: Path = OUT_DIR
    scratch_dir: Path = SCRATCH
    labels_path: Path | None = None


def build_c078_prompt_package(config: C078BootstrapConfig) -> JsonObject:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _prompt_rows()
    write_jsonl(config.out_dir / "prompt_manifest.jsonl", rows)
    summary = {
        "source": "c078_synthetic_direct_green_bootstrap",
        "prompt_count": len(rows),
        "heldout_rows_used": 0,
        "training_started": False,
        "target_positive_threshold_total": MINIMUM_TARGET_POSITIVES,
        "target_positive_threshold_new": MINIMUM_NEW_TARGET_POSITIVES,
        "command_surface": "ComfyUI API text-only Anima/Qwen image generation",
        "decision": "prompt_package_ready",
    }
    (config.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (config.out_dir / "report.md").write_text(_prompt_report(summary), encoding="utf-8")
    return summary


def review_c078_generation(config: C078BootstrapConfig, *, generation_manifest_path: Path | None = None) -> JsonObject:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    path = config.out_dir / "generation_manifest.jsonl" if generation_manifest_path is None else generation_manifest_path
    generated = _read_jsonl(path) if path.is_file() else ()
    labels_path = config.out_dir / "manual_visual_labels.csv" if config.labels_path is None else config.labels_path
    labels = read_label_map(labels_path) if labels_path.is_file() else {}
    reviewed = _review(generated, labels)
    write_jsonl(config.out_dir / "reviewed_synthetic_labels.jsonl", reviewed)
    _write_visual_template(config.out_dir / "visual_label_template.csv", generated)
    sheet_path = config.scratch_dir / "contact_sheet.jpg"
    sheet_written = write_sheet(reviewed, sheet_path)
    summary = _review_summary(generated, reviewed, sheet_path=sheet_path, sheet_written=sheet_written)
    (config.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (config.out_dir / "report.md").write_text(_review_report(summary), encoding="utf-8")
    if summary["decision"] != "ready_for_c079_training_manifest":
        (config.out_dir / "manual_needed_report.md").write_text(_blocked_report(summary), encoding="utf-8")
    return summary


def _prompt_rows() -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for index, archetype in enumerate(ARCHETYPES):
        rows.append(
            {
                "candidate_id": f"c078_synth_{index:02d}",
                "seed": 20260780 + index,
                "prompt": _prompt(archetype),
                "negative": NEGATIVE,
                "source_bucket": "synthetic_direct_green_non_human",
                "source_labels": ["direct_green", "non_human", "single_character", "synthetic_bootstrap"],
                "heldout_excluded": True,
            }
        )
    return tuple(rows)


def _prompt(archetype: str) -> str:
    return (
        "masterpiece, best quality, clean anime manhwa illustration, single character, "
        f"non-human {archetype}, clearly visible green skin, face and body, modest fantasy outfit, "
        "front facing or three-quarter portrait, simple pale background, expressive eyes, "
        "sharp lineart, rich but controlled color palette"
    )


def _review(rows: tuple[JsonObject, ...], labels: dict[str, JsonObject]) -> tuple[JsonObject, ...]:
    reviewed: list[JsonObject] = []
    for row in rows:
        if row.get("status") != "generated":
            continue
        label_row = labels.get(str(row["candidate_id"]), {})
        manual_label = str(label_row.get("manual_label") or "useful_proxy_non_human")
        if manual_label not in LABEL_SCHEMA:
            raise ValueError(f"unknown c078 manual label: {manual_label}")
        reviewed.append(dict(row) | {"manual_label": manual_label, "manual_note": str(label_row.get("manual_note") or "synthetic candidate requires visual confirmation"), "visual_confirmation": manual_label == "target_positive"})
    return tuple(reviewed)


def _review_summary(rows: tuple[JsonObject, ...], reviewed: tuple[JsonObject, ...], *, sheet_path: Path, sheet_written: bool) -> JsonObject:
    new_targets = [row for row in reviewed if row["manual_label"] == "target_positive"]
    blank_count = sum(1 for row in rows if row.get("blank") is True)
    decision = "ready_for_c079_training_manifest" if len(new_targets) >= MINIMUM_NEW_TARGET_POSITIVES and blank_count == 0 else "manual_needed_review_synthetic_refs"
    return {
        "source": "c078_synthetic_direct_green_bootstrap",
        "prompt_count": len(_prompt_rows()),
        "generated_count": sum(1 for row in rows if row.get("status") == "generated"),
        "blank_count": blank_count,
        "reviewed_rows": len(reviewed),
        "new_target_positive_confirmed_count": len(new_targets),
        "target_positive_threshold_total": MINIMUM_TARGET_POSITIVES,
        "target_positive_threshold_new": MINIMUM_NEW_TARGET_POSITIVES,
        "command_surface": "ComfyUI02 API text-only Anima/Qwen image generation",
        "heldout_rows_used": 0,
        "training_started": False,
        "raw_generated_images_committed": False,
        "contact_sheet_path": str(sheet_path),
        "contact_sheet_written": sheet_written,
        "decision": decision,
        "next_training_or_data_action": _next_action(decision),
    }


def _next_action(decision: str) -> str:
    match decision:
        case "ready_for_c079_training_manifest":
            return "build c079 training manifest from visually confirmed synthetic target positives plus guard data"
        case "manual_needed_review_synthetic_refs":
            return "review or regenerate synthetic refs until at least 12 target positives are confirmed"
        case unreachable:
            raise ValueError(f"unexpected c078 decision: {unreachable}")


def _prompt_report(summary: JsonObject) -> str:
    return "\n".join(["# c078 synthetic direct-green bootstrap", "", f"- decision: `{summary['decision']}`", f"- prompt_count: {summary['prompt_count']}", "- training_started: false", ""])


def _review_report(summary: JsonObject) -> str:
    return "\n".join(["# c078 synthetic direct-green bootstrap", "", f"- decision: `{summary['decision']}`", f"- generated_count: {summary['generated_count']}", f"- new_target_positive_confirmed_count: {summary['new_target_positive_confirmed_count']}", f"- contact_sheet_path: `{summary['contact_sheet_path']}`", f"- next_training_or_data_action: {summary['next_training_or_data_action']}", ""])


def _blocked_report(summary: JsonObject) -> str:
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    return f"# c078 synthetic refs not ready\n\n```json\n{rendered}\n```\n"


def _write_visual_template(path: Path, rows: tuple[JsonObject, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("candidate_id", "status", "manual_label", "manual_note", "allowed_labels"), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "status": row.get("status", ""),
                    "manual_label": row.get("manual_label", ""),
                    "manual_note": "",
                    "allowed_labels": "|".join(LABEL_SCHEMA),
                }
            )


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)


if __name__ == "__main__":
    build_c078_prompt_package(C078BootstrapConfig())
