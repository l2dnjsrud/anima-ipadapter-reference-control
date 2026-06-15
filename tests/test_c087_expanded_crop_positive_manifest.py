from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c087_expanded_crop_positive_manifest import (
    C087ManifestConfig,
    build_c087_expanded_crop_positive_manifest,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c087_builds_expanded_crop_and_anchored_manifests(tmp_path: Path) -> None:
    # Given: approved c083 crop pairs plus anchor rows for the c085-style mix.
    reviewed_path = tmp_path / "reviewed.jsonl"
    approved_path = tmp_path / "approved.jsonl"
    c060_path = tmp_path / "c060.jsonl"
    color_root = tmp_path / "color_root"
    heldout_summary = tmp_path / "heldout.json"
    candidates = tuple(
        _candidate(tmp_path, group_index, source_index, crop_index)
        for group_index in range(4)
        for source_index in range(3)
        for crop_index in range(2)
    )
    c060_rows = [_row(f"clean/ref{i:03d}", f"clean/ref{i:03d}") for i in range(32)]
    c060_rows.extend(_row(f"positive/ref{i:03d}", f"positive/tgt{i:03d}") for i in range(58))
    c060_rows.extend(_row(f"failure/ref{i:03d}", f"failure/tgt{i:03d}") for i in range(64))
    _write_jsonl(reviewed_path, candidates)
    _write_jsonl(approved_path, _directed_pairs(candidates))
    _write_jsonl(c060_path, c060_rows)
    heldout_summary.write_text(json.dumps({"heldout_ids": []}), encoding="utf-8")
    for row in c060_rows:
        _write_image(color_root / f"{row['ref_id']}.jpg")
        _write_image(color_root / f"{row['tgt_id']}.jpg")
        _write_caption(color_root / f"{row['tgt_id']}.txt")

    # When: c087 builds an expanded crop manifest and anchored training manifest.
    config = C087ManifestConfig(
        reviewed_labels_path=reviewed_path,
        approved_pairs_path=approved_path,
        expanded_root=tmp_path / "expanded_root",
        expanded_manifest_path=tmp_path / "out" / "c087_crop.jsonl",
        expanded_summary_path=tmp_path / "out" / "c087_crop.summary.json",
        expanded_report_path=tmp_path / "eval" / "crop_report.md",
        anchored_root=tmp_path / "anchored_root",
        anchored_manifest_path=tmp_path / "out" / "c087_anchor.jsonl",
        anchored_summary_path=tmp_path / "out" / "c087_anchor.summary.json",
        combined_summary_path=tmp_path / "eval" / "manifest_stdout.json",
        c060_manifest_path=c060_path,
        color_root=color_root,
        heldout_summary_path=heldout_summary,
        max_pairs_per_group=5,
        max_pairs_per_source_pair=2,
        crop_row_limit=20,
    )
    summary = build_c087_expanded_crop_positive_manifest(config)

    # Then: the chained manifests are larger than c084's tiny smoke setup and leak no heldout.
    expanded_rows = _read_jsonl(config.expanded_manifest_path)
    anchored_rows = _read_jsonl(config.anchored_manifest_path)
    assert summary.decision == "ready_for_c087_expanded_crop_positive_training"
    assert summary.expanded_crop_summary.selected_rows == 20
    assert summary.expanded_crop_summary.approved_groups == 4
    assert summary.expanded_crop_summary.heldout_rows_used == 0
    assert summary.anchored_summary.c084_crop_rows == 20
    assert summary.anchored_summary.total_rows == 100
    assert summary.anchored_summary.heldout_rows_used == 0
    assert len(expanded_rows) == 20
    assert len(anchored_rows) == 100
    assert config.combined_summary_path.is_file()


def _candidate(
    tmp_path: Path,
    group_index: int,
    source_index: int,
    crop_index: int,
) -> JsonObject:
    candidate_id = f"g{group_index}_s{source_index}_c{crop_index}"
    image_path = tmp_path / "crops" / f"{candidate_id}.png"
    _write_image(image_path)
    return {
        "candidate_id": candidate_id,
        "group_id": f"group{group_index}",
        "source_candidate_id": f"group{group_index}_source{source_index}",
        "local_image_path": str(image_path),
        "manual_label": "target_positive",
        "heldout_excluded": True,
    }


def _directed_pairs(candidates: tuple[JsonObject, ...]) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for ref in candidates:
        for tgt in candidates:
            if ref["group_id"] != tgt["group_id"]:
                continue
            rows.append(
                {
                    "group_id": ref["group_id"],
                    "ref_id": f"external/c083_sheet_crop_pairs/{ref['candidate_id']}",
                    "tgt_id": f"external/c083_sheet_crop_pairs/{tgt['candidate_id']}",
                    "prompt": "source prompt",
                }
            )
    return tuple(rows)


def _row(ref_id: str, tgt_id: str) -> JsonObject:
    return {"ref_id": ref_id, "tgt_id": tgt_id, "prompt": "mrcolor_panel_style, safe"}


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...] | list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    parsed: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            parsed.append(raw)
    return tuple(parsed)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "green").save(path)


def _write_caption(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("caption\n", encoding="utf-8")
