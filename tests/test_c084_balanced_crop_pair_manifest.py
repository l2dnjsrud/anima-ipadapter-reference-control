from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.c084_balanced_crop_pair_manifest import (
    C084ManifestConfig,
    C084ManifestError,
    build_c084_balanced_crop_pair_manifest,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c084_balances_crop_pairs_and_materializes_selected_images(tmp_path: Path) -> None:
    # Given: four visually-approved groups with many directed crop pairs per group.
    reviewed_path, pair_path = _fixture_paths(tmp_path)
    candidates = tuple(
        _candidate(tmp_path, group_index, source_index, crop_index)
        for group_index in range(4)
        for source_index in range(3)
        for crop_index in range(2)
    )
    _write_jsonl(reviewed_path, candidates)
    _write_jsonl(pair_path, _directed_pairs(candidates))

    # When: c084 downsamples with group and source-pair caps.
    summary = build_c084_balanced_crop_pair_manifest(
        _config(tmp_path, max_pairs_per_group=5, max_pairs_per_source_pair=2)
    )

    # Then: selected rows are balanced, cross-source, and ready for the trainer.
    rows = _read_jsonl(tmp_path / "out" / "c084.jsonl")
    assert summary.source_pairs == 144
    assert summary.selected_rows == 20
    assert summary.approved_groups == 4
    assert summary.heldout_rows_used == 0
    assert summary.direct_self_pair_rows == 0
    assert summary.same_source_pair_rows == 0
    assert set(summary.selected_group_counts.values()) == {5}
    assert all(count <= 2 for count in summary.selected_source_pair_counts.values())
    assert all(row["ref_id"] != row["tgt_id"] for row in rows)
    assert all(str(row["ref_id"]).startswith("external/c084_sheet_crop_pairs/") for row in rows)
    assert (tmp_path / "scratch" / "external" / "c084_sheet_crop_pairs" / "g0_s0_c0.jpg").is_file()
    assert (tmp_path / "scratch" / "external" / "c084_sheet_crop_pairs" / "g0_s0_c0.txt").is_file()
    assert (tmp_path / "eval" / "manifest_report.md").is_file()


def test_c084_rejects_too_few_selected_groups(tmp_path: Path) -> None:
    # Given: only three groups, below the minimum c084 training gate.
    reviewed_path, pair_path = _fixture_paths(tmp_path)
    candidates = tuple(
        _candidate(tmp_path, group_index, source_index, crop_index)
        for group_index in range(3)
        for source_index in range(2)
        for crop_index in range(1)
    )
    _write_jsonl(reviewed_path, candidates)
    _write_jsonl(pair_path, _directed_pairs(candidates))

    # When/Then: the builder blocks training rather than silently using too little data.
    with pytest.raises(C084ManifestError, match="minimum group"):
        build_c084_balanced_crop_pair_manifest(_config(tmp_path))


def test_c084_rejects_missing_crop_images(tmp_path: Path) -> None:
    # Given: reviewed metadata that points to a missing raw crop.
    reviewed_path, pair_path = _fixture_paths(tmp_path)
    rows = (
        _candidate(tmp_path, 0, 0, 0, write_image=False),
        _candidate(tmp_path, 0, 1, 0),
    )
    _write_jsonl(reviewed_path, rows)
    _write_jsonl(pair_path, _directed_pairs(rows))

    # When/Then: the missing crop blocks manifest generation.
    with pytest.raises(C084ManifestError, match="missing crop image"):
        build_c084_balanced_crop_pair_manifest(_config(tmp_path, minimum_groups=1))


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "reviewed.jsonl", tmp_path / "pairs.jsonl"


def _config(
    tmp_path: Path,
    *,
    max_pairs_per_group: int = 24,
    max_pairs_per_source_pair: int = 2,
    minimum_groups: int = 4,
) -> C084ManifestConfig:
    return C084ManifestConfig(
        reviewed_labels_path=tmp_path / "reviewed.jsonl",
        approved_pairs_path=tmp_path / "pairs.jsonl",
        scratch_image_root=tmp_path / "scratch",
        output_manifest_path=tmp_path / "out" / "c084.jsonl",
        output_summary_path=tmp_path / "out" / "c084.summary.json",
        output_report_path=tmp_path / "eval" / "manifest_report.md",
        max_pairs_per_group=max_pairs_per_group,
        max_pairs_per_source_pair=max_pairs_per_source_pair,
        minimum_groups=minimum_groups,
    )


def _candidate(
    tmp_path: Path,
    group_index: int,
    source_index: int,
    crop_index: int,
    *,
    write_image: bool = True,
) -> JsonObject:
    candidate_id = f"g{group_index}_s{source_index}_c{crop_index}"
    image_path = tmp_path / "crops" / f"{candidate_id}.png"
    if write_image:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"png")
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


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
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
