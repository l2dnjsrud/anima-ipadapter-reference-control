from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c089_shape_distillation_manifest import build_c089_manifest


def test_build_c089_manifest_balances_shape_groups(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_manifest = tmp_path / "source.jsonl"
    heldout_summary = tmp_path / "heldout.json"
    output_root = tmp_path / "out_root"
    output_manifest = tmp_path / "out.jsonl"
    output_summary = tmp_path / "out.summary.json"
    rows = [
        _row(f"external/c084_sheet_crop_pairs/c083_{group}_ref{i:02d}", f"external/c084_sheet_crop_pairs/c083_{group}_tgt{i:02d}")
        for group in ("c082_frog_yokai_guard", "c082_goblin_mage", "c082_green_oni_scout")
        for i in range(5)
    ]
    _write_jsonl(source_manifest, rows)
    heldout_summary.write_text(json.dumps({"heldout_ids": []}), encoding="utf-8")
    for row in rows:
        _write_image(source_root / f"{row['ref_id']}.jpg")
        _write_image(source_root / f"{row['tgt_id']}.jpg")
        _write_caption(source_root / f"{row['tgt_id']}.txt")

    summary = build_c089_manifest(
        source_manifest=source_manifest,
        source_root=source_root,
        output_root=output_root,
        output_manifest=output_manifest,
        output_summary=output_summary,
        heldout_summary=heldout_summary,
        max_rows_per_group=3,
    )

    manifest_rows = _read_jsonl(output_manifest)
    written = json.loads(output_summary.read_text(encoding="utf-8"))
    assert summary.total_rows == 9
    assert summary.heldout_rows_used == 0
    assert summary.selected_group_counts == {
        "c082_frog_yokai_guard": 3,
        "c082_goblin_mage": 3,
        "c082_green_oni_scout": 3,
    }
    assert written["teacher_source_labels"] == [
        "pe_teacher_prediction",
        "pe_token_retrieval",
        "edge_projection_silhouette_probe",
    ]
    assert len(manifest_rows) == 9
    assert (output_root / f"{manifest_rows[0]['ref_id']}.jpg").is_symlink()
    assert (output_root / f"{manifest_rows[0]['tgt_id']}.txt").is_symlink()


def test_build_c089_manifest_blocks_heldout_leakage(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_manifest = tmp_path / "source.jsonl"
    heldout_summary = tmp_path / "heldout.json"
    rows = [_row("external/c084_sheet_crop_pairs/c083_c082_frog_yokai_guard_ref00", "heldout/tgt")]
    _write_jsonl(source_manifest, rows)
    heldout_summary.write_text(json.dumps({"heldout_ids": ["heldout/tgt"]}), encoding="utf-8")
    for row in rows:
        _write_image(source_root / f"{row['ref_id']}.jpg")
        _write_image(source_root / f"{row['tgt_id']}.jpg")
        _write_caption(source_root / f"{row['tgt_id']}.txt")

    try:
        build_c089_manifest(
            source_manifest=source_manifest,
            source_root=source_root,
            output_root=tmp_path / "out_root",
            output_manifest=tmp_path / "out.jsonl",
            output_summary=tmp_path / "out.summary.json",
            heldout_summary=heldout_summary,
        )
    except SystemExit as error:
        assert "heldout leakage" in str(error)
    else:
        raise AssertionError("expected heldout leakage to stop c089 manifest creation")


def _row(ref_id: str, tgt_id: str) -> dict[str, str]:
    return {"ref_id": ref_id, "tgt_id": tgt_id, "prompt": "mrcolor_panel_style, safe"}


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "green").save(path)


def _write_caption(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("caption\n", encoding="utf-8")
