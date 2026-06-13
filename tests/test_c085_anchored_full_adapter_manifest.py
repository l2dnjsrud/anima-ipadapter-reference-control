from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c085_anchored_full_adapter_manifest import build_c085_manifest


def test_build_c085_manifest_balances_crop_and_anchor_rows(tmp_path: Path) -> None:
    c084_root = tmp_path / "c084_root"
    color_root = tmp_path / "color_root"
    c085_root = tmp_path / "c085_root"
    c084_manifest = tmp_path / "c084.jsonl"
    c060_manifest = tmp_path / "c060.jsonl"
    heldout_summary = tmp_path / "heldout.json"
    output_manifest = tmp_path / "out.jsonl"
    output_summary = tmp_path / "out.summary.json"
    c084_rows = [
        _row(f"external/c084/ref{i:03d}", f"external/c084/tgt{i:03d}")
        for i in range(80)
    ]
    c060_rows = [_row(f"clean/ref{i:03d}", f"clean/ref{i:03d}") for i in range(32)]
    c060_rows.extend(_row(f"positive/ref{i:03d}", f"positive/tgt{i:03d}") for i in range(58))
    c060_rows.extend(_row(f"failure/ref{i:03d}", f"failure/tgt{i:03d}") for i in range(64))
    _write_jsonl(c084_manifest, c084_rows)
    _write_jsonl(c060_manifest, c060_rows)
    heldout_summary.write_text(json.dumps({"heldout_ids": ["heldout/ref"]}), encoding="utf-8")
    for row in c084_rows:
        _write_image(c084_root / f"{row['ref_id']}.jpg")
        _write_image(c084_root / f"{row['tgt_id']}.jpg")
        _write_caption(c084_root / f"{row['tgt_id']}.txt")
    for row in c060_rows:
        _write_image(color_root / f"{row['ref_id']}.jpg")
        _write_image(color_root / f"{row['tgt_id']}.jpg")
        _write_caption(color_root / f"{row['tgt_id']}.txt")

    summary = build_c085_manifest(
        c084_manifest=c084_manifest,
        c060_manifest=c060_manifest,
        c084_root=c084_root,
        color_root=color_root,
        output_root=c085_root,
        output_manifest=output_manifest,
        output_summary=output_summary,
        heldout_summary=heldout_summary,
    )

    rows = _read_jsonl(output_manifest)
    assert summary.total_rows == 160
    assert summary.c084_crop_rows == 80
    assert summary.clean_anchor_rows == 32
    assert summary.c052_positive_anchor_rows == 16
    assert summary.failure_anchor_rows == 32
    assert summary.heldout_rows_used == 0
    assert summary.direct_self_pair_rows == 32
    assert len(rows) == 160
    assert (c085_root / f"{rows[0]['ref_id']}.jpg").is_symlink()
    assert (c085_root / f"{rows[0]['tgt_id']}.txt").is_symlink()
    written_summary = json.loads(output_summary.read_text(encoding="utf-8"))
    assert written_summary["decision"] == "ready_for_c085_full_adapter_training"


def test_build_c085_manifest_blocks_heldout_leakage(tmp_path: Path) -> None:
    c084_root = tmp_path / "c084_root"
    color_root = tmp_path / "color_root"
    c084_manifest = tmp_path / "c084.jsonl"
    c060_manifest = tmp_path / "c060.jsonl"
    heldout_summary = tmp_path / "heldout.json"
    c084_rows = [_row(f"external/c084/ref{i:03d}", f"external/c084/tgt{i:03d}") for i in range(80)]
    c060_rows = [_row(f"clean/ref{i:03d}", f"clean/ref{i:03d}") for i in range(154)]
    c060_rows[10] = _row("heldout/ref", "heldout/ref")
    _write_jsonl(c084_manifest, c084_rows)
    _write_jsonl(c060_manifest, c060_rows)
    heldout_summary.write_text(json.dumps({"heldout_ids": ["heldout/ref"]}), encoding="utf-8")
    for row in c084_rows:
        _write_image(c084_root / f"{row['ref_id']}.jpg")
        _write_image(c084_root / f"{row['tgt_id']}.jpg")
        _write_caption(c084_root / f"{row['tgt_id']}.txt")
    for row in c060_rows:
        _write_image(color_root / f"{row['ref_id']}.jpg")
        _write_image(color_root / f"{row['tgt_id']}.jpg")
        _write_caption(color_root / f"{row['tgt_id']}.txt")

    try:
        build_c085_manifest(
            c084_manifest=c084_manifest,
            c060_manifest=c060_manifest,
            c084_root=c084_root,
            color_root=color_root,
            output_root=tmp_path / "out_root",
            output_manifest=tmp_path / "out.jsonl",
            output_summary=tmp_path / "out.summary.json",
            heldout_summary=heldout_summary,
        )
    except SystemExit as error:
        assert "heldout leakage" in str(error)
    else:
        raise AssertionError("expected heldout leakage to stop manifest creation")


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
