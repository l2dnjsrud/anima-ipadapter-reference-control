from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.c080_paired_direct_green_manifest import (
    C080ManifestConfig,
    C080ManifestError,
    build_c080_paired_direct_green_manifest,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c080_builds_paired_c074_rows_with_no_direct_self_pairs(tmp_path: Path) -> None:
    source_manifest, source_root = _source_fixture(tmp_path)
    c074_labels = tmp_path / "c074.jsonl"
    c078_labels = tmp_path / "c078.jsonl"
    c077_labels = tmp_path / "c077.jsonl"
    scratch_root = tmp_path / ".tmp" / "c080_root"
    output_manifest = tmp_path / "out" / "c080.jsonl"
    output_summary = tmp_path / "out" / "c080.summary.json"
    report_path = tmp_path / "eval" / "manifest_report.md"

    for name in ("real_a", "real_b", "real_c", "synth_a", "guard_a"):
        _write_image(tmp_path / "downloads" / f"{name}.jpg")
    _write_jsonl(
        c074_labels,
        (
            _label("real_a", tmp_path / "downloads" / "real_a.jpg"),
            _label("real_b", tmp_path / "downloads" / "real_b.jpg"),
            _label("real_c", tmp_path / "downloads" / "real_c.jpg"),
        ),
    )
    _write_jsonl(c078_labels, (_label("synth_a", tmp_path / "downloads" / "synth_a.jpg"),))
    _write_jsonl(
        c077_labels,
        (_label("guard_a", tmp_path / "downloads" / "guard_a.jpg", label="useful_proxy_non_human"),),
    )

    summary = build_c080_paired_direct_green_manifest(
        C080ManifestConfig(
            source_manifest_path=source_manifest,
            source_image_root=source_root,
            c074_labels_path=c074_labels,
            c078_labels_path=c078_labels,
            c077_labels_path=c077_labels,
            scratch_image_root=scratch_root,
            output_manifest_path=output_manifest,
            output_summary_path=output_summary,
            output_report_path=report_path,
            c074_pair_repeat=2,
            guard_repeat=1,
            source_row_limit=1,
            minimum_c074_pair_sources=3,
        )
    )

    rows = _read_jsonl(output_manifest)
    direct_rows = [
        row
        for row in rows
        if str(row["ref_id"]).startswith("external/c080_c074_paired/")
    ]
    assert len(direct_rows) == 6
    assert all(row["ref_id"] != row["tgt_id"] for row in direct_rows)
    assert summary.c074_paired_training_rows == 6
    assert summary.c078_unpaired_positive_count == 1
    assert summary.c078_training_rows == 0
    assert summary.guard_proxy_training_rows == 1
    assert summary.source_training_rows == 1
    assert summary.heldout_rows_used == 0
    assert (scratch_root / "external" / "c080_c074_paired" / "real_a.jpg").is_file()
    assert (scratch_root / "external" / "c080_c074_paired" / "real_b.txt").is_file()
    assert report_path.is_file()


def test_c080_rejects_too_few_c074_pair_sources(tmp_path: Path) -> None:
    source_manifest, source_root = _source_fixture(tmp_path)
    c074_labels = tmp_path / "c074.jsonl"
    c078_labels = tmp_path / "c078.jsonl"
    c077_labels = tmp_path / "c077.jsonl"
    _write_image(tmp_path / "downloads" / "real.jpg")
    _write_jsonl(c074_labels, (_label("real", tmp_path / "downloads" / "real.jpg"),))
    _write_jsonl(c078_labels, ())
    _write_jsonl(c077_labels, ())

    with pytest.raises(C080ManifestError, match="c074 paired sources"):
        build_c080_paired_direct_green_manifest(
            C080ManifestConfig(
                source_manifest_path=source_manifest,
                source_image_root=source_root,
                c074_labels_path=c074_labels,
                c078_labels_path=c078_labels,
                c077_labels_path=c077_labels,
                scratch_image_root=tmp_path / ".tmp" / "root",
                output_manifest_path=tmp_path / "out.jsonl",
                output_summary_path=tmp_path / "out.summary.json",
                output_report_path=tmp_path / "report.md",
                minimum_c074_pair_sources=2,
            )
        )


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "source_root"
    source_manifest = tmp_path / "source.jsonl"
    _write_asset(source_root, "source/a")
    _write_jsonl(source_manifest, ({"ref_id": "source/a", "tgt_id": "source/a", "prompt": "source a"},))
    return source_manifest, source_root


def _write_asset(source_root: Path, image_id: str) -> None:
    image_path = source_root / f"{image_id}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"jpg")
    image_path.with_suffix(".txt").write_text("source caption\n", encoding="utf-8")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")


def _label(candidate_id: str, image_path: Path, *, label: str = "target_positive") -> JsonObject:
    return {
        "candidate_id": candidate_id,
        "download_status": "downloaded",
        "local_image_path": str(image_path),
        "manual_label": label,
        "external_license_note": "test license",
    }


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
