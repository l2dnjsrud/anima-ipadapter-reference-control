from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tools.generate_pair_manifest import (
    DatasetLayoutError,
    app,
    build_manifest,
    build_rows,
    generate_manifest,
    write_manifest,
)


RUNNER = CliRunner()


def _write_pair(dataset_root: Path, relative_stem: str, caption: str) -> None:
    image_path = dataset_root / f"{relative_stem}.jpg"
    caption_path = dataset_root / f"{relative_stem}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"jpg")
    caption_path.write_text(caption, encoding="utf-8")


def test_build_rows_pairs_next_image_with_nested_relative_ids(tmp_path: Path) -> None:
    dataset_root = tmp_path / "image_dataset"
    _write_pair(dataset_root, "MS-138/MS-138__MS_138-01_YMSS", "caption a\n")
    _write_pair(dataset_root, "MS-138/MS-138__MS_138-02_YMSS", "caption b\n")
    _write_pair(dataset_root, "MS-138/MS-138__MS_138-03_THSZ", "caption c\n")
    _write_pair(dataset_root, "nested/solo/item-01", "solo caption\n")

    rows = build_rows(dataset_root)

    assert [
        {"ref_id": row.ref_id, "tgt_id": row.tgt_id, "prompt": row.prompt}
        for row in rows
    ] == [
        {
            "ref_id": "MS-138/MS-138__MS_138-01_YMSS",
            "tgt_id": "MS-138/MS-138__MS_138-02_YMSS",
            "prompt": "caption b",
        },
        {
            "ref_id": "MS-138/MS-138__MS_138-02_YMSS",
            "tgt_id": "MS-138/MS-138__MS_138-03_THSZ",
            "prompt": "caption c",
        },
        {
            "ref_id": "MS-138/MS-138__MS_138-03_THSZ",
            "tgt_id": "MS-138/MS-138__MS_138-01_YMSS",
            "prompt": "caption a",
        },
        {
            "ref_id": "nested/solo/item-01",
            "tgt_id": "nested/solo/item-01",
            "prompt": "solo caption",
        },
    ]


def test_build_manifest_skips_singletons_by_default_for_paired_training(tmp_path: Path) -> None:
    dataset_root = tmp_path / "image_dataset"
    _write_pair(dataset_root, "group/item-01", "caption 1\n")
    _write_pair(dataset_root, "group/item-02", "caption 2\n")
    _write_pair(dataset_root, "solo/item-01", "solo caption\n")

    result = build_manifest(dataset_root)

    assert result.directories == 2
    assert result.source_images == 3
    assert result.skipped_singleton_directories == 1
    assert [
        {"ref_id": row.ref_id, "tgt_id": row.tgt_id, "prompt": row.prompt}
        for row in result.rows
    ] == [
        {
            "ref_id": "group/item-01",
            "tgt_id": "group/item-02",
            "prompt": "caption 2",
        },
        {
            "ref_id": "group/item-02",
            "tgt_id": "group/item-01",
            "prompt": "caption 1",
        },
    ]


def test_build_manifest_orders_directories_and_images_deterministically(tmp_path: Path) -> None:
    dataset_root = tmp_path / "image_dataset"
    _write_pair(dataset_root, "z/item-02", "z caption 2\n")
    _write_pair(dataset_root, "a/item-02", "a caption 2\n")
    _write_pair(dataset_root, "z/item-01", "z caption 1\n")
    _write_pair(dataset_root, "a/item-01", "a caption 1\n")

    result = build_manifest(dataset_root)

    assert [row.ref_id for row in result.rows] == [
        "a/item-01",
        "a/item-02",
        "z/item-01",
        "z/item-02",
    ]


def test_build_manifest_allows_self_pairs_when_requested(tmp_path: Path) -> None:
    dataset_root = tmp_path / "image_dataset"
    _write_pair(dataset_root, "solo/item-01", "solo caption\n")

    result = build_manifest(dataset_root, allow_self_pairs=True)

    assert result.skipped_singleton_directories == 0
    assert [
        {"ref_id": row.ref_id, "tgt_id": row.tgt_id, "prompt": row.prompt}
        for row in result.rows
    ] == [
        {
            "ref_id": "solo/item-01",
            "tgt_id": "solo/item-01",
            "prompt": "solo caption",
        },
    ]


def test_build_manifest_rejects_duplicate_image_ids(tmp_path: Path) -> None:
    dataset_root = tmp_path / "image_dataset"
    image_path = dataset_root / "group" / "item-01.jpg"
    duplicate_path = dataset_root / "group" / "item-01.JPG"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"jpg")
    duplicate_path.write_bytes(b"jpg")
    image_path.with_suffix(".txt").write_text("caption\n", encoding="utf-8")

    with pytest.raises(DatasetLayoutError, match="Duplicate image id"):
        build_manifest(dataset_root)


def test_build_rows_rejects_missing_caption_sidecar(tmp_path: Path) -> None:
    dataset_root = tmp_path / "image_dataset"
    image_path = dataset_root / "broken" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"jpg")

    with pytest.raises(DatasetLayoutError, match="Missing caption sidecar"):
        build_rows(dataset_root)


def test_write_manifest_emits_jsonl_rows(tmp_path: Path) -> None:
    dataset_root = tmp_path / "image_dataset"
    _write_pair(dataset_root, "group/item-01", "caption 1\n")
    _write_pair(dataset_root, "group/item-02", "caption 2\n")
    output_path = tmp_path / "pairs.jsonl"

    rows = build_rows(dataset_root)
    write_manifest(rows, output_path)

    loaded = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert loaded == [
        {
            "ref_id": "group/item-01",
            "tgt_id": "group/item-02",
            "prompt": "caption 2",
        },
        {
            "ref_id": "group/item-02",
            "tgt_id": "group/item-01",
            "prompt": "caption 1",
        },
    ]


def test_generate_manifest_writes_summary_with_audit_and_split(tmp_path: Path) -> None:
    dataset_root = tmp_path / "image_dataset"
    for index in range(4):
        _write_pair(dataset_root, f"group/item-{index + 1:02d}", f"caption {index + 1}\n")
    output_path = tmp_path / "pairs.jsonl"
    summary_path = tmp_path / "pairs.summary.json"

    summary = generate_manifest(
        dataset_root,
        output_path,
        count_only=False,
        dry_run=False,
        allow_self_pairs=False,
        limit=None,
    )

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.output_path == str(output_path)
    assert loaded["summary_path"] == str(summary_path)
    assert loaded["caption_count"] == 4
    assert loaded["missing_captions"] == []
    assert loaded["duplicate_ids"] == []
    assert loaded["split"] == {
        "strategy": "sorted_95_5",
        "train": {"start_index": 0, "end_index_exclusive": 3, "rows": 3},
        "validation": {"start_index": 3, "end_index_exclusive": 4, "rows": 1},
    }


def test_cli_count_only_prints_summary_without_writing_output(tmp_path: Path) -> None:
    dataset_root = tmp_path / "image_dataset"
    _write_pair(dataset_root, "group/item-01", "caption 1\n")
    _write_pair(dataset_root, "group/item-02", "caption 2\n")
    output_path = tmp_path / "pairs.jsonl"

    result = RUNNER.invoke(app, [str(dataset_root), "--count-only"])

    assert result.exit_code == 0
    assert '"rows": 2' in result.stdout
    assert '"source_images": 2' in result.stdout
    assert not output_path.exists()


def test_cli_limit_caps_written_rows(tmp_path: Path) -> None:
    dataset_root = tmp_path / "image_dataset"
    _write_pair(dataset_root, "group/item-01", "caption 1\n")
    _write_pair(dataset_root, "group/item-02", "caption 2\n")
    _write_pair(dataset_root, "group/item-03", "caption 3\n")
    output_path = tmp_path / "pairs.jsonl"

    result = RUNNER.invoke(
        app,
        [str(dataset_root), "--output", str(output_path), "--limit", "2"],
    )

    assert result.exit_code == 0
    loaded = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(loaded) == 2
