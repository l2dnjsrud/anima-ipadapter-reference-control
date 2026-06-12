from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.c075_tag_positive_manifest import (
    C075ManifestConfig,
    C075ManifestError,
    build_c075_tag_positive_manifest,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c075_builds_tag_positive_manifest_with_external_rows_first(tmp_path: Path) -> None:
    source_root = tmp_path / "source_root"
    source_manifest = tmp_path / "c060.jsonl"
    c074_labels = tmp_path / "c074.jsonl"
    c073_labels = tmp_path / "c073.jsonl"
    output_manifest = tmp_path / "out" / "c075.jsonl"
    output_summary = tmp_path / "out" / "c075.summary.json"
    report_path = tmp_path / "eval" / "manifest_report.md"
    scratch_root = tmp_path / ".tmp" / "c075_root"

    for image_id in ("source/a", "source/b"):
        _write_asset(source_root, image_id)
    _write_jsonl(
        source_manifest,
        (
            {"ref_id": "source/a", "tgt_id": "source/a", "prompt": "source a"},
            {"ref_id": "source/b", "tgt_id": "source/b", "prompt": "source b"},
            {"ref_id": "source/c", "tgt_id": "source/c", "prompt": "limited out"},
        ),
    )
    _write_external_image(tmp_path / "downloads" / "c074_a.jpg")
    _write_external_image(tmp_path / "downloads" / "c074_b.jpg")
    _write_jsonl(
        c074_labels,
        (
            _c074_row("c074_a", tmp_path / "downloads" / "c074_a.jpg"),
            _c074_row("c074_b", tmp_path / "downloads" / "c074_b.jpg"),
            _c074_row("c074_skip", tmp_path / "downloads" / "missing.jpg", label="useful_proxy_non_human"),
        ),
    )
    _write_jsonl(
        c073_labels,
        (
            {"manual_label": "guard_false_positive_human"},
            {"manual_label": "useful_proxy_non_human"},
        ),
    )

    summary = build_c075_tag_positive_manifest(
        C075ManifestConfig(
            source_manifest_path=source_manifest,
            source_image_root=source_root,
            c074_labels_path=c074_labels,
            c073_labels_path=c073_labels,
            scratch_image_root=scratch_root,
            output_manifest_path=output_manifest,
            output_summary_path=output_summary,
            output_report_path=report_path,
            positive_repeat=2,
            source_row_limit=2,
            minimum_target_positives=2,
        )
    )

    rows = _read_jsonl(output_manifest)
    assert [row["ref_id"] for row in rows[:4]] == [
        "external/c074_direct_green/c074_a",
        "external/c074_direct_green/c074_b",
        "external/c074_direct_green/c074_a",
        "external/c074_direct_green/c074_b",
    ]
    assert [row["ref_id"] for row in rows[4:]] == ["source/a", "source/b"]
    assert summary.target_positive_count == 2
    assert summary.target_positive_training_rows == 4
    assert summary.source_training_rows == 2
    assert summary.total_rows == 6
    assert summary.heldout_rows_used == 0
    assert summary.missing_paths == 0
    assert summary.committed_external_image_count == 0
    assert summary.c073_guard_label_counts == {
        "guard_false_positive_human": 1,
        "useful_proxy_non_human": 1,
    }
    saved_summary = json.loads(output_summary.read_text(encoding="utf-8"))
    assert saved_summary["missing_paths"] == 0
    assert report_path.is_file()
    assert (scratch_root / "source" / "a.jpg").is_file()
    assert (scratch_root / "source" / "a.txt").is_file()
    assert (scratch_root / "external" / "c074_direct_green" / "c074_a.jpg").is_file()
    assert (scratch_root / "external" / "c074_direct_green" / "c074_a.txt").is_file()


def test_c075_rejects_too_few_target_positives(tmp_path: Path) -> None:
    source_manifest, source_root = _source_fixture(tmp_path)
    c074_labels = tmp_path / "c074.jsonl"
    _write_external_image(tmp_path / "downloads" / "only.jpg")
    _write_jsonl(c074_labels, (_c074_row("only", tmp_path / "downloads" / "only.jpg"),))

    with pytest.raises(C075ManifestError, match="target positives"):
        build_c075_tag_positive_manifest(
            C075ManifestConfig(
                source_manifest_path=source_manifest,
                source_image_root=source_root,
                c074_labels_path=c074_labels,
                c073_labels_path=None,
                scratch_image_root=tmp_path / ".tmp" / "root",
                output_manifest_path=tmp_path / "out.jsonl",
                output_summary_path=tmp_path / "out.summary.json",
                output_report_path=tmp_path / "report.md",
                minimum_target_positives=2,
            )
        )


def test_c075_rejects_source_row_missing_caption(tmp_path: Path) -> None:
    source_root = tmp_path / "source_root"
    source_manifest = tmp_path / "c060.jsonl"
    c074_labels = tmp_path / "c074.jsonl"
    image_path = source_root / "source" / "missing_caption.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"jpg")
    _write_jsonl(
        source_manifest,
        ({"ref_id": "source/missing_caption", "tgt_id": "source/missing_caption", "prompt": "bad"},),
    )
    _write_external_image(tmp_path / "downloads" / "c074_a.jpg")
    _write_jsonl(c074_labels, (_c074_row("c074_a", tmp_path / "downloads" / "c074_a.jpg"),))

    with pytest.raises(C075ManifestError, match="missing source asset"):
        build_c075_tag_positive_manifest(
            C075ManifestConfig(
                source_manifest_path=source_manifest,
                source_image_root=source_root,
                c074_labels_path=c074_labels,
                c073_labels_path=None,
                scratch_image_root=tmp_path / ".tmp" / "root",
                output_manifest_path=tmp_path / "out.jsonl",
                output_summary_path=tmp_path / "out.summary.json",
                output_report_path=tmp_path / "report.md",
                source_row_limit=1,
                minimum_target_positives=1,
            )
        )


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "source_root"
    source_manifest = tmp_path / "c060.jsonl"
    _write_asset(source_root, "source/a")
    _write_jsonl(source_manifest, ({"ref_id": "source/a", "tgt_id": "source/a", "prompt": "source a"},))
    return source_manifest, source_root


def _write_asset(source_root: Path, image_id: str) -> None:
    image_path = source_root / f"{image_id}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"jpg")
    image_path.with_suffix(".txt").write_text("source caption\n", encoding="utf-8")


def _write_external_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"jpg")


def _c074_row(candidate_id: str, image_path: Path, *, label: str = "target_positive") -> JsonObject:
    return {
        "candidate_id": candidate_id,
        "image_id": f"external/{candidate_id}",
        "download_status": "downloaded",
        "local_image_path": str(image_path),
        "manual_label": label,
        "source_labels": ["green_skin", "monster_girl"],
    }


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)
