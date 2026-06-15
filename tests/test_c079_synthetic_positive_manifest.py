from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.c079_synthetic_positive_manifest import (
    C079ManifestConfig,
    C079ManifestError,
    build_c079_synthetic_positive_manifest,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c079_builds_manifest_with_real_synthetic_and_guard_rows(
    tmp_path: Path,
) -> None:
    source_manifest, source_root = _source_fixture(tmp_path)
    c074_labels = tmp_path / "c074.jsonl"
    c078_labels = tmp_path / "c078.jsonl"
    c077_labels = tmp_path / "c077.jsonl"
    output_manifest = tmp_path / "out" / "c079.jsonl"
    output_summary = tmp_path / "out" / "c079.summary.json"
    report_path = tmp_path / "eval" / "manifest_report.md"
    scratch_root = tmp_path / ".tmp" / "c079_root"
    _write_external_image(tmp_path / "downloads" / "real_a.jpg")
    _write_external_image(tmp_path / "downloads" / "synth_a.png")
    _write_external_image(tmp_path / "downloads" / "proxy_a.jpg")
    _write_external_image(tmp_path / "downloads" / "guard_a.jpg")
    _write_jsonl(c074_labels, (_label("real_a", tmp_path / "downloads" / "real_a.jpg"),))
    _write_jsonl(c078_labels, (_label("synth_a", tmp_path / "downloads" / "synth_a.png"),))
    _write_jsonl(
        c077_labels,
        (
            _label("proxy_a", tmp_path / "downloads" / "proxy_a.jpg", label="useful_proxy_non_human"),
            _label("guard_a", tmp_path / "downloads" / "guard_a.jpg", label="guard_false_positive_human"),
        ),
    )

    summary = build_c079_synthetic_positive_manifest(
        C079ManifestConfig(
            source_manifest_path=source_manifest,
            source_image_root=source_root,
            c074_labels_path=c074_labels,
            c078_labels_path=c078_labels,
            c077_labels_path=c077_labels,
            scratch_image_root=scratch_root,
            output_manifest_path=output_manifest,
            output_summary_path=output_summary,
            output_report_path=report_path,
            source_row_limit=1,
            positive_repeat=2,
            guard_repeat=1,
            minimum_total_target_positives=2,
        )
    )

    rows = _read_jsonl(output_manifest)
    assert [row["ref_id"] for row in rows] == [
        "external/c074_real_direct_green/real_a",
        "external/c078_synthetic_direct_green/synth_a",
        "external/c074_real_direct_green/real_a",
        "external/c078_synthetic_direct_green/synth_a",
        "external/c077_guard_proxy/proxy_a",
        "external/c077_guard_proxy/guard_a",
        "source/a",
    ]
    assert summary.c074_real_positive_count == 1
    assert summary.c078_synthetic_target_positive_count == 1
    assert summary.guard_proxy_count == 2
    assert summary.guard_proxy_training_rows == 2
    assert summary.target_positive_training_rows == 4
    assert summary.total_rows == 7
    assert summary.heldout_rows_used == 0
    assert (scratch_root / "external" / "c078_synthetic_direct_green" / "synth_a.jpg").is_file()
    assert (scratch_root / "external" / "c077_guard_proxy" / "guard_a.txt").is_file()
    assert report_path.is_file()


def test_c079_rejects_too_few_total_target_positives(tmp_path: Path) -> None:
    source_manifest, source_root = _source_fixture(tmp_path)
    c074_labels = tmp_path / "c074.jsonl"
    c078_labels = tmp_path / "c078.jsonl"
    c077_labels = tmp_path / "c077.jsonl"
    _write_external_image(tmp_path / "downloads" / "only.jpg")
    _write_jsonl(c074_labels, (_label("only", tmp_path / "downloads" / "only.jpg"),))
    _write_jsonl(c078_labels, ())
    _write_jsonl(c077_labels, ())

    with pytest.raises(C079ManifestError, match="total target positives"):
        build_c079_synthetic_positive_manifest(
            C079ManifestConfig(
                source_manifest_path=source_manifest,
                source_image_root=source_root,
                c074_labels_path=c074_labels,
                c078_labels_path=c078_labels,
                c077_labels_path=c077_labels,
                scratch_image_root=tmp_path / ".tmp" / "root",
                output_manifest_path=tmp_path / "out.jsonl",
                output_summary_path=tmp_path / "out.summary.json",
                output_report_path=tmp_path / "report.md",
                minimum_total_target_positives=2,
            )
        )


def test_c079_rejects_missing_guard_proxy_image(tmp_path: Path) -> None:
    source_manifest, source_root = _source_fixture(tmp_path)
    c074_labels = tmp_path / "c074.jsonl"
    c078_labels = tmp_path / "c078.jsonl"
    c077_labels = tmp_path / "c077.jsonl"
    _write_external_image(tmp_path / "downloads" / "real.jpg")
    _write_external_image(tmp_path / "downloads" / "synth.jpg")
    _write_jsonl(c074_labels, (_label("real", tmp_path / "downloads" / "real.jpg"),))
    _write_jsonl(c078_labels, (_label("synth", tmp_path / "downloads" / "synth.jpg"),))
    _write_jsonl(
        c077_labels,
        (_label("missing", tmp_path / "downloads" / "missing.jpg", label="useful_proxy_non_human"),),
    )

    with pytest.raises(C079ManifestError, match="missing c077 guard/proxy image"):
        build_c079_synthetic_positive_manifest(
            C079ManifestConfig(
                source_manifest_path=source_manifest,
                source_image_root=source_root,
                c074_labels_path=c074_labels,
                c078_labels_path=c078_labels,
                c077_labels_path=c077_labels,
                scratch_image_root=tmp_path / ".tmp" / "root",
                output_manifest_path=tmp_path / "out.jsonl",
                output_summary_path=tmp_path / "out.summary.json",
                output_report_path=tmp_path / "report.md",
                minimum_total_target_positives=2,
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


def _write_external_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")


def _label(candidate_id: str, image_path: Path, *, label: str = "target_positive") -> JsonObject:
    return {
        "candidate_id": candidate_id,
        "status": "generated",
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
