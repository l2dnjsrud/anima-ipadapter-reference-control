from __future__ import annotations

import json
from pathlib import Path

from tools.c099_real_color_data_gate import C099Config, build_c099_real_color_gate
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c099_gate_excludes_heldout_and_blocks_without_real_direct_green(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "train/a.jpg")
    _touch(dataset_root / "train/b.jpg")
    _touch(dataset_root / "heldout/z.jpg")
    external = tmp_path / "external_green.jpg"
    hard_shape = tmp_path / "c097_root/c097_hard_shape/pair_000_ref.jpg"
    _touch(external)
    _touch(hard_shape)

    out_dir = tmp_path / "out"
    summary = build_c099_real_color_gate(
        _config(
            tmp_path,
            out_dir,
            dataset_root=dataset_root,
            c052_rows=(
                {"ref_id": "train/a", "tgt_id": "train/b", "prompt": "same character"},
                {"ref_id": "heldout/z", "tgt_id": "train/a", "prompt": "leak"},
            ),
            c066_rows=(
                _c066_row("train/a", dataset_root / "train/a.jpg", "direct_green_pixel_candidate"),
                _c066_row("heldout/z", dataset_root / "heldout/z.jpg", "direct_green_attribute"),
            ),
            c074_rows=(_external_row(external),),
            c097_rows=(
                {
                    "ref_id": "c097_hard_shape/pair_000_ref",
                    "tgt_id": "c097_hard_shape/pair_000_target",
                    "neg_id": "c097_hard_shape/pair_000_negative",
                    "prompt": "shape",
                    "shape_group": "frog",
                },
            ),
        )
    )

    manifest_rows = _read_jsonl(out_dir / "c099_candidate_manifest.jsonl")
    image_ids = {str(row["ref_id"]) for row in manifest_rows}
    assert "heldout/z" not in image_ids
    assert summary["heldout_leakage_count"] == 0
    assert summary["omitted_heldout_rows"] == 2
    assert summary["real_local_direct_green_confirmed_rows"] == 0
    assert summary["external_direct_green_positive_rows"] == 1
    assert summary["synthetic_hard_shape_rows"] == 1
    assert summary["decision"] == "c100_blocked_needs_annotation_or_teacher"
    assert (out_dir / "inventory.json").is_file()
    assert "annotation" in str(summary["blocker_reason"])


def test_c099_gate_greenlights_when_real_direct_green_exists(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _touch(dataset_root / "train/a.jpg")
    _touch(dataset_root / "train/b.jpg")
    out_dir = tmp_path / "out"

    summary = build_c099_real_color_gate(
        _config(
            tmp_path,
            out_dir,
            dataset_root=dataset_root,
            c066_rows=(
                _c066_row("train/a", dataset_root / "train/a.jpg", "direct_green_attribute"),
                _c066_row("train/b", dataset_root / "train/b.jpg", "human_negative"),
            ),
        )
    )

    assert summary["real_local_direct_green_confirmed_rows"] == 1
    assert summary["decision"] == "c100_training_greenlit"
    assert "training/" in str(summary["next_c100_command_surface"])


def _config(
    tmp_path: Path,
    out_dir: Path,
    *,
    dataset_root: Path,
    c052_rows: tuple[JsonObject, ...] = (),
    c066_rows: tuple[JsonObject, ...] = (),
    c074_rows: tuple[JsonObject, ...] = (),
    c097_rows: tuple[JsonObject, ...] = (),
) -> C099Config:
    train_manifest = tmp_path / "train.jsonl"
    heldout_manifest = tmp_path / "heldout.jsonl"
    _write_jsonl(train_manifest, ({"ref_id": "train/a", "tgt_id": "train/a", "prompt": "a"}, {"ref_id": "train/b", "tgt_id": "train/b", "prompt": "b"}))
    _write_jsonl(heldout_manifest, ({"ref_id": "heldout/z", "tgt_id": "heldout/z", "prompt": "z"},))
    clean_summary = tmp_path / "clean.summary.json"
    clean_summary.write_text(
        json.dumps({"dataset_root": str(dataset_root), "train_rows": 2, "heldout_rows": 1}),
        encoding="utf-8",
    )
    paths = {
        "c052": c052_rows,
        "c066": c066_rows,
        "c074": c074_rows,
        "c097": c097_rows,
    }
    written: dict[str, Path] = {}
    for name, rows in paths.items():
        path = tmp_path / f"{name}.jsonl"
        _write_jsonl(path, rows)
        written[name] = path
    c097_summary = tmp_path / "c097.summary.json"
    c097_summary.write_text(
        json.dumps({"selected_rows": len(c097_rows), "heldout_rows_used": 0}),
        encoding="utf-8",
    )
    return C099Config(
        dataset_root=dataset_root,
        train_manifest=train_manifest,
        heldout_manifest=heldout_manifest,
        clean32_summary=clean_summary,
        c052_manifest=written["c052"],
        c066_manifest=written["c066"],
        c074_labels=written["c074"],
        c097_manifest=written["c097"],
        c097_summary=c097_summary,
        c097_root=tmp_path / "c097_root",
        out_dir=out_dir,
        plan_path=tmp_path / "plan.md",
    )


def _c066_row(image_id: str, image_path: Path, bucket: str) -> JsonObject:
    return {
        "image_id": image_id,
        "label": "positive" if bucket != "human_negative" else "negative",
        "source_bucket": bucket,
        "image_path": str(image_path),
        "caption": "caption",
    }


def _external_row(image_path: Path) -> JsonObject:
    return {
        "candidate_id": "external_green",
        "manual_label": "target_positive",
        "local_image_path": str(image_path),
        "external_license_note": "test",
    }


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return rows


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
