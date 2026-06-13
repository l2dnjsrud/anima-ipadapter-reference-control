from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c092_qwen_target_manifest import materialize_c092_qwen_target_manifest


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def test_materialize_c092_manifest_excludes_heldout_and_writes_caption(tmp_path: Path) -> None:
    ref_train = tmp_path / "ref/train.jpg"
    ref_heldout = tmp_path / "ref/heldout.jpg"
    target_train = tmp_path / "target/train.png"
    target_heldout = tmp_path / "target/heldout.png"
    _write_image(ref_train, (10, 100, 40))
    _write_image(ref_heldout, (20, 90, 50))
    _write_image(target_train, (30, 200, 80))
    _write_image(target_heldout, (40, 210, 90))
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "label": "crop_pair00",
                        "reference_path": str(ref_train),
                        "prompt_row": {"prompt": "green non-human shape"},
                    },
                    {
                        "label": "heldout07",
                        "reference_path": str(ref_heldout),
                        "prompt_row": {"prompt": "heldout prompt"},
                    },
                ],
                "baseline_candidates": {
                    "crop_pair00": {"teacher": str(target_train)},
                    "heldout07": {"teacher": str(target_heldout)},
                },
            }
        ),
        encoding="utf-8",
    )

    payload = materialize_c092_qwen_target_manifest(
        summary,
        out_manifest=tmp_path / "manifest.jsonl",
        image_root=tmp_path / "root",
        teacher_variant="teacher",
    )

    rows = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert payload["total_rows"] == 1
    assert payload["excluded_labels"] == ["heldout07"]
    assert len(rows) == 1
    assert "crop_pair00" in rows[0]
    assert (tmp_path / "root/c092_qwen_target/crop_pair00_ref.jpg").is_file()
    assert (tmp_path / "root/c092_qwen_target/crop_pair00_teacher.jpg").is_file()
    assert (
        tmp_path / "root/c092_qwen_target/crop_pair00_teacher.txt"
    ).read_text(encoding="utf-8").strip() == "green non-human shape"
