from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c093_anti_collapse_manifest import build_c093_anti_collapse_manifest


def test_build_c093_manifest_adds_collapse_negative_and_excludes_heldout(
    tmp_path: Path,
) -> None:
    ref = tmp_path / "refs/crop_pair00.jpg"
    target = tmp_path / "targets/crop_pair00.png"
    negative = tmp_path / "gate/crop_pair00_c092_qwen_target_w14.png"
    heldout_negative = tmp_path / "gate/heldout07_c092_qwen_target_w14.png"
    for path, color in (
        (ref, (20, 100, 40)),
        (target, (40, 180, 80)),
        (negative, (50, 160, 70)),
        (heldout_negative, (60, 120, 70)),
    ):
        _write_image(path, color)
    c092_summary = tmp_path / "c092.summary.json"
    c092_summary.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "label": "crop_pair00",
                        "ref_id": "c092/source_ref",
                        "tgt_id": "c092/source_target",
                        "prompt": "green non-human character",
                        "reference_source": str(ref),
                        "target_source": str(target),
                    },
                    {
                        "label": "heldout07",
                        "ref_id": "heldout/ref",
                        "tgt_id": "heldout/target",
                        "prompt": "heldout",
                        "reference_source": str(ref),
                        "target_source": str(target),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = build_c093_anti_collapse_manifest(
        c092_manifest_summary=c092_summary,
        c092_gate_dir=tmp_path / "gate",
        output_root=tmp_path / "out_root",
        output_manifest=tmp_path / "out/c093.jsonl",
        output_summary=tmp_path / "out/c093.summary.json",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out/c093.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary.total_rows == 1
    assert summary.heldout_rows_used == 0
    assert summary.excluded_labels == ["heldout07"]
    assert summary.explicit_negative_rows == 1
    assert rows == [
        {
            "ref_id": "c093_anti_collapse/crop_pair00_ref",
            "tgt_id": "c093_anti_collapse/crop_pair00_target",
            "neg_id": "c093_anti_collapse/crop_pair00_c092_qwen_target_w14_negative",
            "prompt": "green non-human character",
            "source_label": "crop_pair00",
            "positive_teacher_variant": "c087_expanded_crop_positive_w14",
            "negative_variant": "c092_qwen_target_w14",
        }
    ]
    assert (tmp_path / "out_root/c093_anti_collapse/crop_pair00_ref.jpg").is_file()
    assert (tmp_path / "out_root/c093_anti_collapse/crop_pair00_target.jpg").is_file()
    assert (
        tmp_path / "out_root/c093_anti_collapse/crop_pair00_c092_qwen_target_w14_negative.jpg"
    ).is_file()
    assert (
        tmp_path / "out_root/c093_anti_collapse/crop_pair00_target.txt"
    ).read_text(encoding="utf-8").strip() == "green non-human character"


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)
