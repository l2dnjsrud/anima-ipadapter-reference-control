from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c089_pilot_report import build_c089_pilot_report


def test_build_c089_pilot_report_promotes_loadable_finite_checkpoint(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    image_root = tmp_path / "images"
    manifest = tmp_path / "manifest.jsonl"
    manifest_summary = tmp_path / "manifest.summary.json"
    checkpoint = tmp_path / "model.safetensors"
    eval_dir.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    rows = [_row("ref0", "tgt0"), _row("ref1", "tgt1")]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest_summary.write_text(
        json.dumps({"total_rows": 2, "heldout_rows_used": 0}),
        encoding="utf-8",
    )
    for row in rows:
        _write_image(image_root / f"{row['ref_id']}.jpg", "green")
        _write_image(image_root / f"{row['tgt_id']}.jpg", "blue")
    (eval_dir / "train_stdout.txt").write_text(
        json.dumps(
            {
                "steps": 32,
                "rows_loaded": 2,
                "first_loss": 0.4,
                "final_loss": 0.2,
                "mean_teacher_loss": 0.01,
                "mean_pe_token_loss": 0.2,
                "mean_pe_retrieval_loss": 0.3,
                "finite_loss": True,
                "checkpoint": {
                    "output_path": str(checkpoint),
                    "loadable": True,
                    "pe_checkpoint_rejected": True,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_c089_pilot_report(
        eval_dir=eval_dir,
        manifest_path=manifest,
        manifest_summary_path=manifest_summary,
        image_root=image_root,
    )

    assert summary["decision"] == "proceed_to_siglip_generation_gate"
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["loss_delta"] == -0.2
    assert (eval_dir / "report.md").is_file()
    assert Image.open(eval_dir / "contact_sheet.jpg").size[0] > 0


def _row(ref_id: str, tgt_id: str) -> dict[str, str]:
    return {"ref_id": ref_id, "tgt_id": tgt_id, "prompt": "safe"}


def _write_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)
