from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from bench.ip_adapter import reference_eval as ref_eval


def _write_image(
    path: Path, color: tuple[int, int, int], size: tuple[int, int] = (32, 48)
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_select_reference_images_spreads_across_sorted_pool(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    for idx in range(5):
        _write_image(root / f"group-{idx}" / f"ref-{idx}.png", (idx, idx, idx))

    selected = ref_eval.select_reference_images(root, limit=3)

    assert [path.name for path in selected] == ["ref-0.png", "ref-2.png", "ref-4.png"]


def test_plan_writes_manifest_and_run_script_with_matching_geometry(
    tmp_path: Path,
) -> None:
    ref_root = tmp_path / "refs"
    _write_image(ref_root / "a" / "wide.png", (20, 30, 40), size=(80, 40))
    checkpoint = tmp_path / "adapter.safetensors"
    checkpoint.write_bytes(b"stub")
    out_dir = tmp_path / "eval"

    manifest = ref_eval.write_plan(
        ref_eval.PlanConfig(
            checkpoint=checkpoint,
            ref_root=ref_root,
            out_dir=out_dir,
            limit_refs=1,
            refs=None,
            seeds=(11,),
            scales=(0.5, 1.0),
            prompt="panel layout test",
            negative_prompt="bad",
            infer_steps=4,
            guidance_scale=3.5,
            flow_shift=1.0,
        )
    )

    manifest_path = out_dir / "manifest.json"
    run_script = out_dir / "run_eval.sh"
    assert manifest_path.exists()
    assert run_script.exists()

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded == manifest
    jobs = loaded["jobs"]
    assert len(jobs) == 3
    assert [job["mode"] for job in jobs] == ["no_ip", "ip", "ip"]
    assert all("--image_size" in job["command"] for job in jobs)
    image_sizes = {tuple(job["image_size"]) for job in jobs}
    assert len(image_sizes) == 1
    assert "--ip_adapter_weight" not in jobs[0]["command"]
    assert "--ip_adapter_weight" in jobs[1]["command"]
    assert "--ip_scale" in jobs[2]["command"]


def test_plan_accepts_explicit_refs_and_requires_checkpoint(tmp_path: Path) -> None:
    ref_root = tmp_path / "refs"
    ref_a = ref_root / "b" / "second.png"
    ref_b = ref_root / "a" / "first.png"
    _write_image(ref_a, (20, 30, 40))
    _write_image(ref_b, (40, 50, 60))
    checkpoint = tmp_path / "adapter.safetensors"
    checkpoint.write_bytes(b"stub")

    manifest = ref_eval.write_plan(
        ref_eval.PlanConfig(
            checkpoint=checkpoint,
            ref_root=ref_root,
            out_dir=tmp_path / "eval",
            limit_refs=1,
            refs=(ref_a, ref_b),
            seeds=(11,),
            scales=(0.5,),
            prompt="panel layout test",
            negative_prompt="bad",
            infer_steps=4,
            guidance_scale=3.5,
            flow_shift=1.0,
        )
    )

    assert [Path(ref["path"]).name for ref in manifest["refs"]] == [
        "second.png",
        "first.png",
    ]

    missing = tmp_path / "missing.safetensors"
    try:
        ref_eval.write_plan(
            ref_eval.PlanConfig(
                checkpoint=missing,
                ref_root=ref_root,
                out_dir=tmp_path / "missing_eval",
                limit_refs=1,
                refs=(ref_a,),
                seeds=(11,),
                scales=(0.5,),
                prompt="panel layout test",
                negative_prompt="bad",
                infer_steps=4,
                guidance_scale=3.5,
                flow_shift=1.0,
            )
        )
    except FileNotFoundError as exc:
        assert "Checkpoint not found" in str(exc)
    else:
        raise AssertionError("missing checkpoint should fail planning")


def test_summarize_scores_selects_best_scale_and_enforces_thresholds(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "out.png"
    _write_image(image_path, (100, 120, 140))
    rows = [
        ref_eval.ScoreRow(
            "ref0", 1, "no_ip", None, image_path, 0.40, 12.0, 100.0, 10, 240
        ),
        ref_eval.ScoreRow(
            "ref1", 1, "no_ip", None, image_path, 0.50, 12.0, 100.0, 10, 240
        ),
        ref_eval.ScoreRow("ref0", 1, "ip", 0.5, image_path, 0.44, 12.0, 100.0, 10, 240),
        ref_eval.ScoreRow("ref1", 1, "ip", 0.5, image_path, 0.54, 12.0, 100.0, 10, 240),
        ref_eval.ScoreRow("ref0", 1, "ip", 1.0, image_path, 0.42, 12.0, 100.0, 10, 240),
        ref_eval.ScoreRow("ref1", 1, "ip", 1.0, image_path, 0.51, 12.0, 100.0, 10, 240),
    ]

    summary = ref_eval.summarize_scores(
        rows,
        min_std=5.0,
        mean_uplift_threshold=0.03,
        improved_rate_threshold=0.75,
    )

    assert summary["pass"] is True
    assert summary["best_scale"] == 0.5
    assert summary["scale_summaries"][0]["mean_uplift"] > 0.03
    assert summary["scale_summaries"][0]["improved_rate"] == 1.0


def test_contact_sheet_and_report_are_written(tmp_path: Path) -> None:
    image_path = tmp_path / "out.png"
    _write_image(image_path, (80, 90, 100))
    rows = [
        ref_eval.ScoreRow(
            "ref0", 1, "no_ip", None, image_path, 0.40, 12.0, 90.0, 10, 240
        ),
        ref_eval.ScoreRow("ref0", 1, "ip", 0.5, image_path, 0.46, 12.0, 90.0, 10, 240),
    ]
    summary = ref_eval.summarize_scores(
        rows,
        min_std=5.0,
        mean_uplift_threshold=0.03,
        improved_rate_threshold=0.75,
    )
    manifest = {
        "checkpoint": "output/ckpt/anima_ip_adapter.safetensors",
        "out_dir": "output/bench/ip_adapter/test_eval",
        "refs": [{"ref_id": "ref0", "path": "ref.png"}],
        "seeds": [1],
        "scales": [0.5],
        "prompt": "prompt",
        "thresholds": {
            "mean_uplift": 0.03,
            "improved_rate": 0.75,
            "min_pixel_std": 5.0,
        },
    }

    sheet = tmp_path / "sheet.jpg"
    report = tmp_path / "report.md"
    ref_eval.write_contact_sheet(rows, sheet)
    ref_eval.write_report(manifest, summary, report)

    assert sheet.exists()
    text = report.read_text(encoding="utf-8")
    assert "PASS" in text
    assert "anima_ip_adapter.safetensors" in text
    assert "## Commands" in text
    assert "output/bench/ip_adapter/test_eval/manifest.json" in text
    assert "output/bench/ip_adapter/test_eval/run_eval.sh" in text
    assert "score --manifest" in text
