from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from tools import c090_siglip_hard_shape_eval
from tools.c090_siglip_hard_shape_data import BASELINE_VARIANTS, siglip_variants
from tools.siglip_auto_caption_types import EvalConfig, JsonObject, Sample, Variant


def _write_pattern(path: Path, base: tuple[int, int, int], accent: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64), base)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 42, 54), fill=accent)
    draw.line((4, 58, 60, 12), fill=(250, 250, 250), width=3)
    image.save(path)


def test_c090_runner_writes_hard_shape_outputs_without_comfy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "source/ref.jpg"
    _write_pattern(reference, (30, 90, 50), (170, 230, 80))
    baselines = {
        variant: tmp_path / "source" / f"{variant}.png"
        for variant in BASELINE_VARIANTS
    }
    for index, path in enumerate(baselines.values()):
        _write_pattern(path, (40 + index * 20, 70, 80), (180, 210 - index * 20, 90))
    source_summary = tmp_path / "source/summary.json"
    source_summary.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "label": "crop_pair00",
                        "prompt": "solo green yokai guard, hard silhouette",
                        "selected_attributes": ["green", "silhouette"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    probe_manifest = tmp_path / "probe.jsonl"
    probe_manifest.write_text(
        json.dumps(
            {
                "sample": "crop_pair00",
                "shape_group": "frog_yokai_guard",
                "reference_path": str(reference),
                "source_summary_path": str(source_summary),
                "candidates": {variant: str(path) for variant, path in baselines.items()},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_run_prompt(
        sample: Sample,
        variant: Variant,
        image_name: str,
        config: EvalConfig,
    ) -> JsonObject:
        calls.append(variant.label)
        output = config.out_dir / f"{sample.label}_{variant.label}.png"
        _write_pattern(output, (20, 80, 80), (200, 220, 90))
        return {
            "prompt_id": f"fake-{variant.label}-{image_name}",
            "image": str(output),
            "image_info": {"filename": output.name, "subfolder": "", "type": "output"},
        }

    monkeypatch.setattr(c090_siglip_hard_shape_eval, "run_prompt", fake_run_prompt)

    config = EvalConfig(
        data_root=tmp_path / "data",
        out_dir=tmp_path / "out",
        comfy_input=tmp_path / "comfy/input",
        comfy_output=tmp_path / "comfy/output",
        base_url="http://unused.invalid",
    )
    summary = c090_siglip_hard_shape_eval.run_c090_eval(probe_manifest, config)

    assert calls == [variant.label for variant in siglip_variants()]
    assert summary["contact_sheet"] == str(config.out_dir / "contact_sheet_hard_shape.jpg")
    assert (config.out_dir / "summary.json").is_file()
    assert (config.out_dir / "contact_sheet_hard_shape.jpg").is_file()
    assert (config.out_dir / "shape_metrics.json").is_file()
    assert (config.out_dir / "metric_rollup.json").is_file()
    assert (config.out_dir / "report.md").is_file()
    assert (config.out_dir / "pixel_nonblank_audit.json").is_file()

    audit = json.loads((config.out_dir / "pixel_nonblank_audit.json").read_text(encoding="utf-8"))
    rollup = json.loads((config.out_dir / "metric_rollup.json").read_text(encoding="utf-8"))
    report = (config.out_dir / "report.md").read_text(encoding="utf-8")
    assert audit["blank_count"] == 0
    assert audit["generated_count"] == len(siglip_variants())
    assert "c089_shape_w14" in rollup["variant_summaries"]
    assert "contact_sheet_hard_shape.jpg" in report
