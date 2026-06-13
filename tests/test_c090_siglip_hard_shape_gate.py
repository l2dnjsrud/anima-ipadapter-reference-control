from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c090_siglip_hard_shape_data import (
    C089_CHECKPOINT,
    materialize_c090_prompt_manifest,
    siglip_variants,
)
from tools.c090_siglip_hard_shape_report import summarize_shape_rows


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def test_materialize_manifest_preserves_reference_and_c089_variant(tmp_path: Path) -> None:
    reference = tmp_path / "ref.jpg"
    candidate = tmp_path / "candidate.png"
    _write_image(reference, (10, 100, 30))
    _write_image(candidate, (20, 90, 40))
    source_summary = tmp_path / "source_summary.json"
    source_summary.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "label": "crop_pair00",
                        "prompt": "solo green yokai guard, chibi shape",
                        "selected_attributes": ["green", "yokai"],
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
                "split": "direct_green",
                "shape_group": "frog_yokai_guard",
                "reference_path": str(reference),
                "source_summary_path": str(source_summary),
                "candidates": {
                    "blend_species_face": str(candidate),
                    "c086_hard_negative_w14": str(candidate),
                    "c087_expanded_crop_positive_w14": str(candidate),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples, baseline_candidates = materialize_c090_prompt_manifest(
        probe_manifest,
        out_dir=tmp_path / "out",
        reference_root=tmp_path / "root",
    )

    assert samples[0].label == "crop_pair00"
    assert samples[0].prompt_row.prompt == "solo green yokai guard, chibi shape"
    assert (tmp_path / "root/c090_refs/crop_pair00.jpg").is_file()
    assert baseline_candidates["crop_pair00"]["blend_species_face"] == str(candidate)
    assert any(variant.checkpoint == C089_CHECKPOINT for variant in siglip_variants())


def test_summarize_shape_rows_prefers_c089_when_it_beats_pilot() -> None:
    rows = [
        {"variant": "siglip_pilot_w14", "uplift": 0.02},
        {"variant": "siglip_pilot_w14", "uplift": 0.01},
        {"variant": "c089_shape_w14", "uplift": 0.09},
        {"variant": "c089_shape_w14", "uplift": 0.07},
        {"variant": "blend_species_face", "uplift": 0.08},
        {"variant": "blend_species_face", "uplift": 0.07},
    ]

    summary = summarize_shape_rows(rows)

    assert summary["best_c089_variant"] == "c089_shape_w14"
    assert summary["decision"] == "c089_shape_siglip_candidate_for_larger_gate"
