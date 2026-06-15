from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from tools.c088_shape_silhouette_probe import (
    C088BuildConfig,
    build_c088_probe_manifest,
    score_shape_silhouette_manifest,
)
from tools.c088_embedding_metrics import score_embedding_manifest
from tools.c088_report import build_c088_rollup


def _save_shape(path: Path, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)
    match kind:
        case "tall":
            draw.rectangle((36, 12, 60, 84), fill="black")
        case "wide":
            draw.rectangle((12, 36, 84, 60), fill="black")
        case "circle":
            draw.ellipse((20, 20, 76, 76), fill="black")
        case unexpected:
            raise AssertionError(f"unknown shape kind: {unexpected}")
    image.save(path)


def _summary(path: Path, data_root: Path, labels: tuple[str, ...]) -> None:
    variants = [
        {"label": "no_ip"},
        {"label": "blend_species_face"},
        {"label": "c085_anchored_full_adapter_w14"},
        {"label": "c086_hard_negative_w14"},
        {"label": "c087_expanded_crop_positive_w14"},
    ]
    results: dict[str, dict[str, str]] = {}
    for label in labels:
        for variant in variants:
            variant_label = str(variant["label"])
            results[f"{label}_{variant_label}"] = {
                "image": str(path.parent / f"{label}_{variant_label}.png")
            }
    path.write_text(
        json.dumps(
            {
                "data_root": str(data_root),
                "variants": variants,
                "samples": [
                    {
                        "label": label,
                        "split": "direct_green",
                        "ref_id": f"refs/{label}",
                        "selected_attributes": ["non_human"],
                    }
                    for label in labels
                ],
                "results": results,
            }
        ),
        encoding="utf-8",
    )


def test_build_c088_probe_manifest_verifies_paths_and_groups(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_dir = tmp_path / "out"
    label = "crop_pair00"
    _save_shape(data_root / "refs" / f"{label}.jpg", "tall")
    for variant in (
        "no_ip",
        "blend_species_face",
        "c085_anchored_full_adapter_w14",
        "c086_hard_negative_w14",
        "c087_expanded_crop_positive_w14",
    ):
        _save_shape(tmp_path / f"{label}_{variant}.png", "wide")
    crop_summary = tmp_path / "crop_summary.json"
    full_summary = tmp_path / "full_summary.json"
    _summary(crop_summary, data_root, (label,))
    _summary(full_summary, data_root, ())

    summary = build_c088_probe_manifest(
        C088BuildConfig(
            crop_summary_path=crop_summary,
            full_summary_path=full_summary,
            output_dir=output_dir,
            crop_limit=1,
            heldout_labels=(),
        )
    )

    assert summary["rows"] == 1
    assert summary["heldout_training_rows_used"] == 0
    row = json.loads((output_dir / "probe_manifest.jsonl").read_text(encoding="utf-8"))
    assert row["shape_group"] == "frog_yokai_guard"
    assert set(row["candidates"]) == {
        "no_ip",
        "blend_species_face",
        "c085_anchored_full_adapter_w14",
        "c086_hard_negative_w14",
        "c087_expanded_crop_positive_w14",
    }


def test_score_shape_silhouette_manifest_ranks_matching_shape_first(tmp_path: Path) -> None:
    ref = tmp_path / "ref.jpg"
    no_ip = tmp_path / "no_ip.png"
    blend = tmp_path / "blend.png"
    c085 = tmp_path / "c085.png"
    c086 = tmp_path / "c086.png"
    c087 = tmp_path / "c087.png"
    _save_shape(ref, "tall")
    _save_shape(no_ip, "wide")
    _save_shape(blend, "circle")
    _save_shape(c085, "tall")
    _save_shape(c086, "circle")
    _save_shape(c087, "wide")
    manifest_path = tmp_path / "probe_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "sample": "shape-test",
                "split": "direct_green",
                "shape_group": "frog_yokai_guard",
                "failure_mode": "adult_humanoid_collapse",
                "reference_path": str(ref),
                "candidates": {
                    "no_ip": str(no_ip),
                    "blend_species_face": str(blend),
                    "c085_anchored_full_adapter_w14": str(c085),
                    "c086_hard_negative_w14": str(c086),
                    "c087_expanded_crop_positive_w14": str(c087),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = score_shape_silhouette_manifest(manifest_path, tmp_path / "out")

    assert result["summary"]["cases"] == 1
    assert result["summary"]["supported_cases"] == 1
    decision = result["case_decisions"][0]
    assert decision["best_variant"] == "c085_anchored_full_adapter_w14"
    assert decision["decision"] == "shape_signal_supports_supervised_objective"
    assert (tmp_path / "out" / "contact_sheet.jpg").is_file()


class FakeEmbedder:
    def __init__(self, vectors: dict[str, tuple[float, float]]) -> None:
        self._vectors = vectors

    def encode_image(self, image_path: Path):
        import torch

        return torch.tensor(self._vectors[image_path.name])


def test_score_embedding_manifest_uses_dynamic_c088_variants(tmp_path: Path) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "ref.jpg",
            "no_ip.png",
            "blend.png",
            "c085.png",
            "c086.png",
            "c087.png",
        )
    }
    for path in paths.values():
        _save_shape(path, "circle")
    manifest_path = tmp_path / "probe_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "sample": "embed-test",
                "split": "direct_green",
                "shape_group": "frog_yokai_guard",
                "failure_mode": "adult_humanoid_collapse",
                "reference_path": str(paths["ref.jpg"]),
                "candidates": {
                    "no_ip": str(paths["no_ip.png"]),
                    "blend_species_face": str(paths["blend.png"]),
                    "c085_anchored_full_adapter_w14": str(paths["c085.png"]),
                    "c086_hard_negative_w14": str(paths["c086.png"]),
                    "c087_expanded_crop_positive_w14": str(paths["c087.png"]),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = score_embedding_manifest(
        manifest_path,
        embedder=FakeEmbedder(
            {
                "ref.jpg": (1.0, 0.0),
                "no_ip.png": (0.0, 1.0),
                "blend.png": (0.5, 0.5),
                "c085.png": (1.0, 0.0),
                "c086.png": (0.2, 0.8),
                "c087.png": (0.4, 0.6),
            }
        ),
        encoder_name="fake",
    )

    assert result["summary"]["supported_cases"] == 1
    assert result["case_decisions"][0]["best_variant"] == "c085_anchored_full_adapter_w14"
    assert {row["variant"] for row in result["rows"]} == {
        "no_ip",
        "blend_species_face",
        "c085_anchored_full_adapter_w14",
        "c086_hard_negative_w14",
        "c087_expanded_crop_positive_w14",
    }


def test_build_c088_rollup_flags_shape_signal_without_embedding_support(tmp_path: Path) -> None:
    def write_metric(filename: str, support_rate: float) -> None:
        (tmp_path / filename).write_text(
            json.dumps(
                {
                    "summary": {
                        "cases": 2,
                        "supported_cases": int(support_rate * 2),
                        "support_rate": support_rate,
                        "decision": "probe",
                    },
                    "case_decisions": [],
                }
            ),
            encoding="utf-8",
        )

    write_metric("shape_silhouette_metrics.json", 0.5)
    write_metric("qwenvl_embedding_metrics.json", 0.0)
    write_metric("siglip_embedding_metrics.json", 0.0)
    write_metric("pe_embedding_metrics.json", 0.0)

    rollup = build_c088_rollup(tmp_path)

    assert rollup["decision"] == "shape_signal_present_encoder_embedding_not_enough"
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "metric_rollup.json").is_file()
    assert (tmp_path / "report.md").is_file()
