from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tools import build_reference_control_audit_manifest
from tools.build_reference_control_audit_manifest import (
    NextRoute,
    build_audit_rows,
    write_audit_outputs,
)


RUNNER = CliRunner()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_eval_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    summary_path = tmp_path / "summary.json"
    visual_path = tmp_path / "visual_audit.json"
    metrics_path = tmp_path / "pe_similarity_metrics.json"
    manifest_path = tmp_path / "suite.jsonl"
    manifest_path.write_text(
        json.dumps({"ref_id": "SG-001/ref", "tgt_id": "SG-001/ref", "prompt": "base prompt"})
        + "\n"
        + json.dumps({"ref_id": "SG-002/ref", "tgt_id": "SG-002/ref", "prompt": "base prompt"})
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        summary_path,
        {
            "samples": [
                {
                    "label": "auto00",
                    "ref_id": "SG-001/ref",
                    "prompt_row": {
                        "prompt": "prompt with green monster",
                        "selected_attributes": ["green non-human face"],
                    },
                },
                {
                    "label": "auto01",
                    "ref_id": "SG-002/ref",
                    "prompt_row": {
                        "prompt": "prompt with blue robe",
                        "selected_attributes": ["blue robe"],
                    },
                },
            ],
            "results": {
                "auto00_no_ip": {"image": "eval/run/auto00_no_ip.png"},
                "auto00_siglip_ref_retrieval_w14": {"image": "eval/run/auto00_ip.png"},
                "auto01_no_ip": {"image": "eval/run/auto01_no_ip.png"},
                "auto01_siglip_ref_retrieval_w14": {"image": "eval/run/auto01_ip.png"},
            },
        },
    )
    _write_json(
        visual_path,
        {
            "decision": "not_ready",
            "best_siglip_variant": "siglip_ref_retrieval_w14",
            "rows": [
                {
                    "sample": "auto00",
                    "best_variant": "siglip_ref_retrieval_w14",
                    "palette_costume_expression_framing_acceptable": True,
                    "identity_distinctive_trait_acceptable": False,
                    "notes": "Green non-human face becomes a human template.",
                },
                {
                    "sample": "auto01",
                    "best_variant": "siglip_ref_retrieval_w14",
                    "palette_costume_expression_framing_acceptable": True,
                    "identity_distinctive_trait_acceptable": True,
                    "notes": "Style is acceptable but metric goes down.",
                },
            ],
        },
    )
    _write_json(
        metrics_path,
        {
            "rows": [
                {
                    "sample": "auto00",
                    "variant": "siglip_ref_retrieval_w14",
                    "uplift": 0.05,
                    "pixel_std": 70.0,
                },
                {
                    "sample": "auto01",
                    "variant": "siglip_ref_retrieval_w14",
                    "uplift": -0.02,
                    "pixel_std": 80.0,
                },
            ]
        },
    )
    return summary_path, visual_path, metrics_path, manifest_path


def test_build_audit_rows_routes_identity_failures_to_stronger_encoder(tmp_path: Path) -> None:
    summary_path, visual_path, metrics_path, manifest_path = _write_eval_inputs(tmp_path)

    rows = build_audit_rows(
        summary_path=summary_path,
        visual_audit_path=visual_path,
        metrics_path=metrics_path,
        suite_manifest_path=manifest_path,
    )

    assert rows[0].case_id == "auto00"
    assert rows[0].failure_tags == (
        "identity_distinctive_trait",
        "non_human_or_special_trait",
        "template_collapse",
    )
    assert rows[0].next_route is NextRoute.STRONGER_ENCODER
    assert rows[0].ip_output == "eval/run/auto00_ip.png"


def test_build_audit_rows_keeps_metric_only_failures_as_prompt_patch(tmp_path: Path) -> None:
    summary_path, visual_path, metrics_path, manifest_path = _write_eval_inputs(tmp_path)

    rows = build_audit_rows(
        summary_path=summary_path,
        visual_audit_path=visual_path,
        metrics_path=metrics_path,
        suite_manifest_path=manifest_path,
    )

    assert rows[1].failure_tags == ("metric_not_improved",)
    assert rows[1].next_route is NextRoute.PROMPT_PATCH


def test_cli_writes_manifest_and_summary(tmp_path: Path) -> None:
    summary_path, visual_path, metrics_path, manifest_path = _write_eval_inputs(tmp_path)
    output_path = tmp_path / "audit.jsonl"
    summary_output_path = tmp_path / "audit.md"

    result = RUNNER.invoke(
        build_reference_control_audit_manifest.app,
        [
            str(summary_path),
            str(visual_path),
            str(metrics_path),
            str(manifest_path),
            str(output_path),
            "--summary-output-path",
            str(summary_output_path),
        ],
    )

    assert result.exit_code == 0
    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert written[0]["next_route"] == "stronger_encoder"
    assert "stronger_encoder: 1" in summary_output_path.read_text(encoding="utf-8")
