from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Final

import typer

from tools.siglip_auto_caption_types import JsonObject, JsonValue


EXPECTED_INIT: Final = "anima_siglip_ip_adapter_c094_shape_supervised_0064_20260613.safetensors"


def write_c095_training_report(stdout_path: Path, output_dir: Path) -> JsonObject:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_training_json(stdout_path)
    summary.setdefault("heldout07_rows", [])
    failures = _training_failures(summary)
    decision = (
        "proceed_to_c095_generation_gate"
        if not failures
        else "c095_training_gate_failed"
    )
    report: JsonObject = {
        "train_stdout": str(stdout_path),
        "training_summary": summary,
        "failures": failures,
        "decision": decision,
    }
    _write_json(output_dir / "training_summary.json", summary)
    _write_json(output_dir / "summary.json", report)
    _write_report(output_dir / "report.md", report)
    return report


def _training_failures(summary: JsonObject) -> list[str]:
    failures: list[str] = []
    if int(summary.get("rows_loaded", 0)) != 10:
        failures.append("rows loaded not 10")
    if int(summary.get("explicit_negative_rows", 0)) != 10:
        failures.append("explicit negative rows not 10")
    if summary.get("heldout07_rows") != []:
        failures.append("heldout07 leaked into training")
    if summary.get("finite_loss") is not True:
        failures.append("loss not finite")
    if int(summary.get("feature_bridge_bottleneck_dim", 0)) != 128:
        failures.append("feature bridge bottleneck not 128")
    if summary.get("train_feature_bridge_only") is not True:
        failures.append("feature bridge only disabled")
    if not _only_bridge_trainable(summary):
        failures.append("non-bridge parameters trainable")
    if Path(str(summary.get("init_checkpoint_path", ""))).name != EXPECTED_INIT:
        failures.append("wrong init checkpoint")
    checkpoint = summary.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("loadable") is not True:
        failures.append("checkpoint not loadable")
    if not isinstance(checkpoint, dict) or checkpoint.get("pe_checkpoint_rejected") is not True:
        failures.append("PE checkpoint not rejected")
    return failures


def _only_bridge_trainable(summary: JsonObject) -> bool:
    raw_names = summary.get("trainable_parameter_names")
    if not isinstance(raw_names, list) or not raw_names:
        return False
    return all(str(name).startswith("feature_bridge.") for name in raw_names)


def _read_training_json(path: Path) -> JsonObject:
    text = path.read_text(encoding="utf-8")
    marker = '{\n  "steps"'
    start = text.find(marker)
    if start < 0:
        start = text.find('{"steps"')
    end = text.rfind("}")
    if start >= 0 and end >= start:
        raw: JsonValue = json.loads(text[start : end + 1])
        if isinstance(raw, dict):
            return raw
    raise TypeError(f"no JSON training summary found in {path}")


def _write_report(path: Path, report: JsonObject) -> None:
    summary = report["training_summary"]
    lines = [
        "# c095 SigLIP Feature-Bridge Training Gate",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Failures: `{json.dumps(report['failures'], ensure_ascii=False)}`",
        f"- Steps: `{summary['steps']}`",
        f"- Rows loaded: `{summary['rows_loaded']}`",
        f"- Explicit negative rows: `{summary['explicit_negative_rows']}`",
        f"- Final loss: `{summary['final_loss']}`",
        f"- Mean shape loss: `{summary['mean_shape_loss']}`",
        f"- Feature bridge bottleneck: `{summary['feature_bridge_bottleneck_dim']}`",
        f"- Bridge-only: `{summary['train_feature_bridge_only']}`",
        f"- Trainable parameters: `{summary['trainable_parameters']}`",
        f"- Trainable names: `{json.dumps(summary['trainable_parameter_names'])}`",
        f"- Init checkpoint: `{summary['init_checkpoint_path']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: JsonObject) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    stdout_path: Annotated[Path, typer.Argument()],
    output_dir: Annotated[Path, typer.Option()] = Path(
        "eval/c095_siglip_feature_bridge_training_20260613"
    ),
) -> None:
    report = write_c095_training_report(stdout_path, output_dir)
    typer.echo(report["decision"])


if __name__ == "__main__":
    app()
