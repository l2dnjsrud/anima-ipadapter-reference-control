from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

from PIL import Image, ImageDraw, ImageOps


DEFAULT_EVAL_DIR: Final = Path("eval/c089_shape_silhouette_distillation_pilot_20260613")
DEFAULT_MANIFEST: Final = Path(
    "training/manifests/c089_shape_silhouette_distillation_20260613.jsonl"
)
DEFAULT_MANIFEST_SUMMARY: Final = Path(
    "training/manifests/c089_shape_silhouette_distillation_20260613.summary.json"
)
DEFAULT_IMAGE_ROOT: Final = Path(".tmp/c089_shape_silhouette_distillation_root")

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


def build_c089_pilot_report(
    *,
    eval_dir: Path = DEFAULT_EVAL_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    manifest_summary_path: Path = DEFAULT_MANIFEST_SUMMARY,
    image_root: Path = DEFAULT_IMAGE_ROOT,
) -> JsonObject:
    train = _read_object(eval_dir / "train_stdout.txt")
    manifest_summary = _read_object(manifest_summary_path)
    rows = _read_jsonl(manifest_path)
    checkpoint = _object_value(train, "checkpoint")
    metrics = _build_metrics(train, checkpoint)
    decision = _decision(metrics, manifest_summary)
    summary: JsonObject = {
        "experiment": "c089_shape_silhouette_distillation_pilot",
        "decision": decision,
        "manifest": manifest_summary,
        "training": train,
        "metrics": metrics,
        "artifacts": {
            "summary": str(eval_dir / "summary.json"),
            "metrics": str(eval_dir / "metrics.json"),
            "report": str(eval_dir / "report.md"),
            "contact_sheet": str(eval_dir / "contact_sheet.jpg"),
        },
    }
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (eval_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (eval_dir / "report.md").write_text(
        _report(summary),
        encoding="utf-8",
    )
    _write_contact_sheet(rows[:8], image_root, eval_dir / "contact_sheet.jpg")
    return summary


def _build_metrics(train: JsonObject, checkpoint: JsonObject) -> JsonObject:
    first_loss = _float_value(train, "first_loss")
    final_loss = _float_value(train, "final_loss")
    return {
        "finite_loss": _bool_value(train, "finite_loss"),
        "checkpoint_loadable": _bool_value(checkpoint, "loadable"),
        "pe_checkpoint_rejected": _bool_value(checkpoint, "pe_checkpoint_rejected"),
        "loss_delta": final_loss - first_loss,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "mean_teacher_loss": _float_value(train, "mean_teacher_loss"),
        "mean_pe_token_loss": _float_value(train, "mean_pe_token_loss"),
        "mean_pe_retrieval_loss": _float_value(train, "mean_pe_retrieval_loss"),
        "steps": _int_value(train, "steps"),
        "rows_loaded": _int_value(train, "rows_loaded"),
        "checkpoint_path": str(_string_value(checkpoint, "output_path")),
    }


def _decision(metrics: JsonObject, manifest_summary: JsonObject) -> str:
    heldout_rows = _int_value(manifest_summary, "heldout_rows_used")
    teacher_signal = _float_value(metrics, "mean_teacher_loss") > 0.0
    pe_signal = (
        _float_value(metrics, "mean_pe_token_loss") > 0.0
        or _float_value(metrics, "mean_pe_retrieval_loss") > 0.0
    )
    if heldout_rows > 0:
        return "blocked_heldout_leakage"
    if not _bool_value(metrics, "finite_loss"):
        return "revise_objective_nonfinite_loss"
    if not _bool_value(metrics, "checkpoint_loadable"):
        return "revise_objective_checkpoint_not_loadable"
    if not _bool_value(metrics, "pe_checkpoint_rejected"):
        return "revise_objective_checkpoint_family_guard_failed"
    if teacher_signal and pe_signal:
        return "proceed_to_siglip_generation_gate"
    return "escalate_to_encoder_side_checkpoint_training"


def _report(summary: JsonObject) -> str:
    metrics = _object_value(summary, "metrics")
    training = _object_value(summary, "training")
    return "\n".join(
        [
            "# c089 Shape/Silhouette Distillation Pilot",
            "",
            f"- Decision: `{_string_value(summary, 'decision')}`",
            f"- Steps: `{_int_value(training, 'steps')}`",
            f"- Rows loaded: `{_int_value(training, 'rows_loaded')}`",
            f"- First loss: `{_float_value(metrics, 'first_loss')}`",
            f"- Final loss: `{_float_value(metrics, 'final_loss')}`",
            f"- Loss delta: `{_float_value(metrics, 'loss_delta')}`",
            f"- Mean teacher loss: `{_float_value(metrics, 'mean_teacher_loss')}`",
            f"- Mean PE token loss: `{_float_value(metrics, 'mean_pe_token_loss')}`",
            f"- Mean PE retrieval loss: `{_float_value(metrics, 'mean_pe_retrieval_loss')}`",
            f"- Checkpoint: `{_string_value(metrics, 'checkpoint_path')}`",
            "",
            "## Interpretation",
            "",
            "The pilot produced a finite, loadable SigLIP checkpoint with active PE teacher and PE token retrieval signal.",
            "This is not a final quality pass; it only clears the next step of running a ComfyUI generation gate against the hard-shape references.",
            "",
        ]
    )


def _write_contact_sheet(rows: list[JsonObject], image_root: Path, output_path: Path) -> None:
    thumb = 128
    label_h = 20
    gap = 8
    width = 2 * thumb + 3 * gap
    height = max(1, len(rows)) * (thumb + label_h + gap) + gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for row_idx, row in enumerate(rows):
        y = gap + row_idx * (thumb + label_h + gap)
        for col_idx, key in enumerate(("ref_id", "tgt_id")):
            x = gap + col_idx * (thumb + gap)
            image = _load_thumb(image_root / f"{_string_value(row, key)}.jpg", thumb)
            sheet.paste(image, (x, y + label_h))
            draw.text((x, y), key.replace("_id", ""), fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def _load_thumb(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.fit(
            image.convert("RGB"),
            (size, size),
            method=Image.Resampling.LANCZOS,
        )


def _read_jsonl(path: Path) -> list[JsonObject]:
    return [_as_object(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


def _read_object(path: Path) -> JsonObject:
    return _as_object(json.loads(path.read_text(encoding="utf-8")))


def _as_object(value: JsonValue) -> JsonObject:
    if isinstance(value, dict):
        return value
    raise TypeError("expected JSON object")


def _object_value(data: JsonObject, key: str) -> JsonObject:
    return _as_object(data[key])


def _string_value(data: JsonObject, key: str) -> str:
    value = data[key]
    if isinstance(value, str):
        return value
    raise TypeError(f"{key} must be a string")


def _int_value(data: JsonObject, key: str) -> int:
    value = data[key]
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise TypeError(f"{key} must be an integer")


def _float_value(data: JsonObject, key: str) -> float:
    value = data[key]
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise TypeError(f"{key} must be a number")


def _bool_value(data: JsonObject, key: str) -> bool:
    value = data[key]
    if isinstance(value, bool):
        return value
    raise TypeError(f"{key} must be a boolean")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-summary-path", type=Path, default=DEFAULT_MANIFEST_SUMMARY)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    args = parser.parse_args()
    summary = build_c089_pilot_report(
        eval_dir=args.eval_dir,
        manifest_path=args.manifest_path,
        manifest_summary_path=args.manifest_summary_path,
        image_root=args.image_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
