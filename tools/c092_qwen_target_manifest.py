from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Final

import typer
from PIL import Image


DEFAULT_SUMMARY: Final = Path(
    "eval/c091_siglip_feature_calibrator_hard_shape_gate_20260613/summary.json"
)
DEFAULT_OUT_MANIFEST: Final = Path(
    "training/manifests/c092_qwen_target_distillation_20260613.jsonl"
)
DEFAULT_IMAGE_ROOT: Final = Path(".tmp/c092_qwen_target_distillation_root")
DEFAULT_TEACHER_VARIANT: Final = "c087_expanded_crop_positive_w14"


@dataclass(frozen=True, slots=True)
class MaterializedRow:
    label: str
    ref_id: str
    tgt_id: str
    prompt: str
    reference_source: str
    target_source: str


def materialize_c092_qwen_target_manifest(
    summary_path: Path,
    *,
    out_manifest: Path,
    image_root: Path,
    teacher_variant: str = DEFAULT_TEACHER_VARIANT,
    exclude_heldout: bool = True,
) -> dict[str, object]:
    summary = _read_json(summary_path)
    samples = summary.get("samples")
    candidates = summary.get("baseline_candidates")
    if not isinstance(samples, list):
        raise TypeError("summary.samples must be a list")
    if not isinstance(candidates, dict):
        raise TypeError("summary.baseline_candidates must be an object")

    rows: list[MaterializedRow] = []
    excluded: list[str] = []
    for raw_sample in samples:
        if not isinstance(raw_sample, dict):
            continue
        label = str(raw_sample["label"])
        if exclude_heldout and label.startswith("heldout"):
            excluded.append(label)
            continue
        prompt_row = raw_sample.get("prompt_row")
        if not isinstance(prompt_row, dict):
            raise TypeError(f"sample {label} prompt_row must be an object")
        source_prompt = str(prompt_row["prompt"])
        source_ref = Path(str(raw_sample["reference_path"]))
        raw_candidates = candidates.get(label)
        if not isinstance(raw_candidates, dict) or teacher_variant not in raw_candidates:
            raise KeyError(f"missing teacher variant {teacher_variant!r} for {label}")
        source_target = Path(str(raw_candidates[teacher_variant]))
        if not source_ref.is_file():
            raise FileNotFoundError(source_ref)
        if not source_target.is_file():
            raise FileNotFoundError(source_target)

        ref_id = f"c092_qwen_target/{label}_ref"
        tgt_id = f"c092_qwen_target/{label}_{teacher_variant}"
        _copy_rgb_jpg(source_ref, image_root / f"{ref_id}.jpg")
        _copy_rgb_jpg(source_target, image_root / f"{tgt_id}.jpg")
        (image_root / f"{tgt_id}.txt").write_text(source_prompt + "\n", encoding="utf-8")
        rows.append(
            MaterializedRow(
                label=label,
                ref_id=ref_id,
                tgt_id=tgt_id,
                prompt=source_prompt,
                reference_source=str(source_ref),
                target_source=str(source_target),
            )
        )

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "ref_id": row.ref_id,
                        "tgt_id": row.tgt_id,
                        "prompt": row.prompt,
                        "source_label": row.label,
                        "teacher_variant": teacher_variant,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary_payload: dict[str, object] = {
        "decision": "ready_for_c092_qwen_target_siglip_distillation",
        "source_summary_path": str(summary_path),
        "output_manifest_path": str(out_manifest),
        "scratch_image_root": str(image_root),
        "teacher_variant": teacher_variant,
        "total_rows": len(rows),
        "excluded_labels": excluded,
        "rows": [asdict(row) for row in rows],
    }
    _write_json(out_manifest.with_suffix(out_manifest.suffix + ".summary.json"), summary_payload)
    return summary_payload


def _copy_rgb_jpg(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, quality=95)


def _read_json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"json root must be object: {path}")
    return raw


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    summary_path: Annotated[Path, typer.Option()] = DEFAULT_SUMMARY,
    out_manifest: Annotated[Path, typer.Option()] = DEFAULT_OUT_MANIFEST,
    image_root: Annotated[Path, typer.Option()] = DEFAULT_IMAGE_ROOT,
    teacher_variant: Annotated[str, typer.Option()] = DEFAULT_TEACHER_VARIANT,
    exclude_heldout: Annotated[bool, typer.Option()] = True,
) -> None:
    payload = materialize_c092_qwen_target_manifest(
        summary_path,
        out_manifest=out_manifest,
        image_root=image_root,
        teacher_variant=teacher_variant,
        exclude_heldout=exclude_heldout,
    )
    typer.echo(
        f"wrote {payload['total_rows']} rows to {payload['output_manifest_path']}"
    )


if __name__ == "__main__":
    app()
