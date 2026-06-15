from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Final

import typer
from PIL import Image


DEFAULT_C092_SUMMARY: Final = Path(
    "training/manifests/c092_qwen_target_distillation_20260613.jsonl.summary.json"
)
DEFAULT_C092_GATE: Final = Path("eval/c092_qwen_target_siglip_generation_gate_20260613")
DEFAULT_OUTPUT_ROOT: Final = Path(".tmp/c093_anti_collapse_root")
DEFAULT_OUTPUT_MANIFEST: Final = Path(
    "training/manifests/c093_siglip_qwen_target_anti_collapse_20260613.jsonl"
)
DEFAULT_OUTPUT_SUMMARY: Final = Path(
    "training/manifests/c093_siglip_qwen_target_anti_collapse_20260613.summary.json"
)
DEFAULT_POSITIVE_VARIANT: Final = "c087_expanded_crop_positive_w14"
DEFAULT_NEGATIVE_VARIANT: Final = "c092_qwen_target_w14"
OUTPUT_PREFIX: Final = "c093_anti_collapse"


@dataclass(frozen=True, slots=True)
class C093ManifestSummary:
    output_manifest_path: str
    scratch_image_root: str
    source_summary_path: str
    c092_gate_dir: str
    total_rows: int
    explicit_negative_rows: int
    heldout_rows_used: int
    excluded_labels: list[str]
    positive_teacher_variant: str
    negative_variant: str
    decision: str


def build_c093_anti_collapse_manifest(
    *,
    c092_manifest_summary: Path = DEFAULT_C092_SUMMARY,
    c092_gate_dir: Path = DEFAULT_C092_GATE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    output_manifest: Path = DEFAULT_OUTPUT_MANIFEST,
    output_summary: Path = DEFAULT_OUTPUT_SUMMARY,
    positive_teacher_variant: str = DEFAULT_POSITIVE_VARIANT,
    negative_variant: str = DEFAULT_NEGATIVE_VARIANT,
) -> C093ManifestSummary:
    source_rows, excluded_labels = _source_rows(c092_manifest_summary)
    manifest_rows: list[dict[str, str]] = []
    for row in source_rows:
        label = str(row["label"])
        if label.startswith("heldout"):
            excluded_labels.append(label)
            continue
        output_ids = _output_ids(label, negative_variant)
        negative_source = c092_gate_dir / f"{label}_{negative_variant}.png"
        _copy_rgb_jpg(Path(str(row["reference_source"])), output_root / f"{output_ids['ref_id']}.jpg")
        _copy_rgb_jpg(Path(str(row["target_source"])), output_root / f"{output_ids['tgt_id']}.jpg")
        _copy_rgb_jpg(negative_source, output_root / f"{output_ids['neg_id']}.jpg")
        (output_root / f"{output_ids['tgt_id']}.txt").write_text(
            str(row["prompt"]) + "\n",
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                **output_ids,
                "prompt": str(row["prompt"]),
                "source_label": label,
                "positive_teacher_variant": positive_teacher_variant,
                "negative_variant": negative_variant,
            }
        )
    if not manifest_rows:
        raise RuntimeError("c093 manifest has no train rows")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = C093ManifestSummary(
        output_manifest_path=str(output_manifest),
        scratch_image_root=str(output_root),
        source_summary_path=str(c092_manifest_summary),
        c092_gate_dir=str(c092_gate_dir),
        total_rows=len(manifest_rows),
        explicit_negative_rows=sum(1 for row in manifest_rows if row.get("neg_id")),
        heldout_rows_used=0,
        excluded_labels=sorted(set(excluded_labels)),
        positive_teacher_variant=positive_teacher_variant,
        negative_variant=negative_variant,
        decision="ready_for_c093_anti_collapse_training",
    )
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _source_rows(path: Path) -> tuple[tuple[dict[str, object], ...], list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("rows"), list):
        raise TypeError(f"invalid c092 summary rows: {path}")
    excluded = raw.get("excluded_labels", [])
    excluded_labels = [
        str(label) for label in excluded if isinstance(label, str) and label
    ]
    return tuple(row for row in raw["rows"] if isinstance(row, dict)), excluded_labels


def _output_ids(label: str, negative_variant: str) -> dict[str, str]:
    return {
        "ref_id": f"{OUTPUT_PREFIX}/{label}_ref",
        "tgt_id": f"{OUTPUT_PREFIX}/{label}_target",
        "neg_id": f"{OUTPUT_PREFIX}/{label}_{negative_variant}_negative",
    }


def _copy_rgb_jpg(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, quality=95)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    c092_manifest_summary: Annotated[Path, typer.Option()] = DEFAULT_C092_SUMMARY,
    c092_gate_dir: Annotated[Path, typer.Option()] = DEFAULT_C092_GATE,
    output_root: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_ROOT,
    output_manifest: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_MANIFEST,
    output_summary: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_SUMMARY,
    positive_teacher_variant: Annotated[str, typer.Option()] = DEFAULT_POSITIVE_VARIANT,
    negative_variant: Annotated[str, typer.Option()] = DEFAULT_NEGATIVE_VARIANT,
) -> None:
    summary = build_c093_anti_collapse_manifest(
        c092_manifest_summary=c092_manifest_summary,
        c092_gate_dir=c092_gate_dir,
        output_root=output_root,
        output_manifest=output_manifest,
        output_summary=output_summary,
        positive_teacher_variant=positive_teacher_variant,
        negative_variant=negative_variant,
    )
    typer.echo(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
