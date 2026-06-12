from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

import typer

from tools.image_feature_embedders import EncoderName, ImageEmbedder, build_image_embedder
from tools.siglip_auto_caption_types import JsonObject

if TYPE_CHECKING:
    import torch


VARIANTS: Final = ("no_ip", "blend_species_face", "c063_calibrator_only_w14")
SUPPORT_UPLIFT: Final = 0.05
SUPPORT_MARGIN: Final = 0.01


@dataclass(frozen=True, slots=True)
class ProbeCase:
    sample: str
    split: str
    failure_attribute: str
    reference_path: Path
    candidates: dict[str, Path]


@dataclass(frozen=True, slots=True)
class VariantScore:
    sample: str
    failure_attribute: str
    variant: str
    cosine: float
    no_ip_cosine: float
    uplift: float
    rank: int


@dataclass(frozen=True, slots=True)
class CaseDecision:
    sample: str
    failure_attribute: str
    best_variant: str
    best_uplift: float
    top_margin: float
    c063_vs_blend_delta: float
    decision: str


@dataclass(frozen=True, slots=True)
class ProbeInputError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def score_failure_attribute_probe(
    manifest_path: Path,
    *,
    embedder: ImageEmbedder,
    encoder_name: str,
) -> JsonObject:
    cases = _read_cases(manifest_path)
    cache: dict[Path, torch.Tensor] = {}
    rows: list[VariantScore] = []
    decisions: list[CaseDecision] = []
    for case in cases:
        ref = _embedding_cached(embedder, cache, case.reference_path)
        scored = [
            (
                variant,
                _cosine(ref, _embedding_cached(embedder, cache, case.candidates[variant])),
            )
            for variant in VARIANTS
        ]
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        no_ip_cosine = dict(scored)["no_ip"]
        rank_by_variant = {variant: index for index, (variant, _) in enumerate(ranked, start=1)}
        rows.extend(
            VariantScore(
                sample=case.sample,
                failure_attribute=case.failure_attribute,
                variant=variant,
                cosine=cosine,
                no_ip_cosine=no_ip_cosine,
                uplift=cosine - no_ip_cosine,
                rank=rank_by_variant[variant],
            )
            for variant, cosine in scored
        )
        decisions.append(_case_decision(case, ranked=ranked, no_ip_cosine=no_ip_cosine))
    return {
        "manifest_path": str(manifest_path),
        "encoder": encoder_name,
        "rows": [asdict(row) for row in sorted(rows, key=lambda row: (row.sample, row.rank))],
        "case_decisions": [asdict(decision) for decision in decisions],
        "summary": _summary(decisions),
    }


def write_probe_report(result: JsonObject, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_report(result), encoding="utf-8")


def _read_cases(manifest_path: Path) -> tuple[ProbeCase, ...]:
    cases: list[ProbeCase] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ProbeInputError(f"manifest row {line_number} must be an object")
            candidates = raw.get("candidates")
            if not isinstance(candidates, dict):
                raise ProbeInputError(f"manifest row {line_number} candidates must be an object")
            cases.append(
                ProbeCase(
                    sample=str(raw["sample"]),
                    split=str(raw["split"]),
                    failure_attribute=str(raw["failure_attribute"]),
                    reference_path=Path(str(raw["reference_path"])),
                    candidates={variant: Path(str(candidates[variant])) for variant in VARIANTS},
                )
            )
    if not cases:
        raise ProbeInputError(f"manifest has no rows: {manifest_path}")
    return tuple(cases)


def _embedding_cached(
    embedder: ImageEmbedder,
    cache: dict[Path, torch.Tensor],
    image_path: Path,
) -> torch.Tensor:
    if image_path not in cache:
        if not image_path.is_file():
            raise ProbeInputError(f"image path does not exist: {image_path}")
        cache[image_path] = embedder.encode_image(image_path)
    return cache[image_path]


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(left, right, dim=0).item())


def _case_decision(
    case: ProbeCase,
    *,
    ranked: list[tuple[str, float]],
    no_ip_cosine: float,
) -> CaseDecision:
    best_variant, best_cosine = ranked[0]
    second_cosine = ranked[1][1] if len(ranked) > 1 else no_ip_cosine
    score_by_variant = dict(ranked)
    best_uplift = best_cosine - no_ip_cosine
    top_margin = best_cosine - second_cosine
    decision = (
        "encoder_space_supports_supervised_signal"
        if best_uplift >= SUPPORT_UPLIFT and top_margin >= SUPPORT_MARGIN
        else "encoder_space_not_enough"
    )
    return CaseDecision(
        sample=case.sample,
        failure_attribute=case.failure_attribute,
        best_variant=best_variant,
        best_uplift=best_uplift,
        top_margin=top_margin,
        c063_vs_blend_delta=(
            score_by_variant["c063_calibrator_only_w14"] - score_by_variant["blend_species_face"]
        ),
        decision=decision,
    )


def _summary(decisions: list[CaseDecision]) -> JsonObject:
    supported = [
        decision
        for decision in decisions
        if decision.decision == "encoder_space_supports_supervised_signal"
    ]
    return {
        "cases": len(decisions),
        "supported_cases": len(supported),
        "support_rate": len(supported) / len(decisions),
        "decision": (
            "encoder_space_has_partial_supervised_signal"
            if supported
            else "encoder_side_checkpoint_required"
        ),
    }


def _render_report(result: JsonObject) -> str:
    summary = result["summary"]
    decisions = result["case_decisions"]
    if not isinstance(summary, dict) or not isinstance(decisions, list):
        raise ProbeInputError("probe result has invalid report shape")
    lines = [
        "# c064 Failure-Attribute Embedding Probe",
        "",
        f"- Encoder: `{result['encoder']}`",
        f"- Manifest: `{result['manifest_path']}`",
        f"- Cases: `{summary['cases']}`",
        f"- Supported cases: `{summary['supported_cases']}`",
        f"- Support rate: `{summary['support_rate']}`",
        f"- Decision: `{summary['decision']}`",
        "",
        "## Case Decisions",
        "",
    ]
    for item in decisions:
        if not isinstance(item, dict):
            raise ProbeInputError("case decision must be an object")
        lines.extend(
            [
                f"- `{item['sample']}` `{item['failure_attribute']}`: "
                f"`{item['decision']}`; best=`{item['best_variant']}`, "
                f"uplift=`{item['best_uplift']}`, top_margin=`{item['top_margin']}`, "
                f"c063_vs_blend_delta=`{item['c063_vs_blend_delta']}`",
            ]
        )
    lines.extend(
        [
            "",
            "This probe measures feature-space separation only. It does not by itself prove generation quality.",
            "",
        ]
    )
    return "\n".join(lines)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest_path: Annotated[Path, typer.Argument()],
    output_path: Annotated[Path, typer.Argument()],
    encoder: Annotated[EncoderName, typer.Option()] = "qwenvl",
    model_id: Annotated[str | None, typer.Option()] = None,
    device: Annotated[str, typer.Option()] = "auto",
    report_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    encoder_name, embedder = build_image_embedder(encoder, model_id=model_id, device=device)
    result = score_failure_attribute_probe(
        manifest_path,
        embedder=embedder,
        encoder_name=encoder_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if report_path is not None:
        write_probe_report(result, report_path)
    typer.echo(f"wrote {output_path}")


if __name__ == "__main__":
    app()
