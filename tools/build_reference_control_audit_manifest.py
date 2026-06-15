from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final

import typer


LOW_PIXEL_STD_THRESHOLD: Final = 5.0

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


class NextRoute(StrEnum):
    PROMPT_PATCH = "prompt_patch"
    PAIR_MINING = "pair_mining"
    STRONGER_ENCODER = "stronger_encoder"
    LINE_CONTROL = "line_control"
    HOLD = "hold"


class AuditInputError(Exception):
    def __init__(self, context: str) -> None:
        self.context = context
        super().__init__(f"invalid reference-control audit input: {self.context}")


class OutputAlreadyExistsError(Exception):
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        super().__init__(f"reference-control audit output already exists: {self.output_path}")


@dataclass(frozen=True, slots=True)
class AuditRow:
    case_id: str
    reference_id: str
    target_id: str
    reference_image: str
    source_prompt: str
    target_prompt: str
    selected_attributes: tuple[str, ...]
    no_ip_output: str
    best_variant: str
    ip_output: str
    metric_delta: float | None
    metric_improved: bool | None
    pixel_std: float | None
    palette_costume_expression_framing_acceptable: bool
    identity_distinctive_trait_acceptable: bool
    failure_tags: tuple[str, ...]
    next_route: NextRoute
    notes: str


def build_audit_rows(
    *,
    summary_path: Path,
    visual_audit_path: Path,
    metrics_path: Path,
    suite_manifest_path: Path,
) -> tuple[AuditRow, ...]:
    summary = _read_json_object(summary_path)
    visual = _read_json_object(visual_audit_path)
    metrics = _read_json_object(metrics_path)
    manifest_rows = _read_manifest_rows(suite_manifest_path)

    samples = {_string(row, "label"): row for row in _objects(summary, "samples")}
    visual_rows = _objects(visual, "rows")
    metric_rows = {
        (_string(row, "sample"), _string(row, "variant")): row
        for row in _objects(metrics, "rows")
    }
    results = _object(summary, "results")
    default_variant = _optional_string(visual, "best_siglip_variant")
    rows: list[AuditRow] = []
    for visual_row in visual_rows:
        case_id = _string(visual_row, "sample")
        sample = samples[case_id]
        best_variant = _optional_string(visual_row, "best_variant") or default_variant
        if best_variant is None:
            raise AuditInputError(f"missing best variant for {case_id}")
        prompt_row = _object(sample, "prompt_row")
        ref_id = _string(sample, "ref_id")
        suite_row = manifest_rows.get(ref_id, {})
        metric_row = metric_rows.get((case_id, best_variant))
        metric_delta = _optional_float(metric_row, "uplift") if metric_row is not None else None
        pixel_std = _optional_float(metric_row, "pixel_std") if metric_row is not None else None
        palette_ok = _bool(visual_row, "palette_costume_expression_framing_acceptable")
        identity_ok = _bool(visual_row, "identity_distinctive_trait_acceptable")
        notes = _string(visual_row, "notes")
        failure_tags = _failure_tags(
            palette_ok=palette_ok,
            identity_ok=identity_ok,
            metric_delta=metric_delta,
            pixel_std=pixel_std,
            notes=notes,
        )
        rows.append(
            AuditRow(
                case_id=case_id,
                reference_id=ref_id,
                target_id=str(suite_row.get("tgt_id", ref_id)),
                reference_image=f"{ref_id}.jpg",
                source_prompt=str(suite_row.get("prompt", "")),
                target_prompt=_string(prompt_row, "prompt"),
                selected_attributes=tuple(_strings(prompt_row, "selected_attributes")),
                no_ip_output=_result_image(results, case_id, "no_ip"),
                best_variant=best_variant,
                ip_output=_result_image(results, case_id, best_variant),
                metric_delta=metric_delta,
                metric_improved=None if metric_delta is None else metric_delta > 0.0,
                pixel_std=pixel_std,
                palette_costume_expression_framing_acceptable=palette_ok,
                identity_distinctive_trait_acceptable=identity_ok,
                failure_tags=failure_tags,
                next_route=_next_route(failure_tags),
                notes=notes,
            )
        )
    return tuple(rows)


def write_audit_outputs(
    rows: tuple[AuditRow, ...],
    output_path: Path,
    summary_output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    for path in (output_path, summary_output_path):
        if path.exists() and not overwrite:
            raise OutputAlreadyExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")
    summary_output_path.write_text(render_summary(rows), encoding="utf-8")


def render_summary(rows: tuple[AuditRow, ...]) -> str:
    route_counts = {route.value: 0 for route in NextRoute}
    for row in rows:
        route_counts[row.next_route.value] += 1
    lines = [
        "# Reference-Control Audit Summary",
        "",
        f"- rows: {len(rows)}",
        "- route counts:",
    ]
    lines.extend(f"  - {route}: {count}" for route, count in route_counts.items())
    return "\n".join(lines) + "\n"


def _failure_tags(
    *,
    palette_ok: bool,
    identity_ok: bool,
    metric_delta: float | None,
    pixel_std: float | None,
    notes: str,
) -> tuple[str, ...]:
    tags: list[str] = []
    lowered = notes.lower()
    if not palette_ok:
        tags.append("palette_costume_expression_framing")
    if not identity_ok:
        tags.append("identity_distinctive_trait")
    if "non-human" in lowered or "special-trait" in lowered:
        tags.append("non_human_or_special_trait")
    if "template" in lowered or "generic" in lowered:
        tags.append("template_collapse")
    if metric_delta is not None and metric_delta <= 0.0:
        tags.append("metric_not_improved")
    if pixel_std is not None and pixel_std < LOW_PIXEL_STD_THRESHOLD:
        tags.append("blank_or_low_variance")
    return tuple(tags)


def _next_route(failure_tags: tuple[str, ...]) -> NextRoute:
    if "identity_distinctive_trait" in failure_tags or "non_human_or_special_trait" in failure_tags:
        return NextRoute.STRONGER_ENCODER
    if "palette_costume_expression_framing" in failure_tags or "metric_not_improved" in failure_tags:
        return NextRoute.PROMPT_PATCH
    if "blank_or_low_variance" in failure_tags:
        return NextRoute.HOLD
    return NextRoute.HOLD


def _read_json_object(path: Path) -> JsonObject:
    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return raw
    raise AuditInputError(f"{path} must contain a JSON object")


def _read_manifest_rows(path: Path) -> dict[str, JsonObject]:
    rows: dict[str, JsonObject] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw: JsonValue = json.loads(line)
            if not isinstance(raw, dict):
                raise AuditInputError(f"{path} contains a non-object JSONL row")
            rows[_string(raw, "ref_id")] = raw
    return rows


def _object(raw: JsonObject, key: str) -> JsonObject:
    value = raw.get(key)
    if isinstance(value, dict):
        return value
    raise AuditInputError(f"{key} must be an object")


def _objects(raw: JsonObject, key: str) -> tuple[JsonObject, ...]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise AuditInputError(f"{key} must be a list")
    rows: list[JsonObject] = []
    for item in value:
        if not isinstance(item, dict):
            raise AuditInputError(f"{key} must contain only objects")
        rows.append(item)
    return tuple(rows)


def _strings(raw: JsonObject, key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise AuditInputError(f"{key} must be a list")
    return tuple(str(item) for item in value)


def _string(raw: JsonObject, key: str) -> str:
    value = raw.get(key)
    if isinstance(value, str):
        return value
    raise AuditInputError(f"{key} must be a string")


def _optional_string(raw: JsonObject, key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise AuditInputError(f"{key} must be a string when present")


def _bool(raw: JsonObject, key: str) -> bool:
    value = raw.get(key)
    if isinstance(value, bool):
        return value
    raise AuditInputError(f"{key} must be a boolean")


def _optional_float(raw: JsonObject, key: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    raise AuditInputError(f"{key} must be numeric when present")


def _result_image(results: JsonObject, case_id: str, variant: str) -> str:
    result = _object(results, f"{case_id}_{variant}")
    return _string(result, "image")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    summary_path: Annotated[Path, typer.Argument()],
    visual_audit_path: Annotated[Path, typer.Argument()],
    metrics_path: Annotated[Path, typer.Argument()],
    suite_manifest_path: Annotated[Path, typer.Argument()],
    output_path: Annotated[Path, typer.Argument()],
    summary_output_path: Annotated[Path, typer.Option()],
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    rows = build_audit_rows(
        summary_path=summary_path,
        visual_audit_path=visual_audit_path,
        metrics_path=metrics_path,
        suite_manifest_path=suite_manifest_path,
    )
    write_audit_outputs(rows, output_path, summary_output_path, overwrite=overwrite)
    typer.echo(f"wrote {output_path}")
    typer.echo(f"wrote {summary_output_path}")


if __name__ == "__main__":
    app()
