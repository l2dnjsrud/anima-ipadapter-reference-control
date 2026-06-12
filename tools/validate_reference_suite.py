from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import typer


@dataclass(frozen=True, slots=True)
class ReferenceSuiteValidation:
    manifest_path: str
    data_root: str
    rows: int
    missing_images: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_images


def validate_reference_suite(
    manifest_path: Path,
    data_root: Path,
) -> ReferenceSuiteValidation:
    missing: list[str] = []
    rows = 0
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            ref_id = str(raw["ref_id"])
            rows += 1
            image_path = data_root / f"{ref_id}.jpg"
            if not image_path.is_file():
                missing.append(str(image_path))
    return ReferenceSuiteValidation(
        manifest_path=str(manifest_path),
        data_root=str(data_root),
        rows=rows,
        missing_images=tuple(missing),
    )


app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest_path: Annotated[Path, typer.Argument()],
    data_root: Annotated[Path, typer.Option()],
) -> None:
    result = validate_reference_suite(manifest_path, data_root)
    typer.echo(json.dumps(asdict(result), ensure_ascii=True, indent=2))
    if not result.ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
