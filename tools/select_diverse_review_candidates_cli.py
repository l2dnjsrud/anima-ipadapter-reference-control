from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from tools.select_diverse_review_candidates import (
    select_diverse_review_candidates,
    write_diverse_review_candidates,
    write_selection_summary,
)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    ranked_path: Annotated[Path, typer.Argument()],
    face_scores_path: Annotated[Path, typer.Argument()],
    output_path: Annotated[Path, typer.Argument()],
    reviewed_path: Annotated[list[Path] | None, typer.Option("--reviewed-path")] = None,
    summary_path: Annotated[Path | None, typer.Option()] = None,
    target_count: Annotated[int, typer.Option(min=1)] = 32,
    max_per_sg_page: Annotated[int, typer.Option(min=1)] = 1,
    min_face_score: Annotated[float, typer.Option()] = 0.07,
    min_similarity: Annotated[float, typer.Option()] = 0.0,
) -> None:
    result = select_diverse_review_candidates(
        ranked_path,
        face_scores_path=face_scores_path,
        reviewed_paths=tuple(reviewed_path or ()),
        target_count=target_count,
        max_per_sg_page=max_per_sg_page,
        min_face_score=min_face_score,
        min_similarity=min_similarity,
    )
    write_diverse_review_candidates(result, output_path=output_path)
    if summary_path is not None:
        write_selection_summary(result, summary_path=summary_path)
    typer.echo(json.dumps(asdict(result.summary), ensure_ascii=True))


if __name__ == "__main__":
    app()
