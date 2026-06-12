from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Sequence, TypedDict

import typer
from PIL import Image

from tools.reference_prompt_manifest import (
    MissingReferenceImageError,
    build_reference_prompt_rows,
    load_reference_source_rows,
    validate_reference_source_images,
    write_reference_prompt_rows,
)
from tools.reference_prompting import default_attribute_candidates


DEFAULT_QWENVL_MODEL_ID: Final = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_IMAGE_INSTRUCTION: Final = (
    "Represent this manhwa/anime reference image for character identity, "
    "visible attributes, color palette, expression, and costume."
)
DEFAULT_TEXT_INSTRUCTION: Final = (
    "Represent this short manhwa/anime character attribute phrase for "
    "image-text retrieval."
)


class ImageInput(TypedDict):
    image: Image.Image


@dataclass(frozen=True, slots=True)
class Qwen3VLScorerConfig:
    model_id: str
    image_instruction: str = DEFAULT_IMAGE_INSTRUCTION
    text_instruction: str = DEFAULT_TEXT_INSTRUCTION


class Qwen3VLReferenceTextScorer:
    def __init__(self, config: Qwen3VLScorerConfig) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"torch_dtype": torch.bfloat16} if device == "cuda" else {}
        self._model = SentenceTransformer(
            config.model_id,
            device=device,
            model_kwargs=model_kwargs,
        )
        self._config = config

    def score(self, image_path: Path, candidate_texts: tuple[str, ...]) -> tuple[float, ...]:
        import torch

        with Image.open(image_path) as image:
            image_input: ImageInput = {"image": image.convert("RGB")}
            image_embedding = self._model.encode(
                [image_input],
                normalize_embeddings=True,
                convert_to_tensor=True,
                prompt=self._config.image_instruction,
            )
        text_embedding = self._model.encode(
            list(candidate_texts),
            normalize_embeddings=True,
            convert_to_tensor=True,
            prompt=self._config.text_instruction,
        )
        scores = torch.as_tensor(image_embedding)[0].float().cpu() @ torch.as_tensor(
            text_embedding
        ).float().cpu().T
        return tuple(float(value) for value in scores.tolist())


app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest_path: Annotated[Path, typer.Argument()],
    dataset_root: Annotated[Path, typer.Argument()],
    output_path: Annotated[Path, typer.Argument()],
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    max_per_category: Annotated[int, typer.Option(min=1)] = 1,
    model_id: Annotated[str, typer.Option()] = DEFAULT_QWENVL_MODEL_ID,
) -> None:
    rows = _limit_rows(load_reference_source_rows(manifest_path), limit)
    try:
        validate_reference_source_images(rows, dataset_root)
    except MissingReferenceImageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    scorer = Qwen3VLReferenceTextScorer(Qwen3VLScorerConfig(model_id=model_id))
    prompt_rows = build_reference_prompt_rows(
        rows,
        dataset_root=dataset_root,
        scorer=scorer,
        candidates=default_attribute_candidates(),
        max_per_category=max_per_category,
    )
    write_reference_prompt_rows(prompt_rows, output_path)
    typer.echo(f"wrote {len(prompt_rows)} prompt rows to {output_path}")


def _limit_rows[
    RowT
](rows: Sequence[RowT], limit: int | None) -> tuple[RowT, ...]:
    if limit is None:
        return tuple(rows)
    return tuple(rows[:limit])


if __name__ == "__main__":
    app()
