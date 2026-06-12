from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from tools.reference_prompting import (
    AttributeCandidate,
    build_reference_prompt,
    default_attribute_candidates,
    select_top_attributes,
)


class MissingReferenceImageError(Exception):
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"reference image does not exist: {self.image_path}"


@dataclass(frozen=True, slots=True)
class ReferencePromptSourceRow:
    ref_id: str
    tgt_id: str
    prompt: str


@dataclass(frozen=True, slots=True)
class ManifestPromptRow:
    ref_id: str
    tgt_id: str
    source_prompt: str
    prompt: str
    selected_attributes: tuple[str, ...]


class ReferenceTextScorer(Protocol):
    def score(self, image_path: Path, candidate_texts: tuple[str, ...]) -> tuple[float, ...]:
        """Return image-text similarity scores in candidate order."""


def build_reference_prompt_rows(
    rows: tuple[ReferencePromptSourceRow, ...],
    *,
    dataset_root: Path,
    scorer: ReferenceTextScorer,
    candidates: tuple[AttributeCandidate, ...] | None = None,
    max_per_category: int = 1,
) -> tuple[ManifestPromptRow, ...]:
    prompt_candidates = candidates if candidates is not None else default_attribute_candidates()
    candidate_texts = tuple(candidate.text for candidate in prompt_candidates)
    validate_reference_source_images(rows, dataset_root)
    built_rows: list[ManifestPromptRow] = []
    for row in rows:
        image_path = dataset_root / f"{row.ref_id}.jpg"
        scores = scorer.score(image_path, candidate_texts)
        selected = select_top_attributes(
            prompt_candidates,
            scores,
            max_per_category=max_per_category,
        )
        built_rows.append(
            ManifestPromptRow(
                ref_id=row.ref_id,
                tgt_id=row.tgt_id,
                source_prompt=row.prompt,
                prompt=build_reference_prompt(row.prompt, selected),
                selected_attributes=tuple(candidate.text for candidate in selected),
            )
        )
    return tuple(built_rows)


def validate_reference_source_images(
    rows: tuple[ReferencePromptSourceRow, ...],
    dataset_root: Path,
) -> None:
    for row in rows:
        image_path = dataset_root / f"{row.ref_id}.jpg"
        if not image_path.is_file():
            raise MissingReferenceImageError(image_path)


def load_reference_source_rows(path: Path) -> tuple[ReferencePromptSourceRow, ...]:
    rows: list[ReferencePromptSourceRow] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            rows.append(
                ReferencePromptSourceRow(
                    ref_id=str(raw["ref_id"]),
                    tgt_id=str(raw["tgt_id"]),
                    prompt=str(raw["prompt"]),
                )
            )
    return tuple(rows)


def write_reference_prompt_rows(
    rows: tuple[ManifestPromptRow, ...],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")
