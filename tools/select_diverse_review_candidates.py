from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, TypeAlias, TypedDict


JsonPrimitive: TypeAlias = str | int | float | bool | None


class RankedRow(TypedDict):
    pair_id: str
    anchor_id: str
    candidate_id: str
    sg_page: str
    similarity: float
    rank: int


class FaceScoreRow(TypedDict):
    pair_id: str
    anchor_face_upper_score: float
    candidate_face_upper_score: float


@dataclass(frozen=True, slots=True)
class DiverseReviewCandidateRow:
    pair_id: str
    anchor_id: str
    candidate_id: str
    sg_page: str
    anchor_face_upper_score: float
    candidate_face_upper_score: float
    qwenvl_similarity: float
    qwenvl_rank: int
    page_seen_before: bool


@dataclass(frozen=True, slots=True)
class DiverseSelectionSummary:
    input_ranked_rows: int
    eligible_rows: int
    selected_rows: int
    unique_sg_pages: int
    new_page_rows: int
    old_page_rows: int
    skipped_seen_pair_ids: int
    min_face_score: float
    min_similarity: float


@dataclass(frozen=True, slots=True)
class DiverseSelectionResult:
    rows: tuple[DiverseReviewCandidateRow, ...]
    summary: DiverseSelectionSummary


@dataclass(frozen=True, slots=True)
class DiverseSelectionInputError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def select_diverse_review_candidates(
    ranked_path: Path,
    *,
    face_scores_path: Path,
    reviewed_paths: tuple[Path, ...],
    target_count: int,
    max_per_sg_page: int,
    min_face_score: float,
    min_similarity: float,
) -> DiverseSelectionResult:
    if target_count < 1:
        raise DiverseSelectionInputError("target_count must be >= 1")
    if max_per_sg_page < 1:
        raise DiverseSelectionInputError("max_per_sg_page must be >= 1")
    reviewed_pair_ids, reviewed_pages = _read_reviewed_sets(reviewed_paths)
    face_scores = _read_face_scores(face_scores_path)
    ranked_rows = _read_ranked_rows(ranked_path)
    skipped_seen_pair_ids = sum(1 for row in ranked_rows if row["pair_id"] in reviewed_pair_ids)
    eligible_rows = tuple(
        _join_ranked_face(row, face_scores[row["pair_id"]], reviewed_pages=reviewed_pages)
        for row in ranked_rows
        if _is_eligible(
            row,
            face_scores=face_scores,
            reviewed_pair_ids=reviewed_pair_ids,
            min_face_score=min_face_score,
            min_similarity=min_similarity,
        )
    )
    selected_rows = _select_by_page_diversity(
        eligible_rows,
        target_count=target_count,
        max_per_sg_page=max_per_sg_page,
    )
    return DiverseSelectionResult(
        rows=selected_rows,
        summary=DiverseSelectionSummary(
            input_ranked_rows=len(ranked_rows),
            eligible_rows=len(eligible_rows),
            selected_rows=len(selected_rows),
            unique_sg_pages=len({row.sg_page for row in selected_rows}),
            new_page_rows=sum(1 for row in selected_rows if not row.page_seen_before),
            old_page_rows=sum(1 for row in selected_rows if row.page_seen_before),
            skipped_seen_pair_ids=skipped_seen_pair_ids,
            min_face_score=min_face_score,
            min_similarity=min_similarity,
        ),
    )


def write_diverse_review_candidates(
    result: DiverseSelectionResult,
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in result.rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def write_selection_summary(result: DiverseSelectionResult, *, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(asdict(result.summary), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _is_eligible(
    row: RankedRow,
    *,
    face_scores: Mapping[str, FaceScoreRow],
    reviewed_pair_ids: set[str],
    min_face_score: float,
    min_similarity: float,
) -> bool:
    if row["pair_id"] in reviewed_pair_ids:
        return False
    scores = _require_face_score(face_scores, row["pair_id"])
    return (
        row["similarity"] >= min_similarity
        and scores["anchor_face_upper_score"] >= min_face_score
        and scores["candidate_face_upper_score"] >= min_face_score
    )


def _select_by_page_diversity(
    rows: tuple[DiverseReviewCandidateRow, ...],
    *,
    target_count: int,
    max_per_sg_page: int,
) -> tuple[DiverseReviewCandidateRow, ...]:
    selected: list[DiverseReviewCandidateRow] = []
    page_counts: dict[str, int] = {}
    for require_new_page in (True, False):
        for row in rows:
            if row in selected:
                continue
            if require_new_page and row.page_seen_before:
                continue
            if not require_new_page and not row.page_seen_before:
                continue
            count = page_counts.get(row.sg_page, 0)
            if count >= max_per_sg_page:
                continue
            selected.append(row)
            page_counts[row.sg_page] = count + 1
            if len(selected) >= target_count:
                return tuple(selected)
    return tuple(selected)


def _join_ranked_face(
    row: RankedRow,
    scores: FaceScoreRow,
    *,
    reviewed_pages: set[str],
) -> DiverseReviewCandidateRow:
    return DiverseReviewCandidateRow(
        pair_id=row["pair_id"],
        anchor_id=row["anchor_id"],
        candidate_id=row["candidate_id"],
        sg_page=row["sg_page"],
        anchor_face_upper_score=scores["anchor_face_upper_score"],
        candidate_face_upper_score=scores["candidate_face_upper_score"],
        qwenvl_similarity=row["similarity"],
        qwenvl_rank=row["rank"],
        page_seen_before=row["sg_page"] in reviewed_pages,
    )


def _read_reviewed_sets(paths: tuple[Path, ...]) -> tuple[set[str], set[str]]:
    pair_ids: set[str] = set()
    sg_pages: set[str] = set()
    for path in paths:
        for line_number, raw in _read_jsonl_objects(path):
            pair_ids.add(_require_str(raw, "pair_id", line_number))
            sg_pages.add(_require_str(raw, "sg_page", line_number))
    return pair_ids, sg_pages


def _read_ranked_rows(path: Path) -> tuple[RankedRow, ...]:
    rows: list[RankedRow] = []
    for line_number, raw in _read_jsonl_objects(path):
        rows.append(
            {
                "pair_id": _require_str(raw, "pair_id", line_number),
                "anchor_id": _require_str(raw, "anchor_id", line_number),
                "candidate_id": _require_str(raw, "candidate_id", line_number),
                "sg_page": _require_str(raw, "sg_page", line_number),
                "similarity": _require_float(raw, "similarity", line_number),
                "rank": _require_int(raw, "rank", line_number),
            }
        )
    return tuple(sorted(rows, key=lambda row: row["rank"]))


def _read_face_scores(path: Path) -> Mapping[str, FaceScoreRow]:
    rows: dict[str, FaceScoreRow] = {}
    for line_number, raw in _read_jsonl_objects(path):
        pair_id = _require_str(raw, "pair_id", line_number)
        rows[pair_id] = {
            "pair_id": pair_id,
            "anchor_face_upper_score": _require_float(raw, "anchor_face_upper_score", line_number),
            "candidate_face_upper_score": _require_float(
                raw,
                "candidate_face_upper_score",
                line_number,
            ),
        }
    return rows


def _require_face_score(scores: Mapping[str, FaceScoreRow], pair_id: str) -> FaceScoreRow:
    if pair_id not in scores:
        raise DiverseSelectionInputError(f"missing face score for pair_id {pair_id}")
    return scores[pair_id]


def _read_jsonl_objects(path: Path) -> tuple[tuple[int, Mapping[str, JsonPrimitive]], ...]:
    rows: list[tuple[int, Mapping[str, JsonPrimitive]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise DiverseSelectionInputError(f"row {line_number} must be an object")
            rows.append((line_number, _parse_jsonl_row(raw, line_number=line_number)))
    return tuple(rows)


def _parse_jsonl_row(
    raw: Mapping[JsonPrimitive, JsonPrimitive],
    *,
    line_number: int,
) -> Mapping[str, JsonPrimitive]:
    parsed: dict[str, JsonPrimitive] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise DiverseSelectionInputError(f"row {line_number} has non-string key")
        parsed[key] = value
    return parsed


def _require_str(raw: Mapping[str, JsonPrimitive], key: str, line_number: int) -> str:
    value = _require_key(raw, key, line_number)
    if not isinstance(value, str):
        raise DiverseSelectionInputError(f"row {line_number} field {key} must be a string")
    return value


def _require_float(raw: Mapping[str, JsonPrimitive], key: str, line_number: int) -> float:
    value = _require_key(raw, key, line_number)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DiverseSelectionInputError(f"row {line_number} field {key} must be numeric")
    return float(value)


def _require_int(raw: Mapping[str, JsonPrimitive], key: str, line_number: int) -> int:
    value = _require_key(raw, key, line_number)
    if isinstance(value, bool):
        raise DiverseSelectionInputError(f"row {line_number} field {key} must be an int")
    if not isinstance(value, int):
        raise DiverseSelectionInputError(f"row {line_number} field {key} must be an int")
    return value


def _require_key(raw: Mapping[str, JsonPrimitive], key: str, line_number: int) -> JsonPrimitive:
    if key not in raw:
        raise DiverseSelectionInputError(f"row {line_number} missing field {key}")
    return raw[key]
