from __future__ import annotations

import json
from pathlib import Path

from tools.select_diverse_review_candidates import (
    select_diverse_review_candidates,
    write_diverse_review_candidates,
)


def _write_jsonl(path: Path, rows: list[dict[str, str | int | float | bool]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_select_diverse_review_candidates_prefers_new_pages_and_fills_old_pages(
    tmp_path: Path,
) -> None:
    ranked_path = tmp_path / "ranked.jsonl"
    face_path = tmp_path / "face.jsonl"
    reviewed_path = tmp_path / "reviewed.jsonl"
    _write_jsonl(
        ranked_path,
        [
            _ranked("p1", "SG-A", 0.95, 1),
            _ranked("p2", "SG-A", 0.94, 2),
            _ranked("p3", "SG-B", 0.93, 3),
            _ranked("p4", "SG-C", 0.92, 4),
            _ranked("p5", "SG-D", 0.91, 5),
        ],
    )
    _write_jsonl(face_path, [_face(pair_id) for pair_id in ("p1", "p2", "p3", "p4", "p5")])
    _write_jsonl(
        reviewed_path,
        [
            {"pair_id": "p3", "sg_page": "SG-B"},
            {"pair_id": "old", "sg_page": "SG-D"},
        ],
    )

    result = select_diverse_review_candidates(
        ranked_path,
        face_scores_path=face_path,
        reviewed_paths=(reviewed_path,),
        target_count=3,
        max_per_sg_page=1,
        min_face_score=0.08,
        min_similarity=0.0,
    )

    assert [row.pair_id for row in result.rows] == ["p1", "p4", "p5"]
    assert result.summary.new_page_rows == 2
    assert result.summary.old_page_rows == 1
    assert result.summary.skipped_seen_pair_ids == 1


def test_write_diverse_review_candidates_preserves_scores(tmp_path: Path) -> None:
    ranked_path = tmp_path / "ranked.jsonl"
    face_path = tmp_path / "face.jsonl"
    output_path = tmp_path / "selected.jsonl"
    _write_jsonl(ranked_path, [_ranked("p1", "SG-A", 0.95, 1)])
    _write_jsonl(face_path, [_face("p1")])

    result = select_diverse_review_candidates(
        ranked_path,
        face_scores_path=face_path,
        reviewed_paths=(),
        target_count=1,
        max_per_sg_page=1,
        min_face_score=0.08,
        min_similarity=0.90,
    )
    write_diverse_review_candidates(result, output_path=output_path)

    [row] = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert row["pair_id"] == "p1"
    assert row["anchor_face_upper_score"] == 0.11
    assert row["candidate_face_upper_score"] == 0.12
    assert row["qwenvl_similarity"] == 0.95
    assert row["qwenvl_rank"] == 1


def _ranked(pair_id: str, sg_page: str, similarity: float, rank: int) -> dict[str, str | int | float]:
    return {
        "pair_id": pair_id,
        "anchor_id": f"{pair_id}/a",
        "candidate_id": f"{pair_id}/b",
        "sg_page": sg_page,
        "similarity": similarity,
        "rank": rank,
    }


def _face(pair_id: str) -> dict[str, str | float]:
    return {
        "pair_id": pair_id,
        "anchor_id": f"{pair_id}/a",
        "candidate_id": f"{pair_id}/b",
        "sg_page": "unused",
        "anchor_face_upper_score": 0.11,
        "candidate_face_upper_score": 0.12,
        "decision": "keep_face_upper_body_pair",
    }
