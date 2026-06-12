from __future__ import annotations

from pathlib import Path

from tools.build_true_identity_candidate_review import build_candidate_rows


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")
    path.with_suffix(".txt").write_text("caption", encoding="utf-8")


def _name(prefix: str, panel: str) -> str:
    return f"{prefix}_candidate_{panel}.jpg"


def test_build_candidate_rows_excludes_duplicate_panel_pairs(tmp_path: Path) -> None:
    panel_a = "00001_SG-001-01_page_100x200_s01"
    panel_b = "00002_SG-001-01_page_120x200_s02"
    _write_image(tmp_path / "001-100/SG-001" / _name("v4", panel_a))
    _write_image(tmp_path / "001-100/SG-001" / _name("v5", panel_a))
    _write_image(tmp_path / "001-100/SG-001" / _name("v4", panel_b))

    rows = build_candidate_rows(tmp_path, limit=4)

    assert len(rows) == 1
    assert rows[0].anchor_panel_key != rows[0].candidate_panel_key
    assert rows[0].sg_page == "SG-001-01"
    assert rows[0].review_label == "unlabeled"
