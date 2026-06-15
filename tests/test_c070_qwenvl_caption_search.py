from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c070_qwenvl_caption_search import (
    C070Config,
    build_c070_qwenvl_caption_search,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c070_caption_search_uses_semantic_caption_and_excludes_heldout(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    _write_pair(dataset_root, "train/green_monster", (30, 170, 70), "green-skinned demon monster face")
    _write_pair(dataset_root, "train/leaf", (25, 180, 55), "generic manhwa character panel")
    _write_pair(dataset_root, "train/red_eye", (180, 40, 40), "red glowing demonic eye")
    _write_pair(dataset_root, "heldout/green_monster", (20, 190, 60), "green non-human monster face")
    all_manifest = tmp_path / "all.jsonl"
    heldout_manifest = tmp_path / "heldout.jsonl"
    c069_reviewed = tmp_path / "c069.jsonl"
    _write_jsonl(
        all_manifest,
        (
            {"ref_id": "train/green_monster"},
            {"ref_id": "train/leaf"},
            {"ref_id": "train/red_eye"},
            {"ref_id": "heldout/green_monster"},
        ),
    )
    _write_jsonl(heldout_manifest, ({"ref_id": "heldout/green_monster"},))
    _write_jsonl(c069_reviewed, ({"image_id": "train/leaf"},))

    summary = build_c070_qwenvl_caption_search(
        C070Config(
            dataset_root=dataset_root,
            all_manifest_path=all_manifest,
            heldout_manifest_path=heldout_manifest,
            c069_reviewed_path=c069_reviewed,
            out_dir=tmp_path / "out",
            top_k_per_bucket=2,
        )
    )

    candidates = _read_jsonl(tmp_path / "out" / "candidate_manifest.jsonl")
    reviewed = _read_jsonl(tmp_path / "out" / "reviewed_candidate_labels.jsonl")
    assert summary["heldout_rows_used"] == 0
    assert summary["caption_keyword_hit_images"] == 2
    assert summary["direct_green_target_positive_count"] == 1
    assert summary["decision"] == "external_manual_data_required"
    assert {row["image_id"] for row in candidates}.isdisjoint({"heldout/green_monster"})
    assert any(row["review_label"] == "target_positive" for row in reviewed)
    assert (tmp_path / "out" / "annotated_review_sheet.jpg").is_file()


def _write_pair(root: Path, stem: str, color: tuple[int, int, int], caption: str) -> None:
    image_path = root / f"{stem}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 128), color).save(image_path)
    image_path.with_suffix(".txt").write_text(caption, encoding="utf-8")


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return rows
