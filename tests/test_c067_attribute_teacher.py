from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c067_attribute_teacher_core import (
    C067Config,
    DEFAULT_ATTRIBUTE_QUERIES,
    build_attribute_manifest,
    score_attribute_manifest,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


class FakeAttributeScorer:
    def score(self, image_path: Path, candidate_texts: tuple[str, ...]) -> tuple[float, ...]:
        stem = image_path.stem
        return tuple(_fake_score(stem, text) for text in candidate_texts)


def test_build_attribute_manifest_excludes_heldout_from_c066_and_clean32(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    _write_image(dataset_root / "train/a.jpg", (30, 120, 60))
    _write_image(dataset_root / "heldout/b.jpg", (90, 90, 90))
    c066_path = tmp_path / "c066.jsonl"
    c066_path.write_text(
        "\n".join(
            [
                json.dumps(_c066_row("train/a", dataset_root / "train/a.jpg")),
                json.dumps(_c066_row("heldout/b", dataset_root / "heldout/b.jpg")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        json.dumps({"ref_id": "train/a", "tgt_id": "train/a", "prompt": "caption"})
        + "\n",
        encoding="utf-8",
    )
    heldout_path = tmp_path / "heldout.jsonl"
    heldout_path.write_text(
        json.dumps({"ref_id": "heldout/b", "tgt_id": "heldout/b", "prompt": "held"})
        + "\n",
        encoding="utf-8",
    )

    summary = build_attribute_manifest(
        C067Config(
            c066_manifest_path=c066_path,
            train_manifest_path=train_path,
            heldout_manifest_path=heldout_path,
            dataset_root=dataset_root,
            out_dir=tmp_path / "out",
        )
    )

    rows = _read_jsonl(tmp_path / "out" / "attribute_query_manifest.jsonl")
    assert summary["candidate_count"] == 1
    assert summary["heldout_rows_used"] == 0
    assert summary["query_count"] == len(DEFAULT_ATTRIBUTE_QUERIES)
    assert rows[0]["image_id"] == "train/a"
    assert rows[0]["attribute_queries"][0]["key"] == "direct_green_non_human_face"


def test_score_attribute_manifest_writes_topk_summary_and_review_sheet(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    green_path = dataset_root / "train/green_face.jpg"
    human_path = dataset_root / "train/human_face.jpg"
    _write_image(green_path, (20, 180, 60))
    _write_image(human_path, (140, 110, 90))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    manifest_path = out_dir / "attribute_query_manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True) + "\n"
            for row in (
                _manifest_row("train/green_face", green_path, "direct_green_pixel_candidate"),
                _manifest_row("train/human_face", human_path, "human_negative"),
            )
        ),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps({"candidate_count": 2, "heldout_rows_used": 0, "query_count": 6}),
        encoding="utf-8",
    )

    summary = score_attribute_manifest(
        manifest_path,
        out_dir=out_dir,
        scorer=FakeAttributeScorer(),
        top_k=1,
    )

    topk = json.loads((out_dir / "attribute_topk.json").read_text(encoding="utf-8"))
    assert summary["scorer_status"] == "scored"
    assert summary["score_rows"] == 2 * len(DEFAULT_ATTRIBUTE_QUERIES)
    assert topk["direct_green_non_human_face"][0]["image_id"] == "train/green_face"
    assert (out_dir / "attribute_review_sheet.jpg").is_file()
    assert "requires_manual_review" in summary["decision"]


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 128), color).save(path)


def _c066_row(image_id: str, image_path: Path) -> JsonObject:
    return {
        "image_id": image_id,
        "label": "positive",
        "source_bucket": "direct_green_pixel_candidate",
        "candidate_source": "image_green_pixel_scan",
        "matched_keywords": [],
        "selected_attributes": [],
        "caption": "caption",
        "image_path": str(image_path),
        "caption_path": str(image_path.with_suffix(".txt")),
        "green_ratio": 0.1,
        "strong_green_ratio": 0.02,
        "red_ratio": 0.01,
        "source_split": "train_or_dataset_scan",
        "heldout_excluded": False,
        "path_exists": True,
    }


def _manifest_row(image_id: str, image_path: Path, source_bucket: str) -> JsonObject:
    return {
        "candidate_id": image_id,
        "image_id": image_id,
        "image_path": str(image_path),
        "caption": "caption",
        "source_manifest": "test",
        "source_bucket": source_bucket,
        "source_label": "positive",
        "green_ratio": 0.1,
        "strong_green_ratio": 0.02,
        "red_ratio": 0.01,
        "attribute_queries": [
            {"key": query.key, "text": query.text, "kind": query.kind}
            for query in DEFAULT_ATTRIBUTE_QUERIES
        ],
    }


def _fake_score(stem: str, text: str) -> float:
    if "green" in stem and "green non-human" in text:
        return 0.9
    if "human" in stem and "ordinary human" in text:
        return 0.8
    return 0.1


def _read_jsonl(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return rows
