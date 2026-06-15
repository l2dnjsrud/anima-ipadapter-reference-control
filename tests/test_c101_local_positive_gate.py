from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.c101_local_positive_gate import C101Config, build_c101_annotation_package
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def test_c101_promotes_only_direct_green_prior_positive(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _image(data_root / "green.jpg", (30, 180, 50))
    _image(data_root / "object.jpg", (40, 160, 60))
    _image(data_root / "proxy.jpg", (170, 180, 180))
    out_dir = tmp_path / "out"

    summary = build_c101_annotation_package(
        _config(
            tmp_path,
            out_dir,
            c100_rows=(
                _c100("green", data_root / "green.jpg", "direct_green_pixel_candidate"),
                _c100("object", data_root / "object.jpg", "direct_green_pixel_candidate"),
                _c100("proxy", data_root / "proxy.jpg", "fang_profile_proxy"),
            ),
            prior_rows=(
                _prior("green", "target_positive", query_key="direct_green_non_human_face"),
                _prior("object", "false_positive_background_object"),
                _prior("proxy", "target_positive", query_key="red_glowing_eye"),
            ),
            min_reviewed_positive=1,
        )
    )

    labels = _labels_by_id(out_dir / "reviewed_local_labels.jsonl")
    assert set(labels) == {"green", "object", "proxy"}
    assert labels["green"]["manual_label"] == "local_positive"
    assert labels["object"]["manual_label"] == "local_negative"
    assert labels["proxy"]["manual_label"] == "unclear"
    assert summary["heldout_leakage_count"] == 0
    assert summary["review_required_count"] == 0
    assert summary["teacher_only_positive_count"] == 0
    assert summary["reviewed_local_positive_count"] == 1
    assert summary["decision"] == "c102_training_greenlit"


def test_c101_blocks_when_prior_reviews_are_only_proxy_or_negative(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _image(data_root / "proxy.jpg", (170, 180, 180))
    _image(data_root / "unknown.jpg", (80, 90, 80))
    out_dir = tmp_path / "out"

    summary = build_c101_annotation_package(
        _config(
            tmp_path,
            out_dir,
            c100_rows=(
                _c100("proxy", data_root / "proxy.jpg", "pale_non_human_proxy"),
                _c100("unknown", data_root / "unknown.jpg", "direct_green_pixel_candidate"),
            ),
            prior_rows=(
                _prior("proxy", "useful_proxy_positive", query_key="side_profile_silhouette"),
            ),
        )
    )

    labels = _labels_by_id(out_dir / "reviewed_local_labels.jsonl")
    assert labels["proxy"]["manual_label"] == "unclear"
    assert labels["unknown"]["manual_label"] == "unclear"
    assert summary["reviewed_local_positive_count"] == 0
    assert summary["unclear_count"] == 2
    assert summary["decision"] == "c102_blocked_needs_manual_annotation_or_teacher"


def test_c101_treats_human_negative_anchor_as_local_negative(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _image(data_root / "human.jpg", (180, 160, 140))
    out_dir = tmp_path / "out"

    build_c101_annotation_package(
        _config(
            tmp_path,
            out_dir,
            c100_rows=(_c100("human", data_root / "human.jpg", "direct_green_pixel_candidate"),),
            prior_rows=(_prior("human", "negative_anchor", query_key="human_negative"),),
        )
    )

    labels = _labels_by_id(out_dir / "reviewed_local_labels.jsonl")
    assert labels["human"]["manual_label"] == "local_negative"


def test_c101_blocks_if_candidate_manifest_contains_heldout_row(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _image(data_root / "green.jpg", (30, 180, 50))
    _image(data_root / "heldout.jpg", (20, 180, 30))
    out_dir = tmp_path / "out"

    summary = build_c101_annotation_package(
        _config(
            tmp_path,
            out_dir,
            c100_rows=(
                _c100("green", data_root / "green.jpg", "direct_green_pixel_candidate"),
                _c100("heldout", data_root / "heldout.jpg", "direct_green_pixel_candidate"),
            ),
            prior_rows=(
                _prior("green", "target_positive", query_key="direct_green_non_human_face"),
                _prior("heldout", "target_positive", query_key="direct_green_non_human_face"),
            ),
            min_reviewed_positive=1,
        )
    )

    assert summary["input_candidate_rows"] == 2
    assert summary["reviewed_rows"] == 1
    assert summary["heldout_leakage_count"] == 1
    assert summary["decision"] == "c102_blocked_needs_manual_annotation_or_teacher"


def _config(
    tmp_path: Path,
    out_dir: Path,
    *,
    c100_rows: tuple[JsonObject, ...],
    prior_rows: tuple[JsonObject, ...],
    min_reviewed_positive: int = 8,
) -> C101Config:
    c100_manifest = tmp_path / "c100.jsonl"
    c100_summary = tmp_path / "c100.summary.json"
    c100_review_sheet = tmp_path / "c100.jpg"
    heldout_manifest = tmp_path / "heldout.jsonl"
    prior_labels = tmp_path / "prior.jsonl"
    _write_jsonl(c100_manifest, c100_rows)
    _write_jsonl(heldout_manifest, ({"ref_id": "heldout"},))
    _write_jsonl(prior_labels, prior_rows)
    c100_summary.write_text(
        json.dumps({"decision": "c101_blocked_needs_manual_annotation_or_teacher", "candidate_rows": len(c100_rows)}),
        encoding="utf-8",
    )
    Image.new("RGB", (32, 32), "white").save(c100_review_sheet)
    return C101Config(
        c100_manifest=c100_manifest,
        c100_summary=c100_summary,
        c100_review_sheet=c100_review_sheet,
        heldout_manifest=heldout_manifest,
        prior_label_paths=(prior_labels,),
        out_dir=out_dir,
        plan_path=tmp_path / "plan.md",
        min_reviewed_positive=min_reviewed_positive,
    )


def _c100(image_id: str, image_path: Path, bucket: str) -> JsonObject:
    return {
        "image_id": image_id,
        "source_type": "real_local_color",
        "source_bucket": bucket,
        "candidate_source": "test",
        "label": "positive",
        "review_label": "needs_review",
        "review_status": "needs_review",
        "caption": "caption",
        "image_path": str(image_path),
        "green_ratio": 0.2,
        "strong_green_ratio": 0.1,
        "red_ratio": 0.0,
        "paths_ok": image_path.is_file(),
    }


def _prior(image_id: str, review_label: str, *, query_key: str = "direct_green_non_human_face") -> JsonObject:
    return {
        "image_id": image_id,
        "query_key": query_key,
        "review_label": review_label,
        "review_note": "prior visual review",
        "score": 0.5,
    }


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path)


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _labels_by_id(path: Path) -> dict[str, JsonObject]:
    rows: dict[str, JsonObject] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows[str(raw["image_id"])] = raw
    return rows
