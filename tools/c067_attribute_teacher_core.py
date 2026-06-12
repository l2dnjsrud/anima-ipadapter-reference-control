from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw, ImageOps

from tools.siglip_auto_caption_types import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class AttributeQuery:
    key: str
    text: str
    kind: str


@dataclass(frozen=True, slots=True)
class C067Config:
    c066_manifest_path: Path
    train_manifest_path: Path
    heldout_manifest_path: Path
    dataset_root: Path
    out_dir: Path


class AttributeTextScorer(Protocol):
    def score(self, image_path: Path, candidate_texts: tuple[str, ...]) -> tuple[float, ...]:
        """Return image-text similarity scores in query order."""


DEFAULT_ATTRIBUTE_QUERIES: tuple[AttributeQuery, ...] = (
    AttributeQuery(
        "direct_green_non_human_face",
        "green non-human demon face with red glowing eye, colored skin, monster portrait",
        "target_positive",
    ),
    AttributeQuery(
        "red_glowing_eye",
        "single red glowing demonic eye on a fantasy martial arts character",
        "target_positive",
    ),
    AttributeQuery(
        "side_profile_silhouette",
        "side profile portrait silhouette, sharp nose profile, character face in profile",
        "target_positive",
    ),
    AttributeQuery(
        "beard_headwear_crop",
        "old bearded martial arts master or court official with black hat, close-up crop",
        "target_positive",
    ),
    AttributeQuery(
        "human_negative",
        "ordinary human wuxia martial artist face, natural skin, normal human portrait",
        "negative_anchor",
    ),
    AttributeQuery(
        "background_object_green",
        "green background objects, leaves, plants, palace decor, not a green character",
        "false_positive_guard",
    ),
)


def build_attribute_manifest(config: C067Config) -> JsonObject:
    heldout_ids = _read_ids(config.heldout_manifest_path)
    rows = _candidate_rows(config, heldout_ids=heldout_ids)
    manifest_path = config.out_dir / "attribute_query_manifest.jsonl"
    _write_jsonl(manifest_path, rows)
    summary: JsonObject = {
        "candidate_count": len(rows),
        "query_count": len(DEFAULT_ATTRIBUTE_QUERIES),
        "heldout_rows_used": sum(1 for row in rows if str(row["image_id"]) in heldout_ids),
        "missing_paths": sum(1 for row in rows if not Path(str(row["image_path"])).is_file()),
        "source_counts": dict(Counter(str(row["source_manifest"]) for row in rows)),
        "scorer_status": "not_scored",
        "decision": "ready_for_attribute_scoring",
    }
    _write_summary_and_report(config.out_dir, summary)
    return summary


def score_attribute_manifest(
    manifest_path: Path,
    *,
    out_dir: Path,
    scorer: AttributeTextScorer,
    top_k: int = 8,
) -> JsonObject:
    rows = _read_jsonl(manifest_path)
    query_texts = tuple(query.text for query in DEFAULT_ATTRIBUTE_QUERIES)
    score_rows: list[JsonObject] = []
    by_query: dict[str, list[JsonObject]] = {query.key: [] for query in DEFAULT_ATTRIBUTE_QUERIES}
    for row in rows:
        image_path = Path(str(row["image_path"]))
        scores = scorer.score(image_path, query_texts)
        for query, score in zip(DEFAULT_ATTRIBUTE_QUERIES, scores, strict=True):
            scored = _score_row(row, query, float(score))
            score_rows.append(scored)
            by_query[query.key].append(scored)
    topk = _topk(by_query, top_k=top_k)
    _write_jsonl(out_dir / "attribute_scores.jsonl", score_rows)
    (out_dir / "attribute_topk.json").write_text(
        json.dumps(topk, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_review_sheet(topk, out_dir / "attribute_review_sheet.jpg")
    summary = _load_summary(out_dir)
    summary.update(
        {
            "scorer_status": "scored",
            "score_rows": len(score_rows),
            "top_k": top_k,
            "direct_green_teacher_candidate_count": _direct_green_count(topk),
            "decision": _decision(topk),
        }
    )
    _write_summary_and_report(out_dir, summary)
    return summary


def _candidate_rows(config: C067Config, *, heldout_ids: set[str]) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    seen: set[str] = set()
    for raw in _read_jsonl(config.train_manifest_path):
        image_id = str(raw["ref_id"])
        if image_id in heldout_ids or image_id in seen:
            continue
        image_path = config.dataset_root / f"{image_id}.jpg"
        seen.add(image_id)
        rows.append(_from_train(raw, image_path))
    for raw in _read_jsonl(config.c066_manifest_path):
        image_id = str(raw["image_id"])
        if image_id in heldout_ids or image_id in seen:
            continue
        seen.add(image_id)
        rows.append(_from_c066(raw))
    return tuple(rows)


def _from_c066(raw: JsonObject) -> JsonObject:
    return {
        "candidate_id": str(raw["image_id"]),
        "image_id": str(raw["image_id"]),
        "image_path": str(raw["image_path"]),
        "caption": str(raw.get("caption", "")),
        "source_manifest": "c066",
        "source_bucket": str(raw.get("source_bucket", "unknown")),
        "source_label": str(raw.get("label", "unknown")),
        "green_ratio": float(raw.get("green_ratio", 0.0)),
        "strong_green_ratio": float(raw.get("strong_green_ratio", 0.0)),
        "red_ratio": float(raw.get("red_ratio", 0.0)),
        "attribute_queries": _query_objects(),
    }


def _from_train(raw: JsonObject, image_path: Path) -> JsonObject:
    return {
        "candidate_id": str(raw["ref_id"]),
        "image_id": str(raw["ref_id"]),
        "image_path": str(image_path),
        "caption": str(raw.get("prompt", "")),
        "source_manifest": "clean32_train",
        "source_bucket": "clean32_train",
        "source_label": "unlabeled_train",
        "green_ratio": 0.0,
        "strong_green_ratio": 0.0,
        "red_ratio": 0.0,
        "attribute_queries": _query_objects(),
    }


def _score_row(candidate: JsonObject, query: AttributeQuery, score: float) -> JsonObject:
    return {
        "image_id": str(candidate["image_id"]),
        "image_path": str(candidate["image_path"]),
        "source_manifest": str(candidate["source_manifest"]),
        "source_bucket": str(candidate["source_bucket"]),
        "source_label": str(candidate["source_label"]),
        "query_key": query.key,
        "query_text": query.text,
        "query_kind": query.kind,
        "score": score,
    }


def _topk(by_query: dict[str, list[JsonObject]], *, top_k: int) -> dict[str, list[JsonObject]]:
    top: dict[str, list[JsonObject]] = {}
    for key, values in by_query.items():
        ranked = sorted(values, key=lambda row: (-float(row["score"]), str(row["image_id"])))
        top[key] = ranked[:top_k]
    return top


def _direct_green_count(topk: dict[str, list[JsonObject]]) -> int:
    direct = topk.get("direct_green_non_human_face", [])
    guard = {str(row["image_id"]): float(row["score"]) for row in topk.get("background_object_green", [])}
    return sum(1 for row in direct if float(row["score"]) >= guard.get(str(row["image_id"]), -1.0))


def _decision(topk: dict[str, list[JsonObject]]) -> str:
    if _direct_green_count(topk) >= 1:
        return "candidate_teacher_seed_requires_manual_review"
    return "insufficient_teacher_seed_requires_manual_annotation"


def _write_review_sheet(topk: dict[str, list[JsonObject]], output_path: Path) -> None:
    cell_w = 224
    cell_h = 244
    keys = tuple(topk)
    row_count = max((len(rows) for rows in topk.values()), default=1)
    sheet = Image.new("RGB", (cell_w * len(keys), cell_h * (row_count + 1)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, key in enumerate(keys):
        draw.text((col * cell_w + 8, 8), key[:28], fill="black")
        for row_index, row in enumerate(topk[key]):
            x = col * cell_w
            y = (row_index + 1) * cell_h
            with Image.open(Path(str(row["image_path"]))) as image:
                thumb = ImageOps.fit(image.convert("RGB"), (cell_w, cell_h - 44))
            sheet.paste(thumb, (x, y))
            label = f"{float(row['score']):.3f} {str(row['source_bucket'])[:18]}"
            draw.text((x + 4, y + cell_h - 40), label, fill="black")
            draw.text((x + 4, y + cell_h - 22), Path(str(row["image_id"])).name[:28], fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def _query_objects() -> list[JsonObject]:
    return [asdict(query) for query in DEFAULT_ATTRIBUTE_QUERIES]


def _read_ids(path: Path) -> set[str]:
    return {str(row["ref_id"]) for row in _read_jsonl(path)}


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw: JsonValue = json.loads(line)
            if isinstance(raw, dict):
                rows.append(raw)
    return tuple(rows)


def _write_jsonl(path: Path, rows: list[JsonObject] | tuple[JsonObject, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_summary(out_dir: Path) -> JsonObject:
    path = out_dir / "summary.json"
    if not path.is_file():
        return {}
    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _write_summary_and_report(out_dir: Path, summary: JsonObject) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")


def _render_report(summary: JsonObject) -> str:
    lines = [
        "# c067 Attribute Teacher / Reranker Seed",
        "",
        f"- Candidate count: `{summary.get('candidate_count', 0)}`",
        f"- Query count: `{summary.get('query_count', len(DEFAULT_ATTRIBUTE_QUERIES))}`",
        f"- Heldout rows used: `{summary.get('heldout_rows_used', 0)}`",
        f"- Scorer status: `{summary.get('scorer_status', 'not_scored')}`",
        f"- Decision: `{summary.get('decision', 'pending')}`",
        "",
        "## Attribute Queries",
        "",
    ]
    for query in DEFAULT_ATTRIBUTE_QUERIES:
        lines.append(f"- `{query.key}` ({query.kind}): {query.text}")
    lines.append("")
    return "\n".join(lines)
