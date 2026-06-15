# /// script
# dependencies = ["typer"]
# ///
# --- How to run -----------------------------------------------------
# PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
#   tools/c107_qwen_teacher_distillation.py build

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Final

import typer

from tools.c079_manifest_io import read_jsonl, write_jsonl
from tools.siglip_auto_caption_types import JsonObject, JsonValue


DEFAULT_SOURCE_MANIFEST: Final = Path("training/manifests/c097_siglip_hard_shape_expanded_pairs_20260613.jsonl")
DEFAULT_IMAGE_ROOT: Final = Path(".tmp/c097_siglip_hard_shape_expanded_root")
DEFAULT_SCORE_PATH: Final = Path("eval/c106_qwen_teacher_feature_distillation_20260613/qwenvl_pair_scores.json")
DEFAULT_OUTPUT_MANIFEST: Final = Path("training/manifests/c107_qwen_teacher_distillation_20260613.jsonl")
DEFAULT_SUMMARY_PATH: Final = Path("training/manifests/c107_qwen_teacher_distillation_20260613.summary.json")
DEFAULT_PLAN_PATH: Final = Path("docs/c107_qwen_teacher_distillation_plan_ko.md")
TEACHER_MARGIN: Final = 0.21772855200937813

app = typer.Typer(add_completion=False)


@dataclass(frozen=True, slots=True)
class C107DistillationError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class C107DistillationConfig:
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST
    image_root: Path = DEFAULT_IMAGE_ROOT
    score_path: Path = DEFAULT_SCORE_PATH
    output_manifest: Path = DEFAULT_OUTPUT_MANIFEST
    summary_path: Path = DEFAULT_SUMMARY_PATH
    plan_path: Path = DEFAULT_PLAN_PATH


@dataclass(frozen=True, slots=True)
class TeacherPairScore:
    positive: float
    negative: float
    margin: float
    positive_anchor: str
    positive_candidate: str
    negative_anchor: str
    negative_candidate: str


@dataclass(frozen=True, slots=True)
class SourceIds:
    ref_id: str
    tgt_id: str
    neg_id: str


@dataclass(frozen=True, slots=True)
class C107DistillationSummary:
    source: str
    source_manifest: str
    score_path: str
    image_root: str
    output_manifest: str
    rows_written: int
    explicit_negative_rows: int
    heldout_rows_used: int
    heldout_rows_rejected: int
    missing_path_count: int
    teacher_source: str
    c106_teacher_margin: float
    teacher_margin_mean: float
    teacher_margin_min: float
    teacher_margin_max: float
    decision: str


def build_c107_training_manifest(
    config: C107DistillationConfig = C107DistillationConfig(),
) -> C107DistillationSummary:
    teacher_scores = _teacher_score_map(config.score_path)
    rows: list[JsonObject] = []
    margins: list[float] = []
    heldout_rejected = 0
    selected_index = 0
    for line_number, source in read_jsonl(config.source_manifest):
        ids = _source_ids(source, config.source_manifest, line_number)
        if _has_heldout(ids):
            heldout_rejected += 1
            continue
        pair_id = f"pair_{selected_index:03d}"
        score = teacher_scores.get(pair_id)
        if score is None:
            raise C107DistillationError(f"missing teacher score for {pair_id}")
        _assert_pair_files(config.image_root, ids)
        _assert_score_ids(pair_id, score, ids)
        rows.append(_training_row(source, ids, score))
        margins.append(score.margin)
        selected_index += 1
    if not rows:
        raise C107DistillationError(f"no trainable c107 rows from {config.source_manifest}")
    write_jsonl(config.output_manifest, tuple(rows))
    summary = C107DistillationSummary(
        source="C097 hard-shape expanded pairs + C106 QwenVL teacher scores",
        source_manifest=str(config.source_manifest),
        score_path=str(config.score_path),
        image_root=str(config.image_root),
        output_manifest=str(config.output_manifest),
        rows_written=len(rows),
        explicit_negative_rows=sum(1 for row in rows if isinstance(row.get("neg_id"), str)),
        heldout_rows_used=0,
        heldout_rows_rejected=heldout_rejected,
        missing_path_count=0,
        teacher_source="C106",
        c106_teacher_margin=TEACHER_MARGIN,
        teacher_margin_mean=sum(margins) / len(margins),
        teacher_margin_min=min(margins),
        teacher_margin_max=max(margins),
        decision="c107_manifest_ready_for_bounded_training",
    )
    _write_json(config.summary_path, asdict(summary))
    _write_plan(config.plan_path, summary)
    return summary


def _teacher_score_map(score_path: Path) -> dict[str, TeacherPairScore]:
    raw = _read_json(score_path)
    raw_rows = raw.get("rows")
    if not isinstance(raw_rows, list):
        raise C107DistillationError(f"{score_path} missing rows")
    positives: dict[str, JsonObject] = {}
    negatives: dict[str, JsonObject] = {}
    for row_value in raw_rows:
        if not isinstance(row_value, dict):
            raise C107DistillationError(f"{score_path} score row must be object")
        row: JsonObject = row_value
        pair_id = _string(row, "pair_id", score_path)
        label = _string(row, "label", score_path)
        match label:
            case "positive":
                positives[pair_id] = row
            case "negative":
                negatives[pair_id] = row
            case _:
                raise C107DistillationError(f"{score_path} unknown score label: {label}")
    scores: dict[str, TeacherPairScore] = {}
    for pair_id, positive_row in positives.items():
        negative_row = negatives.get(pair_id)
        if negative_row is None:
            raise C107DistillationError(f"missing score labels for {pair_id}")
        positive = _float(positive_row, "cosine", score_path)
        negative = _float(negative_row, "cosine", score_path)
        scores[pair_id] = TeacherPairScore(
            positive=positive,
            negative=negative,
            margin=positive - negative,
            positive_anchor=_string(positive_row, "anchor_id", score_path),
            positive_candidate=_string(positive_row, "candidate_id", score_path),
            negative_anchor=_string(negative_row, "anchor_id", score_path),
            negative_candidate=_string(negative_row, "candidate_id", score_path),
        )
    return scores


def _source_ids(row: JsonObject, path: Path, line_number: int) -> SourceIds:
    return SourceIds(
        ref_id=_line_string(row, "ref_id", path, line_number),
        tgt_id=_line_string(row, "tgt_id", path, line_number),
        neg_id=_line_string(row, "neg_id", path, line_number),
    )


def _training_row(source: JsonObject, ids: SourceIds, score: TeacherPairScore) -> JsonObject:
    return {
        "ref_id": ids.ref_id,
        "tgt_id": ids.tgt_id,
        "neg_id": ids.neg_id,
        "prompt": str(source.get("prompt", "")),
        "shape_group": str(source.get("shape_group", "")),
        "negative_shape_group": str(source.get("negative_shape_group", "")),
        "source_pose_pair": str(source.get("source_pose_pair", "")),
        "teacher_source": "C106",
        "teacher_positive_cosine": score.positive,
        "teacher_negative_cosine": score.negative,
        "teacher_margin": score.margin,
        "c106_teacher_margin": TEACHER_MARGIN,
    }


def _assert_pair_files(image_root: Path, ids: SourceIds) -> None:
    required = (
        image_root / f"{ids.ref_id}.jpg",
        image_root / f"{ids.tgt_id}.jpg",
        image_root / f"{ids.tgt_id}.txt",
        image_root / f"{ids.neg_id}.jpg",
    )
    missing = tuple(path for path in required if not path.is_file())
    if missing:
        raise C107DistillationError("missing c107 training files: " + ", ".join(str(path) for path in missing[:5]))


def _assert_score_ids(pair_id: str, score: TeacherPairScore, ids: SourceIds) -> None:
    expected = (ids.ref_id, ids.tgt_id, ids.ref_id, ids.neg_id)
    actual = (
        score.positive_anchor,
        score.positive_candidate,
        score.negative_anchor,
        score.negative_candidate,
    )
    if actual != expected:
        raise C107DistillationError(f"teacher score id mismatch for {pair_id}")


def _has_heldout(ids: SourceIds) -> bool:
    return "heldout" in ids.ref_id or "heldout" in ids.tgt_id or "heldout" in ids.neg_id


def _read_json(path: Path) -> JsonObject:
    if not path.is_file():
        raise C107DistillationError(f"json not found: {path}")
    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise C107DistillationError(f"{path} must be a JSON object")
    return raw


def _string(row: JsonObject, key: str, path: Path) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise C107DistillationError(f"{path} missing {key}")
    return value


def _line_string(row: JsonObject, key: str, path: Path, line_number: int) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise C107DistillationError(f"{path}:{line_number} missing {key}")
    return value


def _float(row: JsonObject, key: str, path: Path) -> float:
    value = row.get(key)
    if not isinstance(value, int | float):
        raise C107DistillationError(f"{path} missing numeric {key}")
    return float(value)


def _write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_plan(path: Path, summary: C107DistillationSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# C107 Qwen Teacher Distillation Plan

C106에서 QwenVL teacher가 hard-shape positive/negative를 강하게 분리했으므로, 그 판단을 `neg_id` hard-negative contrastive 학습 manifest로 보존한다.

- Source manifest: `{summary.source_manifest}`
- Teacher score: `{summary.score_path}`
- Output manifest: `{summary.output_manifest}`
- Rows / explicit negatives: `{summary.rows_written}` / `{summary.explicit_negative_rows}`
- Heldout rows used: `{summary.heldout_rows_used}`
- C106 teacher margin: `{summary.c106_teacher_margin}`
- C107 mean row margin: `{summary.teacher_margin_mean}`
- Decision: `{summary.decision}`
""",
        encoding="utf-8",
    )


@app.command()
def build(
    source_manifest: Annotated[Path, typer.Option()] = DEFAULT_SOURCE_MANIFEST,
    image_root: Annotated[Path, typer.Option()] = DEFAULT_IMAGE_ROOT,
    score_path: Annotated[Path, typer.Option()] = DEFAULT_SCORE_PATH,
    output_manifest: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_MANIFEST,
    summary_path: Annotated[Path, typer.Option()] = DEFAULT_SUMMARY_PATH,
    plan_path: Annotated[Path, typer.Option()] = DEFAULT_PLAN_PATH,
) -> None:
    summary = build_c107_training_manifest(
        C107DistillationConfig(
            source_manifest=source_manifest,
            image_root=image_root,
            score_path=score_path,
            output_manifest=output_manifest,
            summary_path=summary_path,
            plan_path=plan_path,
        )
    )
    typer.echo(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
