from __future__ import annotations

import csv
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from tools.c071_seed_package import LABEL_SCHEMA
from tools.siglip_auto_caption_types import JsonObject, JsonValue

DEFAULT_CANDIDATES: Final = Path("eval/c072_external_direct_green_source_discovery_20260612/external_candidates.jsonl")
DEFAULT_OUT_DIR: Final = Path("eval/c073_external_candidate_visual_review_20260612")
DEFAULT_SCRATCH: Final = Path(".tmp/c073_external_candidate_visual_review")
MAX_IMAGE_BYTES: Final = 10_000_000
MINIMUM_TARGET_POSITIVES: Final = 4
SOURCE_C072_COMMIT: Final = "ad60ea7"

type FetchImage = Callable[[str, Path, float, int], "DownloadOutcome"]


@dataclass(frozen=True, slots=True)
class C073VisualReviewConfig:
    candidates_path: Path = DEFAULT_CANDIDATES
    out_dir: Path = DEFAULT_OUT_DIR
    scratch_dir: Path = DEFAULT_SCRATCH
    labels_path: Path | None = None
    timeout_seconds: float = 15.0
    max_image_bytes: int = MAX_IMAGE_BYTES


@dataclass(frozen=True, slots=True)
class DownloadOutcome:
    status: str
    byte_count: int
    width: int
    height: int
    error: str


class C073ReviewError(ValueError):
    pass


def build_c073_external_candidate_visual_review(
    config: C073VisualReviewConfig,
    *,
    fetch_image: FetchImage | None = None,
) -> JsonObject:
    selected_fetch_image = _fetch_image if fetch_image is None else fetch_image
    candidates = _read_jsonl(config.candidates_path)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = config.out_dir / "manual_visual_labels.csv" if config.labels_path is None else config.labels_path
    images_dir = config.scratch_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    downloads = tuple(_download_candidate(row, images_dir, config, selected_fetch_image) for row in candidates)
    _write_jsonl(config.out_dir / "download_manifest.jsonl", downloads)
    _write_label_template(config.out_dir / "visual_label_template.csv", downloads)
    sheet_path = config.scratch_dir / "contact_sheet.jpg"
    contact_sheet_written = _write_contact_sheet(downloads, sheet_path)
    if labels_path.is_file():
        reviewed = _review_downloads(downloads, _read_labels(labels_path))
        _write_jsonl(config.out_dir / "reviewed_external_labels.jsonl", reviewed)
    else:
        reviewed = ()
    summary = _summary(downloads, reviewed, sheet_path=sheet_path, contact_sheet_written=contact_sheet_written)
    (config.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (config.out_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _download_candidate(
    row: JsonObject,
    images_dir: Path,
    config: C073VisualReviewConfig,
    fetch_image: FetchImage,
) -> JsonObject:
    candidate_id = str(row["candidate_id"])
    local_path = images_dir / f"{candidate_id}.jpg"
    outcome = fetch_image(str(row["image_path"]), local_path, config.timeout_seconds, config.max_image_bytes)
    return dict(row) | {
        "download_status": outcome.status,
        "download_error": outcome.error,
        "downloaded_bytes": outcome.byte_count,
        "image_width": outcome.width,
        "image_height": outcome.height,
        "local_image_path": str(local_path),
        "large_downloads_performed": False,
    }


def _fetch_image(url: str, destination: Path, timeout_seconds: float, max_image_bytes: int) -> DownloadOutcome:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content = response.read(max_image_bytes + 1)
    except urllib.error.URLError as exc:
        return DownloadOutcome("failed", 0, 0, 0, f"url_error:{exc.reason}")
    except TimeoutError:
        return DownloadOutcome("failed", 0, 0, 0, "timeout")
    except OSError as exc:
        return DownloadOutcome("failed", 0, 0, 0, f"os_error:{exc}")
    if len(content) > max_image_bytes:
        return DownloadOutcome("failed", len(content), 0, 0, "too_large")
    destination.write_bytes(content)
    try:
        with Image.open(destination) as image:
            rgb = image.convert("RGB")
            rgb.save(destination, quality=94)
            width, height = rgb.size
    except UnidentifiedImageError:
        destination.unlink(missing_ok=True)
        return DownloadOutcome("failed", len(content), 0, 0, "unidentified_image")
    except OSError as exc:
        destination.unlink(missing_ok=True)
        return DownloadOutcome("failed", len(content), 0, 0, f"image_error:{exc}")
    return DownloadOutcome("downloaded", len(content), width, height, "")


def _write_contact_sheet(rows: tuple[JsonObject, ...], output_path: Path) -> bool:
    downloaded = tuple(row for row in rows if row["download_status"] == "downloaded")
    if not downloaded:
        return False
    cell_w = 260
    cell_h = 310
    cols = 4
    rows_count = (len(downloaded) + cols - 1) // cols
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows_count), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(downloaded):
        x = (index % cols) * cell_w
        y = (index // cols) * cell_h
        with Image.open(Path(str(row["local_image_path"]))) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (cell_w, cell_h - 74))
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y))
        draw.text((x + 4, y + cell_h - 70), str(row["candidate_id"])[:34], fill="black")
        draw.text((x + 4, y + cell_h - 52), str(row["source_buckets"])[:34], fill="black")
        draw.text((x + 4, y + cell_h - 34), str(row["external_license_note"])[:34], fill="black")
        draw.text((x + 4, y + cell_h - 16), str(row["review_notes"])[:34], fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return True


def _review_downloads(downloads: tuple[JsonObject, ...], labels: tuple[JsonObject, ...]) -> tuple[JsonObject, ...]:
    label_map = {str(row["candidate_id"]): row for row in labels}
    reviewed: list[JsonObject] = []
    for row in downloads:
        if row["download_status"] != "downloaded":
            continue
        label_row = label_map.get(str(row["candidate_id"]))
        if label_row is None:
            raise C073ReviewError(f"missing visual label for downloaded candidate: {row['candidate_id']}")
        manual_label = str(label_row.get("manual_label", ""))
        if manual_label not in LABEL_SCHEMA:
            raise C073ReviewError(f"unknown visual label: {manual_label}")
        reviewed.append(
            dict(row) | {
                "manual_label": manual_label,
                "manual_note": str(label_row.get("manual_note", "")),
                "visual_confirmation": manual_label == "target_positive",
            }
        )
    return tuple(reviewed)


def _summary(
    downloads: tuple[JsonObject, ...],
    reviewed: tuple[JsonObject, ...],
    *,
    sheet_path: Path,
    contact_sheet_written: bool,
) -> JsonObject:
    target_count = len({str(row["image_id"]) for row in reviewed if row["manual_label"] == "target_positive"})
    labels_pending = bool(downloads) and not reviewed
    decision = _decision(target_count, labels_pending)
    return {
        "source": "c073_external_candidate_visual_review",
        "source_c072_commit": SOURCE_C072_COMMIT,
        "candidate_count": len(downloads),
        "c072_candidate_rows": len(downloads),
        "downloaded_count": sum(1 for row in downloads if row["download_status"] == "downloaded"),
        "failed_count": sum(1 for row in downloads if row["download_status"] != "downloaded"),
        "reviewed_rows": len(reviewed),
        "label_counts": _count(reviewed, "manual_label"),
        "unique_target_positive_count": target_count,
        "target_positive_confirmed_count": target_count,
        "minimum_target_positive_required": MINIMUM_TARGET_POSITIVES,
        "label_schema": list(LABEL_SCHEMA),
        "heldout_rows_used": sum(1 for row in downloads if bool(row.get("heldout_excluded", False))),
        "large_downloads_performed": False,
        "contact_sheet_path": str(sheet_path),
        "contact_sheet_written": contact_sheet_written,
        "committed_external_images": False,
        "committed_external_image_count": 0,
        "decision": decision,
        "next_training_or_data_action": _next_action(decision),
    }


def _decision(target_count: int, labels_pending: bool) -> str:
    if labels_pending:
        return "visual_labels_pending"
    if target_count >= MINIMUM_TARGET_POSITIVES:
        return "ready_for_encoder_training"
    return "external_manual_data_required"


def _next_action(decision: str) -> str:
    match decision:
        case "ready_for_encoder_training":
            return "import reviewed_external_labels target positives into the next supervised encoder-feature training manifest"
        case "visual_labels_pending":
            return "fill manual_visual_labels.csv from the contact sheet and rerun c073 review"
        case "external_manual_data_required":
            return "collect more external/user/manual direct-green non-human target positives before training"
        case unreachable:
            raise C073ReviewError(f"unexpected decision: {unreachable}")


def _report(summary: JsonObject) -> str:
    lines = [
        "# c073 external candidate visual review",
        "",
        f"- decision: `{summary['decision']}`",
        f"- downloaded_count: {summary['downloaded_count']}",
        f"- reviewed_rows: {summary['reviewed_rows']}",
        f"- unique_target_positive_count: {summary['unique_target_positive_count']}",
        f"- large_downloads_performed: {str(summary['large_downloads_performed']).lower()}",
        f"- committed_external_images: {str(summary['committed_external_images']).lower()}",
        f"- contact_sheet_path: `{summary['contact_sheet_path']}`",
        "",
        "## Next action",
        str(summary["next_training_or_data_action"]),
        "",
    ]
    return "\n".join(lines)


def _write_label_template(path: Path, rows: tuple[JsonObject, ...]) -> None:
    fieldnames = ("candidate_id", "image_id", "download_status", "manual_label", "manual_note", "allowed_labels")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "image_id": row["image_id"],
                    "download_status": row["download_status"],
                    "manual_label": "",
                    "manual_note": "",
                    "allowed_labels": "|".join(LABEL_SCHEMA),
                }
            )


def _read_labels(path: Path) -> tuple[JsonObject, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _count(rows: tuple[JsonObject, ...], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    build_c073_external_candidate_visual_review(C073VisualReviewConfig())
