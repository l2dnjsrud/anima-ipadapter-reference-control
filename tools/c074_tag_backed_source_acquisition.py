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
from tools.siglip_auto_caption_types import JsonObject

OUT_DIR: Final = Path("eval/c074_tag_backed_direct_green_source_acquisition_20260612")
SCRATCH: Final = Path(".tmp/c074_tag_backed_direct_green_source_acquisition")
SOURCE_C073_COMMIT: Final = "778e8e8"
MINIMUM_TARGET_POSITIVES: Final = 4
MAX_BYTES: Final = 4_194_304

type FetchImage = Callable[[str, Path], bool]


@dataclass(frozen=True, slots=True)
class C074Config:
    out_dir: Path = OUT_DIR
    scratch_dir: Path = SCRATCH
    labels_path: Path | None = None


def build_c074_tag_backed_source_acquisition(
    config: C074Config,
    *,
    fetch_image: FetchImage | None = None,
) -> JsonObject:
    selected_fetch = _fetch_image if fetch_image is None else fetch_image
    config.out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = config.scratch_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    sources = _source_rows()
    candidates = _candidate_rows()
    downloads = tuple(_download(row, images_dir, selected_fetch) for row in candidates)
    _write_jsonl(config.out_dir / "source_manifest.jsonl", sources)
    _write_jsonl(config.out_dir / "external_candidates.jsonl", candidates)
    _write_jsonl(config.out_dir / "download_manifest.jsonl", downloads)
    _write_template(config.out_dir / "candidate_template.csv", downloads)
    sheet_path = config.scratch_dir / "contact_sheet.jpg"
    _write_sheet(downloads, sheet_path)
    labels_path = config.out_dir / "manual_visual_labels.csv" if config.labels_path is None else config.labels_path
    reviewed = _review(downloads, labels_path) if labels_path.is_file() else ()
    if reviewed:
        _write_jsonl(config.out_dir / "reviewed_external_labels.jsonl", reviewed)
    summary = _summary(sources, downloads, reviewed, sheet_path)
    (config.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (config.out_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _source_rows() -> tuple[JsonObject, ...]:
    return (
        _source(
            "CyberHarem/neeko_leagueoflegends",
            "mit; not-for-all-audiences; source images crawled from multiple sites",
            "direct_sample_assets",
            10,
            "core tags include green_skin, colored_skin, tail, monster_girl; cluster 0 excluded due adult tags",
        ),
        _source("OneIG-Bench/OneIG-Bench", "cc-by-nc-4.0", "prompt_only_no_image_rows", 0, "green skin/lizard tail prompts found but no target image assets"),
        _source("CaptionEmporium/anime-caption-danbooru-2021-sfw-5m-hq", "cc-by-sa-4.0", "caption_tags_no_image_rows", 0, "tag-rich source; viewer search unstable and rows do not expose images"),
        _source("mrzjy/splash-art-gacha-collection-10k", "cc-by-sa-4.0", "image_caption_search_timeout", 0, "image rows exist but live search timed out for green queries"),
    )


def _source(repo: str, license_note: str, image_status: str, candidates: int, note: str) -> JsonObject:
    return {
        "repo": repo,
        "official_url": f"https://huggingface.co/datasets/{repo}",
        "access_status": "public",
        "license_note": license_note,
        "image_url_availability": image_status,
        "potential_candidate_count": candidates,
        "probe_note": note,
    }


def _candidate_rows() -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for cluster in (1, 2):
        for sample in range(5):
            image_id = f"external/CyberHarem/neeko_leagueoflegends/samples/{cluster}/clu{cluster}-sample{sample}"
            rows.append(
                {
                    "candidate_id": f"c074_neeko_c{cluster}_{sample}",
                    "image_id": image_id,
                    "image_path": f"https://huggingface.co/datasets/CyberHarem/neeko_leagueoflegends/resolve/main/samples/{cluster}/clu{cluster}-sample{sample}.png",
                    "source_bucket": "tag_backed_direct_green_non_human",
                    "suggested_label": "target_positive",
                    "source_experiments": ["c074_tag_backed_source_acquisition"],
                    "source_labels": ["green_skin", "colored_skin", "tail", "monster_girl"],
                    "source_buckets": ["CyberHarem/neeko_leagueoflegends"],
                    "review_notes": ["neeko dataset core tags; adult-tagged cluster 0 excluded"],
                    "rank": len(rows) + 1,
                    "external_source_url": "https://huggingface.co/datasets/CyberHarem/neeko_leagueoflegends",
                    "external_license_note": "mit; not-for-all-audiences; source copyright should be reviewed before redistribution",
                    "heldout_excluded": False,
                }
            )
    return tuple(rows)


def _download(row: JsonObject, images_dir: Path, fetch_image: FetchImage) -> JsonObject:
    local_path = images_dir / f"{row['candidate_id']}.jpg"
    ok = fetch_image(str(row["image_path"]), local_path)
    width, height = _dimensions(local_path) if ok else (0, 0)
    return dict(row) | {
        "download_status": "downloaded" if ok else "failed",
        "local_image_path": str(local_path),
        "image_width": width,
        "image_height": height,
        "large_downloads_performed": False,
    }


def _fetch_image(url: str, destination: Path) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            content = response.read(MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    if len(content) > MAX_BYTES:
        return False
    destination.write_bytes(content)
    try:
        with Image.open(destination) as image:
            image.convert("RGB").save(destination, quality=94)
    except (UnidentifiedImageError, OSError):
        destination.unlink(missing_ok=True)
        return False
    return True


def _dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _write_sheet(rows: tuple[JsonObject, ...], output_path: Path) -> None:
    downloaded = tuple(row for row in rows if row["download_status"] == "downloaded")
    if not downloaded:
        return
    cell_w, cell_h, cols = 240, 300, 5
    sheet = Image.new("RGB", (cell_w * cols, cell_h * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(downloaded):
        x, y = (index % cols) * cell_w, (index // cols) * cell_h
        with Image.open(Path(str(row["local_image_path"]))) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (cell_w, cell_h - 54))
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y))
        draw.text((x + 3, y + cell_h - 50), str(row["candidate_id"]), fill="black")
        draw.text((x + 3, y + cell_h - 30), "green_skin monster_girl", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def _review(downloads: tuple[JsonObject, ...], labels_path: Path) -> tuple[JsonObject, ...]:
    labels = {str(row["candidate_id"]): row for row in _read_csv(labels_path)}
    reviewed: list[JsonObject] = []
    for row in downloads:
        label = labels.get(str(row["candidate_id"]))
        if label is None:
            raise ValueError(f"missing c074 label: {row['candidate_id']}")
        manual_label = str(label["manual_label"])
        if manual_label not in LABEL_SCHEMA:
            raise ValueError(f"unknown c074 label: {manual_label}")
        reviewed.append(dict(row) | {"manual_label": manual_label, "manual_note": str(label.get("manual_note", "")), "visual_reviewed": True})
    return tuple(reviewed)


def _summary(sources: tuple[JsonObject, ...], downloads: tuple[JsonObject, ...], reviewed: tuple[JsonObject, ...], sheet_path: Path) -> JsonObject:
    positive_count = len({str(row["image_id"]) for row in reviewed if row.get("manual_label") == "target_positive"})
    decision = "ready_for_encoder_training" if positive_count >= MINIMUM_TARGET_POSITIVES else "external_manual_data_required"
    return {
        "source": "c074_tag_backed_direct_green_source_acquisition",
        "source_c073_commit": SOURCE_C073_COMMIT,
        "inspected_source_count": len(sources),
        "row_probe_count": len(sources),
        "source_candidate_counts": {str(row["repo"]): int(row["potential_candidate_count"]) for row in sources},
        "image_url_availability": {str(row["repo"]): str(row["image_url_availability"]) for row in sources},
        "candidate_count": len(downloads),
        "downloaded_count": sum(1 for row in downloads if row["download_status"] == "downloaded"),
        "reviewed_rows": len(reviewed),
        "label_counts": _count(reviewed, "manual_label"),
        "target_positive_confirmed_count": positive_count,
        "minimum_target_positive_required": MINIMUM_TARGET_POSITIVES,
        "label_schema": list(LABEL_SCHEMA),
        "heldout_rows_used": 0,
        "large_downloads_performed": False,
        "committed_external_image_count": 0,
        "contact_sheet_path": str(sheet_path),
        "license_caution": "CyberHarem/neeko declares MIT but is not-for-all-audiences and source images are crawled from multiple sites; review rights before redistribution.",
        "decision": decision,
        "next_training_or_data_action": "build a bounded c075 training manifest from visually confirmed c074 positives plus existing guards" if decision == "ready_for_encoder_training" else "collect more target positives",
    }


def _report(summary: JsonObject) -> str:
    return "\n".join(
        [
            "# c074 tag-backed direct-green source acquisition",
            "",
            f"- decision: `{summary['decision']}`",
            f"- inspected_source_count: {summary['inspected_source_count']}",
            f"- candidate_count: {summary['candidate_count']}",
            f"- downloaded_count: {summary['downloaded_count']}",
            f"- target_positive_confirmed_count: {summary['target_positive_confirmed_count']}",
            f"- license_caution: {summary['license_caution']}",
            f"- next_training_or_data_action: {summary['next_training_or_data_action']}",
            "",
        ]
    )


def _write_template(path: Path, rows: tuple[JsonObject, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("candidate_id", "image_id", "download_status", "manual_label", "manual_note"), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"candidate_id": row["candidate_id"], "image_id": row["image_id"], "download_status": row["download_status"], "manual_label": "", "manual_note": ""})


def _read_csv(path: Path) -> tuple[JsonObject, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _count(rows: tuple[JsonObject, ...], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    build_c074_tag_backed_source_acquisition(C074Config())
