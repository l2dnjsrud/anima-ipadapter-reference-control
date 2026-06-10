"""Reference-control evaluation harness for PE-Core IP-Adapter checkpoints.

The harness has two phases:

* ``plan`` writes a deterministic manifest plus ``run_eval.sh``. The generated
  commands create one no-IP baseline and one IP job per scale for each
  reference/seed pair.
* ``score`` reads the manifest after generation, computes nonblank checks and
  PE pooled-cosine similarity against each reference, then writes CSV/JSON,
  a contact sheet, and a markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from PIL import Image, ImageDraw

from scripts.tasks._common import INFERENCE_BASE, ROOT

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
MEAN_UPLIFT_THRESHOLD = 0.03
IMPROVED_RATE_THRESHOLD = 0.75
MIN_PIXEL_STD = 5.0


@dataclass(frozen=True)
class PlanConfig:
    checkpoint: Path
    ref_root: Path
    out_dir: Path
    limit_refs: int
    refs: tuple[Path, ...] | None
    seeds: tuple[int, ...]
    scales: tuple[float, ...]
    prompt: str
    negative_prompt: str
    infer_steps: int
    guidance_scale: float
    flow_shift: float


@dataclass(frozen=True)
class ScoreRow:
    ref_id: str
    seed: int
    mode: Literal["no_ip", "ip"]
    scale: float | None
    image_path: Path
    cosine: float
    pixel_std: float
    pixel_mean: float
    pixel_min: int
    pixel_max: int


class RefEntry(TypedDict):
    ref_id: str
    path: str
    image_size: list[int]


class JobEntry(TypedDict):
    job_id: str
    ref_id: str
    ref_path: str
    seed: int
    mode: str
    scale: float | None
    job_dir: str
    image_size: list[int]
    command: list[str]


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def select_reference_images(root: Path, limit: int) -> list[Path]:
    """Select a deterministic, evenly-spaced subset of recursive image files."""
    pool = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit <= 0:
        return []
    if len(pool) <= limit:
        return pool
    if limit == 1:
        return [pool[0]]

    last = len(pool) - 1
    indices = [round(i * last / (limit - 1)) for i in range(limit)]
    selected: list[Path] = []
    seen: set[int] = set()
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(pool[idx])
    cursor = 0
    while len(selected) < limit and cursor < len(pool):
        if cursor not in seen:
            selected.append(pool[cursor])
            seen.add(cursor)
        cursor += 1
    return selected


def validate_reference_images(refs: tuple[Path, ...]) -> list[Path]:
    """Validate explicit reference image paths and preserve caller ordering."""
    selected: list[Path] = []
    missing: list[str] = []
    invalid: list[str] = []
    for ref in refs:
        if not ref.is_file():
            missing.append(str(ref))
            continue
        if ref.suffix.lower() not in IMAGE_SUFFIXES:
            invalid.append(str(ref))
            continue
        selected.append(ref)
    if missing:
        raise FileNotFoundError("Missing reference image(s): " + ", ".join(missing))
    if invalid:
        raise ValueError("Unsupported reference image suffix: " + ", ".join(invalid))
    return selected


def _closest_bucket_hw(image_path: Path) -> tuple[int, int]:
    """Return ``(H, W)`` from the closest constant-token bucket."""
    from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS

    with Image.open(image_path) as image:
        width, height = image.size
    target = width / height
    bucket_w, bucket_h = min(
        CONSTANT_TOKEN_BUCKETS,
        key=lambda wh: abs((wh[0] / wh[1]) - target),
    )
    return int(bucket_h), int(bucket_w)


def _job_dir(
    out_dir: Path, ref_id: str, seed: int, mode: str, scale: float | None
) -> Path:
    scale_part = "none" if scale is None else str(scale).replace(".", "p")
    return out_dir / "images" / ref_id / f"seed_{seed}" / f"{mode}_{scale_part}"


def _base_command(
    *,
    save_path: Path,
    prompt: str,
    negative_prompt: str,
    seed: int,
    infer_steps: int,
    guidance_scale: float,
    flow_shift: float,
    image_size: tuple[int, int],
) -> list[str]:
    return [
        *INFERENCE_BASE,
        "--save_path",
        str(save_path),
        "--prompt",
        prompt,
        "--negative_prompt",
        negative_prompt,
        "--seed",
        str(seed),
        "--infer_steps",
        str(infer_steps),
        "--guidance_scale",
        str(guidance_scale),
        "--flow_shift",
        str(flow_shift),
        "--image_size",
        str(image_size[0]),
        str(image_size[1]),
    ]


def _build_jobs(
    config: PlanConfig, refs: list[Path]
) -> tuple[list[RefEntry], list[JobEntry]]:
    ref_entries: list[RefEntry] = []
    jobs: list[JobEntry] = []
    for ref_index, ref_path in enumerate(refs):
        ref_id = f"ref{ref_index:02d}"
        image_size = _closest_bucket_hw(ref_path)
        ref_entries.append(
            {
                "ref_id": ref_id,
                "path": _repo_rel(ref_path),
                "image_size": [image_size[0], image_size[1]],
            }
        )
        for seed in config.seeds:
            no_ip_dir = _job_dir(config.out_dir, ref_id, seed, "no_ip", None)
            no_ip_command = _base_command(
                save_path=no_ip_dir,
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                seed=seed,
                infer_steps=config.infer_steps,
                guidance_scale=config.guidance_scale,
                flow_shift=config.flow_shift,
                image_size=image_size,
            )
            jobs.append(
                {
                    "job_id": f"{ref_id}_seed{seed}_no_ip",
                    "ref_id": ref_id,
                    "ref_path": _repo_rel(ref_path),
                    "seed": seed,
                    "mode": "no_ip",
                    "scale": None,
                    "job_dir": _repo_rel(no_ip_dir),
                    "image_size": [image_size[0], image_size[1]],
                    "command": no_ip_command,
                }
            )
            for scale in config.scales:
                ip_dir = _job_dir(config.out_dir, ref_id, seed, "ip", scale)
                ip_command = [
                    *_base_command(
                        save_path=ip_dir,
                        prompt=config.prompt,
                        negative_prompt=config.negative_prompt,
                        seed=seed,
                        infer_steps=config.infer_steps,
                        guidance_scale=config.guidance_scale,
                        flow_shift=config.flow_shift,
                        image_size=image_size,
                    ),
                    "--ip_adapter_weight",
                    str(config.checkpoint),
                    "--ip_image",
                    str(ref_path),
                    "--ip_scale",
                    str(scale),
                ]
                jobs.append(
                    {
                        "job_id": f"{ref_id}_seed{seed}_ip_{str(scale).replace('.', 'p')}",
                        "ref_id": ref_id,
                        "ref_path": _repo_rel(ref_path),
                        "seed": seed,
                        "mode": "ip",
                        "scale": float(scale),
                        "job_dir": _repo_rel(ip_dir),
                        "image_size": [image_size[0], image_size[1]],
                        "command": ip_command,
                    }
                )
    return ref_entries, jobs


def write_plan(config: PlanConfig) -> dict:
    """Write ``manifest.json`` and ``run_eval.sh`` for a deterministic sweep."""
    if not config.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint}")

    refs = (
        validate_reference_images(config.refs)
        if config.refs is not None
        else select_reference_images(config.ref_root, config.limit_refs)
    )
    if not refs:
        raise FileNotFoundError(f"No reference images found under {config.ref_root}")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    ref_entries, jobs = _build_jobs(config, refs)
    manifest = {
        "checkpoint": _repo_rel(config.checkpoint),
        "out_dir": _repo_rel(config.out_dir),
        "ref_root": _repo_rel(config.ref_root),
        "refs": ref_entries,
        "seeds": list(config.seeds),
        "scales": list(config.scales),
        "prompt": config.prompt,
        "negative_prompt": config.negative_prompt,
        "infer_steps": config.infer_steps,
        "guidance_scale": config.guidance_scale,
        "flow_shift": config.flow_shift,
        "thresholds": {
            "mean_uplift": MEAN_UPLIFT_THRESHOLD,
            "improved_rate": IMPROVED_RATE_THRESHOLD,
            "min_pixel_std": MIN_PIXEL_STD,
        },
        "jobs": jobs,
    }

    manifest_path = config.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    run_script = config.out_dir / "run_eval.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        "",
    ]
    lines.extend(shlex.join(job["command"]) for job in jobs)
    run_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mode = run_script.stat().st_mode
    run_script.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return manifest


def _newest_png(job_dir: Path) -> Path:
    pngs = sorted(
        (path for path in job_dir.glob("*.png") if not path.name.endswith("_ref.png")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not pngs:
        raise FileNotFoundError(f"No generated PNG found in {job_dir}")
    return pngs[0]


def _pixel_stats(image_path: Path) -> tuple[float, float, int, int]:
    with Image.open(image_path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    return float(arr.mean()), float(arr.std()), int(arr.min()), int(arr.max())


def _image_to_minus1to1(image_path: Path):
    import torch

    with Image.open(image_path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(arr / 127.5 - 1.0).permute(2, 0, 1).contiguous()
    return tensor


def _encode_pe_pooled(bundle, image_path: Path):
    from library.training.cmmd import pool_and_normalize
    from library.vision.encoder import encode_pe_from_imageminus1to1

    tensor = _image_to_minus1to1(image_path)
    feats = encode_pe_from_imageminus1to1(
        bundle, tensor.unsqueeze(0), same_bucket=True
    )[0]
    return pool_and_normalize(feats).cpu()


def score_manifest(
    manifest: dict,
    *,
    device: str,
    min_std: float,
    mean_uplift_threshold: float,
    improved_rate_threshold: float,
) -> tuple[list[ScoreRow], dict]:
    """Score generated images from a manifest."""
    import torch

    from library.vision.encoder import load_pe_encoder

    bundle = load_pe_encoder(torch.device(device), name="pe", dtype=torch.bfloat16)
    pooled_cache: dict[Path, torch.Tensor] = {}
    rows: list[ScoreRow] = []
    for job in manifest["jobs"]:
        job_dir = ROOT / job["job_dir"]
        image_path = _newest_png(job_dir)
        ref_path = ROOT / job["ref_path"]
        for path in (ref_path, image_path):
            if path not in pooled_cache:
                with torch.no_grad():
                    pooled_cache[path] = _encode_pe_pooled(bundle, path)
        cosine = torch.nn.functional.cosine_similarity(
            pooled_cache[ref_path],
            pooled_cache[image_path],
            dim=0,
        ).item()
        mean, std, pixel_min, pixel_max = _pixel_stats(image_path)
        rows.append(
            ScoreRow(
                ref_id=job["ref_id"],
                seed=int(job["seed"]),
                mode="ip" if job["mode"] == "ip" else "no_ip",
                scale=None if job["scale"] is None else float(job["scale"]),
                image_path=image_path,
                cosine=float(cosine),
                pixel_std=std,
                pixel_mean=mean,
                pixel_min=pixel_min,
                pixel_max=pixel_max,
            )
        )
    summary = summarize_scores(
        rows,
        min_std=min_std,
        mean_uplift_threshold=mean_uplift_threshold,
        improved_rate_threshold=improved_rate_threshold,
    )
    return rows, summary


def summarize_scores(
    rows: list[ScoreRow],
    *,
    min_std: float,
    mean_uplift_threshold: float,
    improved_rate_threshold: float,
) -> dict:
    """Aggregate score rows into per-scale quality summaries."""
    baselines: dict[tuple[str, int], ScoreRow] = {
        (row.ref_id, row.seed): row for row in rows if row.mode == "no_ip"
    }
    scale_rows: dict[float, list[tuple[ScoreRow, ScoreRow]]] = {}
    for row in rows:
        if row.mode != "ip" or row.scale is None:
            continue
        base = baselines[(row.ref_id, row.seed)]
        scale_rows.setdefault(row.scale, []).append((row, base))

    scale_summaries: list[dict] = []
    for scale, pairs in sorted(scale_rows.items()):
        uplifts = [ip.cosine - base.cosine for ip, base in pairs]
        improved = [uplift > 0.0 for uplift in uplifts]
        scale_summaries.append(
            {
                "scale": scale,
                "cases": len(pairs),
                "mean_ip_cosine": sum(ip.cosine for ip, _base in pairs) / len(pairs),
                "mean_no_ip_cosine": sum(base.cosine for _ip, base in pairs)
                / len(pairs),
                "mean_uplift": sum(uplifts) / len(uplifts),
                "improved_rate": sum(1 for flag in improved if flag) / len(improved),
            }
        )
    if not scale_summaries:
        raise ValueError("No IP score rows found")

    best = max(
        scale_summaries,
        key=lambda item: (
            item["mean_uplift"],
            item["improved_rate"],
            item["mean_ip_cosine"],
        ),
    )
    nonblank = all(row.pixel_std > min_std for row in rows)
    passed = (
        nonblank
        and best["mean_uplift"] >= mean_uplift_threshold
        and best["improved_rate"] >= improved_rate_threshold
    )
    return {
        "pass": passed,
        "best_scale": best["scale"],
        "generated_count": len(rows),
        "nonblank": nonblank,
        "thresholds": {
            "mean_uplift": mean_uplift_threshold,
            "improved_rate": improved_rate_threshold,
            "min_pixel_std": min_std,
        },
        "scale_summaries": scale_summaries,
    }


def write_scores_csv(rows: list[ScoreRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "ref_id",
                "seed",
                "mode",
                "scale",
                "image_path",
                "cosine",
                "pixel_std",
                "pixel_mean",
                "pixel_min",
                "pixel_max",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.ref_id,
                    row.seed,
                    row.mode,
                    "" if row.scale is None else row.scale,
                    _repo_rel(row.image_path),
                    f"{row.cosine:.6f}",
                    f"{row.pixel_std:.6f}",
                    f"{row.pixel_mean:.6f}",
                    row.pixel_min,
                    row.pixel_max,
                ]
            )


def write_contact_sheet(
    rows: list[ScoreRow], path: Path, *, thumb_size: tuple[int, int] = (192, 192)
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows, key=lambda row: (row.ref_id, row.seed, row.mode, row.scale or -1.0)
    )
    cols = min(4, max(1, len(ordered)))
    label_h = 34
    rows_n = (len(ordered) + cols - 1) // cols
    sheet = Image.new(
        "RGB", (cols * thumb_size[0], rows_n * (thumb_size[1] + label_h)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(ordered):
        col = idx % cols
        grid_row = idx // cols
        x = col * thumb_size[0]
        y = grid_row * (thumb_size[1] + label_h)
        with Image.open(row.image_path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail(thumb_size)
        sheet.paste(thumb, (x, y + label_h))
        scale = "none" if row.scale is None else f"{row.scale:g}"
        draw.text(
            (x + 4, y + 4), f"{row.ref_id} s{row.seed} {row.mode} {scale}", fill="black"
        )
        draw.text(
            (x + 4, y + 18),
            f"cos={row.cosine:.3f} std={row.pixel_std:.1f}",
            fill="black",
        )
    sheet.save(path)


def _manifest_artifact_path(manifest: dict, manifest_path: Path | None) -> str:
    if manifest_path is not None:
        return _repo_rel(manifest_path)
    out_dir = manifest.get("out_dir")
    if isinstance(out_dir, str):
        return str(Path(out_dir) / "manifest.json")
    return "manifest.json"


def _run_script_artifact_path(manifest: dict) -> str:
    out_dir = manifest.get("out_dir")
    if isinstance(out_dir, str):
        return str(Path(out_dir) / "run_eval.sh")
    return "run_eval.sh"


def write_report(
    manifest: dict,
    summary: dict,
    path: Path,
    *,
    manifest_path: Path | None = None,
    score_command: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if summary["pass"] else "FAIL"
    manifest_artifact = _manifest_artifact_path(manifest, manifest_path)
    run_script_artifact = _run_script_artifact_path(manifest)
    score_command_text = (
        shlex.join(score_command)
        if score_command is not None
        else shlex.join(
            [
                "python",
                "bench/ip_adapter/reference_eval.py",
                "score",
                "--manifest",
                manifest_artifact,
                "--min-std",
                str(summary["thresholds"]["min_pixel_std"]),
                "--mean-uplift-threshold",
                str(summary["thresholds"]["mean_uplift"]),
                "--improved-rate-threshold",
                str(summary["thresholds"]["improved_rate"]),
            ]
        )
    )
    lines = [
        "# IP-Adapter Reference-Control Evaluation",
        "",
        f"**Status:** {status}",
        f"**Checkpoint:** `{manifest['checkpoint']}`",
        f"**Prompt:** {manifest.get('prompt', '')}",
        f"**Seeds:** {manifest.get('seeds', [])}",
        f"**Scales:** {manifest.get('scales', [])}",
        "",
        "## Thresholds",
        "",
        f"- Mean uplift: `{summary['thresholds']['mean_uplift']}`",
        f"- Improved rate: `{summary['thresholds']['improved_rate']}`",
        f"- Min pixel std: `{summary['thresholds']['min_pixel_std']}`",
        "",
        "## Summary",
        "",
        f"- Best scale: `{summary['best_scale']}`",
        f"- Generated images: `{summary['generated_count']}`",
        f"- Nonblank: `{summary['nonblank']}`",
        "",
        "## Commands",
        "",
        f"- Generation manifest: `{manifest_artifact}`",
        f"- Generation script: `{run_script_artifact}`",
        "- Generation command coverage: `run_eval.sh` contains the exact no-IP and IP-scale command for every manifest job.",
        "- Score command:",
        "",
        "```bash",
        score_command_text,
        "```",
        "",
        "| Scale | Cases | Mean IP Cos | Mean No-IP Cos | Mean Uplift | Improved Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["scale_summaries"]:
        lines.append(
            "| {scale:g} | {cases} | {mean_ip_cosine:.4f} | {mean_no_ip_cosine:.4f} | "
            "{mean_uplift:.4f} | {improved_rate:.2%} |".format(**item)
        )
    lines.extend(["", "## References", ""])
    for ref in manifest.get("refs", []):
        lines.append(f"- `{ref['ref_id']}`: `{ref['path']}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- PE pooled-cosine measures visual/reference proximity in the same PE-Core family used by this local IP-Adapter path.",
            "- This dataset currently uses path-derived groups for generic layout/style captions, so the result is layout/style reference-control rather than verified character identity recovery.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plan_from_args(args: argparse.Namespace) -> int:
    config = PlanConfig(
        checkpoint=Path(args.checkpoint),
        ref_root=Path(args.ref_root),
        out_dir=Path(args.out_dir),
        limit_refs=args.limit_refs,
        refs=None if args.ref is None else tuple(Path(path) for path in args.ref),
        seeds=tuple(args.seeds),
        scales=tuple(args.scales),
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.guidance_scale,
        flow_shift=args.flow_shift,
    )
    manifest = write_plan(config)
    print(f"refs={len(manifest['refs'])} jobs={len(manifest['jobs'])}")
    print(f"manifest={config.out_dir / 'manifest.json'}")
    print(f"run_script={config.out_dir / 'run_eval.sh'}")
    print("thresholds: mean_uplift>=0.03 improved_rate>=0.75 min_pixel_std>5.0")
    return 0


def _score_from_args(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = _read_manifest(manifest_path)
    rows, summary = score_manifest(
        manifest,
        device=args.device,
        min_std=args.min_std,
        mean_uplift_threshold=args.mean_uplift_threshold,
        improved_rate_threshold=args.improved_rate_threshold,
    )
    out_dir = ROOT / manifest["out_dir"]
    write_scores_csv(rows, out_dir / "scores.csv")
    summary_full = {
        **summary,
        "selected_checkpoint": manifest["checkpoint"],
        "refs": manifest["refs"],
        "seeds": manifest["seeds"],
        "scales": manifest["scales"],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_full, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_contact_sheet(rows, out_dir / "contact_sheet.jpg")
    score_command = [
        sys.executable,
        *sys.argv,
    ]
    write_report(
        manifest,
        summary_full,
        out_dir / "report.md",
        manifest_path=manifest_path,
        score_command=score_command,
    )
    print(f"pass={summary_full['pass']} best_scale={summary_full['best_scale']}")
    print(f"report={out_dir / 'report.md'}")
    return 0 if summary_full["pass"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="write manifest and run_eval.sh")
    plan.add_argument("--checkpoint", required=True)
    plan.add_argument("--ref-root", default="post_image_dataset/resized")
    plan.add_argument(
        "--ref",
        action="append",
        help="Explicit reference image path. Repeat to fix the evaluation set.",
    )
    plan.add_argument("--out-dir", required=True)
    plan.add_argument("--limit-refs", type=int, default=4)
    plan.add_argument("--seeds", type=int, nargs="+", default=[1234, 2234])
    plan.add_argument("--scales", type=float, nargs="+", default=[0.6, 0.9, 1.2])
    plan.add_argument(
        "--prompt",
        default=(
            "masterpiece, best quality, score_7, safe. full color manga panel, "
            "clean linework, expressive character acting, cinematic composition."
        ),
    )
    plan.add_argument("--negative-prompt", default="")
    plan.add_argument("--infer-steps", type=int, default=16)
    plan.add_argument("--guidance-scale", type=float, default=3.5)
    plan.add_argument("--flow-shift", type=float, default=3.0)
    plan.set_defaults(func=_plan_from_args)

    score = sub.add_parser("score", help="score generated images from manifest")
    score.add_argument("--manifest", required=True)
    score.add_argument("--device", default="cuda:0")
    score.add_argument("--min-std", type=float, default=MIN_PIXEL_STD)
    score.add_argument(
        "--mean-uplift-threshold", type=float, default=MEAN_UPLIFT_THRESHOLD
    )
    score.add_argument(
        "--improved-rate-threshold", type=float, default=IMPROVED_RATE_THRESHOLD
    )
    score.set_defaults(func=_score_from_args)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
