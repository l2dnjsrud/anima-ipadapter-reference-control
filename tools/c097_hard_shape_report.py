from __future__ import annotations

import json
from pathlib import Path

from tools.siglip_auto_caption_types import JsonObject


def write_c097_summary(path: Path, summary: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def c097_report(summary: JsonObject) -> str:
    lines = [
        "# C097 hard-shape data expansion",
        "",
        f"- decision: `{summary['decision']}`",
        f"- selected_rows: `{summary['selected_rows']}`",
        f"- explicit_negative_rows: `{summary['explicit_negative_rows']}`",
        f"- heldout_rows_used: `{summary['heldout_rows_used']}`",
        f"- heldout_rows_rejected: `{summary['heldout_rows_rejected']}`",
        f"- groups: `{json.dumps(summary['selected_group_counts'], ensure_ascii=False)}`",
        f"- manifest: `{summary['output_manifest']}`",
        f"- review_sheet: `{summary['review_sheet']}`",
        "",
        "C096은 10행 shallow encoder-LoRA라 hard-shape collapse를 깨지 못했다. C097은 더 깊은 SigLIP encoder adaptation 전에",
        "4개 비인간/마스코트 계열 crop-pair를 확장하고, 각 row에 다른 shape group negative를 붙여 다음 학습 입력을 만든다.",
        "",
    ]
    return "\n".join(lines)
