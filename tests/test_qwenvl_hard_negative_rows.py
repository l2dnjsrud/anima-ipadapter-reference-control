from __future__ import annotations

import json
from pathlib import Path

from training.qwenvl_contrastive_smoke import explicit_negative_or_fallback
from training.siglip_smoke_data import load_pair_rows
from training.siglip_smoke_types import PairRow


def test_load_pair_rows_preserves_optional_negative_id(tmp_path: Path) -> None:
    # Given: a training manifest row with a generated failure image as hard negative.
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "ref_id": "source/ref",
                "tgt_id": "source/target",
                "neg_id": "generated/failure",
                "prompt": "single character reference control",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # When: the generic pair loader reads the row.
    rows = load_pair_rows(manifest, limit=8)

    # Then: the hard negative id stays available to the QwenVL trainer.
    assert rows == [
        PairRow(
            ref_id="source/ref",
            tgt_id="source/target",
            neg_id="generated/failure",
            prompt="single character reference control",
        )
    ]


def test_explicit_negative_or_fallback_uses_same_target_and_prompt() -> None:
    # Given: a positive row with an explicit generated-failure negative.
    row = PairRow(
        ref_id="source/ref",
        tgt_id="source/target",
        neg_id="generated/failure",
        prompt="single character reference control",
    )
    fallback = PairRow(
        ref_id="other/ref",
        tgt_id="other/target",
        prompt="fallback prompt",
    )

    # When: QwenVL contrastive training chooses the wrong-reference row.
    negative = explicit_negative_or_fallback(row, fallback)

    # Then: only the reference image is swapped; target and text stay aligned.
    assert negative == PairRow(
        ref_id="generated/failure",
        tgt_id="source/target",
        prompt="single character reference control",
    )


def test_explicit_negative_or_fallback_keeps_legacy_rows_unchanged() -> None:
    # Given: an older manifest row without a hard negative.
    row = PairRow(ref_id="source/ref", tgt_id="source/target", prompt="positive")
    fallback = PairRow(ref_id="other/ref", tgt_id="other/target", prompt="fallback")

    # When/Then: the existing deterministic fallback behavior is preserved.
    assert explicit_negative_or_fallback(row, fallback) is fallback
