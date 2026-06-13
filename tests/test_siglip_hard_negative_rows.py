from __future__ import annotations

from pathlib import Path

import pytest

from training.hard_negative_rows import explicit_negative_or_fallback
from training.pe_teacher_features import get_wrong_pe_features
from training.siglip_prepared_cache import get_wrong_prepared
from training.siglip_smoke_types import PairRow, SmokeConfig


def test_explicit_negative_or_fallback_swaps_only_reference() -> None:
    row = PairRow(
        ref_id="source/ref",
        tgt_id="source/target",
        neg_id="generated/collapse",
        prompt="green non-human character",
    )
    fallback = PairRow(ref_id="other/ref", tgt_id="other/target", prompt="fallback")

    negative = explicit_negative_or_fallback(row, fallback)

    assert negative == PairRow(
        ref_id="generated/collapse",
        tgt_id="source/target",
        prompt="green non-human character",
    )


def test_get_wrong_prepared_uses_explicit_negative_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        PairRow(
            ref_id="source/ref",
            tgt_id="source/target",
            neg_id="generated/collapse",
            prompt="green non-human character",
        ),
        PairRow(ref_id="other/ref", tgt_id="other/target", prompt="fallback"),
    ]
    captured: dict[str, PairRow] = {}
    sentinel = object()

    def fake_prepare_training_row(row: PairRow, *_args):
        captured["row"] = row
        return sentinel

    monkeypatch.setattr(
        "training.siglip_prepared_cache.prepare_training_row",
        fake_prepare_training_row,
    )

    result = get_wrong_prepared(
        cache=None,
        rows=rows,
        row_index=0,
        config=_config(tmp_path),
        vae=None,
        text_encoder=None,
        anima=None,
        siglip=None,
        processor=None,
        prepare_text_inputs=None,
        device=None,
        dtype=None,
    )

    assert result is sentinel
    assert captured["row"] == PairRow(
        ref_id="generated/collapse",
        tgt_id="source/target",
        prompt="green non-human character",
    )


def test_get_wrong_prepared_keeps_cached_fallback_for_legacy_rows(tmp_path: Path) -> None:
    rows = [
        PairRow(ref_id="source/ref", tgt_id="source/target", prompt="positive"),
        PairRow(ref_id="other/ref", tgt_id="other/target", prompt="fallback"),
    ]
    cached = [object(), object()]

    result = get_wrong_prepared(
        cache=cached,
        rows=rows,
        row_index=0,
        config=_config(tmp_path),
        vae=None,
        text_encoder=None,
        anima=None,
        siglip=None,
        processor=None,
        prepare_text_inputs=None,
        device=None,
        dtype=None,
    )

    assert result is cached[1]


def test_get_wrong_pe_features_uses_explicit_negative_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        PairRow("source/ref", "source/target", "green", neg_id="generated/collapse"),
        PairRow("other/ref", "other/target", "fallback"),
    ]
    captured: dict[str, PairRow] = {}
    sentinel = object()

    def fake_encode_pe_features(row: PairRow, *_args):
        captured["row"] = row
        return sentinel

    monkeypatch.setattr(
        "training.pe_teacher_features.encode_pe_features",
        fake_encode_pe_features,
    )

    result = get_wrong_pe_features(
        cache=None,
        rows=rows,
        row_index=0,
        config=_config(tmp_path),
        pe_encoder=None,
        encode_pe_from_imageminus1to1=None,
        device=None,
        dtype=None,
    )

    assert result is sentinel
    assert captured["row"] == PairRow("generated/collapse", "source/target", "green")


def test_get_wrong_pe_features_keeps_cached_fallback_for_legacy_rows(
    tmp_path: Path,
) -> None:
    rows = [
        PairRow(ref_id="source/ref", tgt_id="source/target", prompt="positive"),
        PairRow(ref_id="other/ref", tgt_id="other/target", prompt="fallback"),
    ]
    cached = [object(), object()]

    result = get_wrong_pe_features(
        cache=cached,
        rows=rows,
        row_index=0,
        config=_config(tmp_path),
        pe_encoder=None,
        encode_pe_from_imageminus1to1=None,
        device=None,
        dtype=None,
    )

    assert result is cached[1]


def _config(tmp_path: Path) -> SmokeConfig:
    return SmokeConfig(
        manifest_path=tmp_path / "manifest.jsonl",
        image_root=tmp_path,
        output_path=tmp_path / "out.safetensors",
        dit_path=tmp_path / "dit.safetensors",
        text_encoder_path=tmp_path / "text.safetensors",
        vae_path=tmp_path / "vae.safetensors",
        pe_checkpoint_path=tmp_path / "pe.safetensors",
        siglip_model_id="local",
        device="cpu",
        steps=1,
        resolution=256,
        lr=1e-5,
        seed=1,
        max_rows=2,
    )
