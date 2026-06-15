from __future__ import annotations

from pathlib import Path

import torch

from training.siglip_shape_cache import get_reference_latents
from training.siglip_smoke_types import PairRow, SmokeConfig


def test_get_reference_latents_encodes_reference_image(
    tmp_path: Path,
    monkeypatch,
) -> None:
    row = PairRow("ref/item", "target/item", "prompt")
    _touch_pair_files(tmp_path, row)
    captured: dict[str, Path] = {}
    sentinel = torch.zeros(1, 4, 8, 8)

    def fake_encode_target_latents(_vae, path: Path, *_args):
        captured["path"] = path
        return sentinel

    monkeypatch.setattr(
        "training.siglip_shape_cache.encode_target_latents",
        fake_encode_target_latents,
    )

    result = get_reference_latents(
        None,
        [row],
        0,
        _config(tmp_path),
        vae=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(result, sentinel)
    assert captured["path"] == tmp_path / "ref/item.jpg"


def test_get_reference_latents_uses_cache_when_available(tmp_path: Path) -> None:
    row = PairRow("ref/item", "target/item", "prompt")
    cached = [torch.ones(1, 4, 8, 8)]

    result = get_reference_latents(
        cached,
        [row],
        0,
        _config(tmp_path),
        vae=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert result is cached[0]


def _touch_pair_files(root: Path, row: PairRow) -> None:
    for relative in (f"{row.ref_id}.jpg", f"{row.tgt_id}.jpg", f"{row.tgt_id}.txt"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")


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
        max_rows=1,
    )
