from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.qwenvl_contrastive_smoke import run_qwenvl_contrastive_smoke
from training.siglip_smoke_types import SmokeConfig, SmokeInputError


def test_qwenvl_contrastive_smoke_rejects_single_row(tmp_path: Path) -> None:
    """Given one row, contrastive QwenVL training fails before heavy model load."""

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"ref_id": "a", "tgt_id": "b", "prompt": "comic panel"}) + "\n",
        encoding="utf-8",
    )
    for name in ("a.jpg", "b.jpg", "b.txt", "dit.safetensors", "text.safetensors", "vae.safetensors"):
        (tmp_path / name).write_text("", encoding="utf-8")

    with pytest.raises(SmokeInputError, match="at least two rows"):
        run_qwenvl_contrastive_smoke(
            SmokeConfig(
                manifest_path=manifest,
                image_root=tmp_path,
                output_path=tmp_path / "out.safetensors",
                dit_path=tmp_path / "dit.safetensors",
                text_encoder_path=tmp_path / "text.safetensors",
                vae_path=tmp_path / "vae.safetensors",
                pe_checkpoint_path=tmp_path / "pe.safetensors",
                siglip_model_id="Qwen/Qwen3-VL-Embedding-2B",
                device="cpu",
                steps=1,
                resolution=128,
                lr=1e-5,
                seed=1,
                max_rows=1,
            ),
            contrastive_weight=0.25,
            contrastive_margin=0.05,
        )
