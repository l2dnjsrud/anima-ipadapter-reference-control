from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from qwenvl_checkpoint import QwenVLCheckpointError, load_qwenvl_adapter
from qwenvl_model import IPAdapterQwenVL
from siglip_checkpoint import SigLIPCheckpointError, detect_siglip_checkpoint
from training.qwenvl_real_smoke import encode_qwenvl_embedding
from training.qwenvl_smoke_checkpoint import (
    load_trainable_qwenvl_adapter,
    save_qwenvl_adapter_checkpoint,
    verify_qwenvl_checkpoint,
)
from training.siglip_smoke_types import SmokeConfig


def test_encode_qwenvl_embedding_uses_instruction_and_returns_tensor(
    tmp_path: Path,
) -> None:
    """Given an image path, QwenVL encoding returns a normalized tensor contract."""

    class FakeSentenceModel:
        def encode(
            self,
            inputs: list[dict[str, object]],
            *,
            normalize_embeddings: bool,
            convert_to_tensor: bool,
            prompt: str,
        ) -> torch.Tensor:
            assert len(inputs) == 1
            assert isinstance(inputs[0]["image"], Image.Image)
            assert normalize_embeddings is True
            assert convert_to_tensor is True
            assert prompt == "anime color identity"
            return torch.tensor([[1.0, 2.0, 3.0]])

    image_path = tmp_path / "ref.jpg"
    Image.new("RGB", (32, 48), (20, 40, 80)).save(image_path)

    embedding = encode_qwenvl_embedding(
        FakeSentenceModel(),
        image_path,
        instruction="anime color identity",
        device=torch.device("cpu"),
    )

    assert embedding.shape == (1, 3)
    assert embedding.dtype == torch.float32
    assert embedding.device.type == "cpu"


def test_qwenvl_checkpoint_save_verify_round_trips(
    tmp_path: Path,
) -> None:
    """Given a trained QwenVL adapter, checkpoint helpers save and verify it."""

    output = tmp_path / "adapter.safetensors"
    adapter = _tiny_qwenvl_adapter()

    save_qwenvl_adapter_checkpoint(adapter, output)
    verification = verify_qwenvl_checkpoint(output, _pe_marker_checkpoint(tmp_path))
    loaded = load_qwenvl_adapter(output)

    assert verification.loadable is True
    assert verification.pe_checkpoint_rejected is True
    assert loaded.embedding_dim == 12
    with pytest.raises(SigLIPCheckpointError, match="QwenVL"):
        detect_siglip_checkpoint(loaded.state_dict())


def test_load_trainable_qwenvl_adapter_can_continue_checkpoint(
    tmp_path: Path,
) -> None:
    """Given an init checkpoint, QwenVL training reloads it with trainable params."""

    checkpoint = tmp_path / "init.safetensors"
    save_qwenvl_adapter_checkpoint(_tiny_qwenvl_adapter(), checkpoint)

    adapter = load_trainable_qwenvl_adapter(
        _smoke_config(tmp_path, init_checkpoint_path=checkpoint),
        torch.device("cpu"),
    )

    assert adapter.training is True
    assert all(parameter.requires_grad for parameter in adapter.parameters())


def test_verify_qwenvl_checkpoint_rejects_missing_file(tmp_path: Path) -> None:
    """Given a missing checkpoint path, verification fails loudly."""

    with pytest.raises(QwenVLCheckpointError):
        verify_qwenvl_checkpoint(
            tmp_path / "missing.safetensors",
            _pe_marker_checkpoint(tmp_path),
        )


def _tiny_qwenvl_adapter() -> IPAdapterQwenVL:
    return IPAdapterQwenVL(
        embedding_dim=12,
        dit_dim=16,
        num_blocks=2,
        num_queries=3,
        resampler_depth=1,
        resampler_heads=2,
        resampler_dim=16,
        resampler_dim_head=8,
        ip_heads=4,
        time_embed_dim=10,
    )


def _pe_marker_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "pe.safetensors"
    from safetensors.torch import save_file

    save_file({"ip_centroid": torch.zeros(1)}, str(path))
    return path


def _smoke_config(tmp_path: Path, *, init_checkpoint_path: Path | None) -> SmokeConfig:
    return SmokeConfig(
        manifest_path=tmp_path / "manifest.jsonl",
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
        init_checkpoint_path=init_checkpoint_path,
    )
