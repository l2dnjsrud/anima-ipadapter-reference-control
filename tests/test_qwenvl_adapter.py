from __future__ import annotations

import pytest
import torch

from qwenvl_checkpoint import (
    QwenVLCheckpointError,
    build_qwenvl_adapter_from_state,
    detect_qwenvl_checkpoint,
)
from qwenvl_model import IPAdapterQwenVL
from siglip_checkpoint import SigLIPCheckpointError, detect_siglip_checkpoint
from siglip_model import IPAdapterSigLIP


def test_qwenvl_embedding_adapter_shapes_when_embedding_is_present() -> None:
    """Given QwenVL embeddings, adapter tokens and block output have Anima shape."""

    adapter = _tiny_qwenvl_adapter()
    embedding = torch.randn(2, 12)
    timestep = torch.tensor([0.25, 0.75])

    image_tokens = adapter.encode_ref(embedding, timestep=timestep)
    out = adapter.forward_block(1, torch.randn(2, 4, 16), image_tokens)

    assert image_tokens.shape == (2, 3, 16)
    assert out.shape == (2, 4, 16)


def test_qwenvl_checkpoint_detection_round_trips_tiny_state() -> None:
    """Given a QwenVL state dict, detection reconstructs the same architecture."""

    state = _tiny_qwenvl_adapter().state_dict()

    spec = detect_qwenvl_checkpoint(state)
    loaded = build_qwenvl_adapter_from_state(state)

    assert spec.embedding_dim == 12
    assert spec.num_blocks == 2
    assert spec.num_queries == 3
    assert loaded.encode_ref(torch.randn(1, 12), timestep=torch.tensor([0.5])).shape == (
        1,
        3,
        16,
    )


def test_qwenvl_checkpoint_detection_rejects_siglip_state() -> None:
    """Given a SigLIP checkpoint, QwenVL detection must fail loudly."""

    state = _tiny_siglip_adapter().state_dict()

    with pytest.raises(QwenVLCheckpointError, match="QwenVL family marker"):
        detect_qwenvl_checkpoint(state)


def test_qwenvl_checkpoint_detection_rejects_pe_core_state() -> None:
    """Given a PE-Core checkpoint marker, QwenVL detection must fail loudly."""

    with pytest.raises(QwenVLCheckpointError, match="PE-Core"):
        detect_qwenvl_checkpoint({"ip_centroid": torch.zeros(1)})


def test_siglip_detector_rejects_qwenvl_family_marker() -> None:
    """Given a QwenVL-marked state, SigLIP detection must fail loudly."""

    state = _tiny_qwenvl_adapter().state_dict()

    with pytest.raises(SigLIPCheckpointError, match="QwenVL"):
        detect_siglip_checkpoint(state)


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


def _tiny_siglip_adapter() -> IPAdapterSigLIP:
    return IPAdapterSigLIP(
        siglip_dim=8,
        siglip_shallow_dim=8,
        dit_dim=16,
        num_blocks=2,
        num_queries=3,
        resampler_depth=1,
        resampler_heads=2,
        resampler_dim=16,
        resampler_dim_head=8,
        intermediate_dim=8,
        intermediate_layers=1,
        intermediate_heads=2,
        ip_heads=4,
        time_embed_dim=10,
        use_intermediate_encoder=False,
    )
