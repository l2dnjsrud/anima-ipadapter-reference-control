from __future__ import annotations

import pytest
import torch

from training.latent_shape_loss import (
    denoised_latents_from_velocity,
    latent_edge_projection_loss,
)
from training.siglip_smoke_types import SmokeInputError


def test_denoised_latents_from_velocity_recovers_clean_latents() -> None:
    clean = torch.tensor([[[[0.2, -0.4], [0.6, -0.8]]]])
    noise = torch.tensor([[[[0.9, 0.1], [-0.3, 0.5]]]])
    sigma = torch.tensor([0.25])
    noisy = (1.0 - sigma.view(1, 1, 1, 1)) * clean + sigma.view(1, 1, 1, 1) * noise
    velocity = noise - clean

    denoised = denoised_latents_from_velocity(noisy, velocity, sigma)

    assert torch.allclose(denoised, clean)


def test_denoised_latents_accepts_scheduler_sigma_broadcast_shape() -> None:
    noisy = torch.ones(1, 4, 8, 8)
    velocity = torch.full_like(noisy, 0.5)
    sigmas = torch.full((1, 1, 1, 1), 0.25)

    denoised = denoised_latents_from_velocity(noisy, velocity, sigmas)

    assert torch.allclose(denoised, torch.full_like(noisy, 0.875))


def test_latent_edge_projection_loss_is_zero_for_identical_latents() -> None:
    latents = torch.zeros(1, 4, 8, 8)
    latents[:, :, 2:6, 2:6] = 1.0

    loss = latent_edge_projection_loss(latents, latents)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_latent_edge_projection_loss_penalizes_shifted_shape() -> None:
    target = torch.zeros(1, 4, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    shifted = torch.zeros_like(target)
    shifted[:, :, 1:5, 1:5] = 1.0

    loss = latent_edge_projection_loss(shifted, target)

    assert float(loss) > 0.01


def test_latent_edge_projection_loss_includes_reference_weight() -> None:
    predicted = torch.zeros(1, 4, 8, 8)
    target = torch.zeros_like(predicted)
    reference = torch.zeros_like(predicted)
    predicted[:, :, 2:6, 2:6] = 1.0
    target[:, :, 2:6, 2:6] = 1.0
    reference[:, :, 1:5, 1:5] = 1.0

    without_reference = latent_edge_projection_loss(predicted, target)
    with_reference = latent_edge_projection_loss(
        predicted,
        target,
        reference_latents=reference,
        reference_weight=1.0,
    )

    assert torch.allclose(without_reference, torch.tensor(0.0), atol=1e-6)
    assert float(with_reference) > float(without_reference)


def test_latent_edge_projection_loss_rejects_shape_mismatch() -> None:
    predicted = torch.zeros(1, 4, 8, 8)
    target = torch.zeros(1, 4, 7, 8)

    with pytest.raises(SmokeInputError, match="must match"):
        latent_edge_projection_loss(predicted, target)
