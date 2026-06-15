from __future__ import annotations

import torch
import torch.nn.functional as F

from training.siglip_smoke_types import SmokeInputError


def denoised_latents_from_velocity(
    noisy_latents: torch.Tensor,
    velocity_prediction: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    _require_same_latent_shape(noisy_latents, velocity_prediction)
    sigma = _sigma_view(sigmas, noisy_latents)
    return noisy_latents.float() - sigma * velocity_prediction.float()


def latent_edge_projection_loss(
    predicted_latents: torch.Tensor,
    target_latents: torch.Tensor,
    *,
    reference_latents: torch.Tensor | None = None,
    reference_weight: float = 0.0,
) -> torch.Tensor:
    _require_same_latent_shape(predicted_latents, target_latents)
    base = _shape_distance(predicted_latents, target_latents)
    if reference_latents is None or reference_weight <= 0.0:
        return base
    _require_same_latent_shape(predicted_latents, reference_latents)
    return base + reference_weight * _shape_distance(predicted_latents, reference_latents)


def _shape_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_edge = _edge_map(left)
    right_edge = _edge_map(right)
    edge = F.mse_loss(_normalize_map(left_edge), _normalize_map(right_edge))
    left_h, left_w = _projections(left_edge)
    right_h, right_w = _projections(right_edge)
    height = F.mse_loss(_normalize_vector(left_h), _normalize_vector(right_h))
    width = F.mse_loss(_normalize_vector(left_w), _normalize_vector(right_w))
    return edge + 0.5 * (height + width)


def _edge_map(latents: torch.Tensor) -> torch.Tensor:
    gray = latents.float().mean(dim=1, keepdim=True)
    kernel_x = gray.new_tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    kernel_y = gray.new_tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    grad_x = F.conv2d(gray, kernel_x.view(1, 1, 3, 3), padding=1)
    grad_y = F.conv2d(gray, kernel_y.view(1, 1, 3, 3), padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)


def _projections(edge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return edge.mean(dim=-1), edge.mean(dim=-2)


def _normalize_map(value: torch.Tensor) -> torch.Tensor:
    return _normalize_vector(value.flatten(start_dim=1)).view_as(value)


def _normalize_vector(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=1, keepdim=True).clamp_min(1e-6)


def _sigma_view(sigmas: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    if sigmas.ndim == 0:
        return sigmas.reshape(1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
    if sigmas.ndim == 1:
        return sigmas.reshape(-1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
    if sigmas.ndim != 4:
        raise SmokeInputError("sigmas must be scalar, 1D, or 4D broadcastable tensor")
    if not _is_broadcastable_to_latents(sigmas, latents):
        raise SmokeInputError(
            f"sigmas are not broadcastable to latents: {tuple(sigmas.shape)} != {tuple(latents.shape)}"
        )
    return sigmas.to(device=latents.device, dtype=latents.dtype)


def _is_broadcastable_to_latents(sigmas: torch.Tensor, latents: torch.Tensor) -> bool:
    return all(left in (1, right) for left, right in zip(sigmas.shape, latents.shape))


def _require_same_latent_shape(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.ndim != 4 or right.ndim != 4:
        raise SmokeInputError("latents must be 4D tensors shaped [B, C, H, W]")
    if left.shape != right.shape:
        raise SmokeInputError(f"latent shapes must match: {tuple(left.shape)} != {tuple(right.shape)}")
