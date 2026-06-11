from __future__ import annotations

import torch

from training.pe_teacher_distillation import (
    predict_with_pe_teacher,
    teacher_distillation_loss,
)
from training.siglip_smoke_types import SmokeInputError


def test_teacher_distillation_loss_is_zero_when_predictions_match() -> None:
    student = torch.ones(1, 2, 2, 2)
    teacher = torch.ones_like(student)

    loss = teacher_distillation_loss(student, teacher)

    assert torch.equal(loss, torch.zeros_like(loss))


def test_teacher_distillation_loss_is_positive_when_predictions_differ() -> None:
    student = torch.ones(1, 1, 2, 2)
    teacher = torch.zeros_like(student)

    loss = teacher_distillation_loss(student, teacher)

    assert torch.allclose(loss, torch.tensor(1.0))


def test_teacher_distillation_loss_rejects_shape_mismatch() -> None:
    student = torch.ones(1, 1, 2, 2)
    teacher = torch.ones(1, 1, 4, 4)

    try:
        teacher_distillation_loss(student, teacher)
    except SmokeInputError as error:
        assert "teacher prediction must match student prediction shape" in str(error)
    else:
        raise AssertionError("shape mismatch should fail")


def test_predict_with_pe_teacher_restores_patched_attention() -> None:
    anima = _FakeAnima()
    network = _FakePENetwork(anima)
    noisy = torch.ones(1, 1, 1, 1)
    timesteps = torch.ones(1)
    crossattn_emb = torch.zeros(1, 1, 1, 1, 1)
    padding_mask = torch.zeros(1, 1, 1, 1)
    pe_features = torch.full((1, 2, 4), 2.0)

    prediction = predict_with_pe_teacher(
        anima=anima,
        network=network,
        pe_features=pe_features,
        noisy=noisy,
        timesteps=timesteps,
        crossattn_emb=crossattn_emb,
        padding_mask=padding_mask,
    )

    assert torch.allclose(prediction, torch.full_like(noisy, 17.0))
    assert network.calls == ["apply_to", "encode_ip_tokens", "set_ip_tokens", "clear_ip_tokens", "remove_from"]
    assert torch.allclose(
        anima.blocks[0].cross_attn.forward(noisy.unsqueeze(2), None, crossattn_emb),
        torch.full_like(noisy.unsqueeze(2), 2.0),
    )


class _FakeCrossAttention:
    def __init__(self) -> None:
        self.forward = self.original_forward

    def original_forward(
        self,
        x: torch.Tensor,
        _attn_params,
        _context: torch.Tensor,
        _rope_cos_sin=None,
    ) -> torch.Tensor:
        return x + 1.0


class _FakeBlock:
    def __init__(self) -> None:
        self.cross_attn = _FakeCrossAttention()


class _FakeAnima:
    def __init__(self) -> None:
        self.blocks = [_FakeBlock()]

    def __call__(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        *,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        del timesteps, padding_mask
        return self.blocks[0].cross_attn.forward(noisy, None, crossattn_emb)


class _FakePENetwork:
    def __init__(self, anima: _FakeAnima) -> None:
        self.anima = anima
        self.calls: list[str] = []
        self.ip_tokens: torch.Tensor | None = None
        self.original = anima.blocks[0].cross_attn.forward

    def apply_to(self, _text_encoders, _unet) -> None:
        self.calls.append("apply_to")
        self.original = self.anima.blocks[0].cross_attn.forward

        def patched_forward(
            x: torch.Tensor,
            _attn_params,
            _context: torch.Tensor,
            _rope_cos_sin=None,
        ) -> torch.Tensor:
            if self.ip_tokens is None:
                return x
            return x + self.ip_tokens.sum()

        self.anima.blocks[0].cross_attn.forward = patched_forward

    def encode_ip_tokens(self, pe_features: torch.Tensor) -> torch.Tensor:
        self.calls.append("encode_ip_tokens")
        return pe_features

    def set_ip_tokens(self, ip_tokens: torch.Tensor) -> None:
        self.calls.append("set_ip_tokens")
        self.ip_tokens = ip_tokens

    def clear_ip_tokens(self) -> None:
        self.calls.append("clear_ip_tokens")
        self.ip_tokens = None

    def remove_from(self) -> None:
        self.calls.append("remove_from")
        self.anima.blocks[0].cross_attn.forward = self.original
