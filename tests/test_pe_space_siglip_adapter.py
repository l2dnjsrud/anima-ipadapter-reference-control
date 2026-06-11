from __future__ import annotations

import torch

from training.pe_space_siglip_adapter import build_pe_space_siglip_adapter
from siglip_model import IPAdapterSigLIP


def test_build_pe_space_siglip_adapter_copies_compatible_base_and_pe_kv() -> None:
    base = _tiny_adapter(ip_hidden_dim=16)
    teacher = _FakePENetwork()

    adapter = build_pe_space_siglip_adapter(base, teacher)

    assert adapter.resampler.proj_out.weight.shape == (8, 16)
    assert torch.allclose(
        adapter.intermediate_encoder.shallow_proj.weight,
        base.intermediate_encoder.shallow_proj.weight,
    )
    assert torch.allclose(adapter.ip_cross_attns[0].to_k_ip.weight, teacher.to_k_ip[0].weight)
    assert torch.allclose(adapter.ip_cross_attns[1].to_v_ip.weight, teacher.to_v_ip[1].weight)
    assert torch.allclose(adapter.ip_scales[0], torch.tensor([0.5]))
    assert torch.allclose(adapter.ip_scales[1], torch.tensor([-0.25]))


def _tiny_adapter(*, ip_hidden_dim: int) -> IPAdapterSigLIP:
    return IPAdapterSigLIP(
        siglip_dim=8,
        siglip_shallow_dim=8,
        dit_dim=16,
        ip_hidden_dim=ip_hidden_dim,
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
        use_intermediate_encoder=True,
    )


class _FakePENetwork:
    def __init__(self) -> None:
        self.context_dim = 8
        self.num_blocks = 2
        self.to_k_ip = torch.nn.ModuleList([torch.nn.Linear(8, 16, bias=False) for _ in range(2)])
        self.to_v_ip = torch.nn.ModuleList([torch.nn.Linear(8, 16, bias=False) for _ in range(2)])
        self.ip_gate = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.5)), torch.nn.Parameter(torch.tensor(-0.25))]
        )
