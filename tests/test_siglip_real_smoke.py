from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from siglip_checkpoint import SigLIPCheckpointError, load_siglip_adapter
from siglip_model import IPAdapterSigLIP
from training.siglip_real_smoke import (
    freeze_module,
    save_adapter_checkpoint,
    trainable_parameter_count,
    verify_checkpoint,
)
from training.siglip_smoke_data import load_pair_rows, resolve_pair_paths
from training.siglip_smoke_patch import patched_cross_attention
from training.siglip_smoke_types import SmokeInputError


def test_load_pair_rows_parses_limited_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"ref_id": "a", "tgt_id": "b", "prompt": "caption b"}),
                json.dumps({"ref_id": "c", "tgt_id": "d", "prompt": "caption d"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_pair_rows(manifest, limit=1)

    assert len(rows) == 1
    assert rows[0].ref_id == "a"
    assert rows[0].tgt_id == "b"
    assert rows[0].prompt == "caption b"


def test_resolve_pair_paths_rejects_missing_files(tmp_path: Path) -> None:
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        json.dumps({"ref_id": "a", "tgt_id": "b", "prompt": "caption b"}) + "\n",
        encoding="utf-8",
    )
    row = load_pair_rows(manifest, limit=1)[0]

    try:
        resolve_pair_paths(row, tmp_path)
    except SmokeInputError as error:
        assert "missing pair file" in str(error)
    else:
        raise AssertionError("missing files should fail")


def test_freeze_module_leaves_only_adapter_trainable() -> None:
    base = torch.nn.Linear(2, 2)
    adapter = _tiny_adapter()

    frozen_count = freeze_module(base)

    assert frozen_count == 6
    assert trainable_parameter_count(base) == 0
    assert trainable_parameter_count(adapter) > 0


def test_checkpoint_roundtrip_and_pe_rejection(tmp_path: Path) -> None:
    output = tmp_path / "siglip.safetensors"
    pe_path = tmp_path / "pe.safetensors"
    save_adapter_checkpoint(_tiny_adapter(), output)
    save_file({"ip_centroid": torch.zeros(1)}, str(pe_path))

    verification = verify_checkpoint(output, pe_path)

    assert verification.loadable is True
    assert verification.pe_checkpoint_rejected is True
    assert load_siglip_adapter(output).num_blocks == 2
    try:
        load_siglip_adapter(pe_path)
    except SigLIPCheckpointError:
        pass
    else:
        raise AssertionError("PE checkpoint should be rejected by SigLIP loader")


def test_patched_cross_attention_adds_adapter_block_output() -> None:
    anima = _FakeAnima()
    adapter = _FakeAdapter()
    query = torch.ones(1, 2, 4)
    context = torch.zeros(1, 1, 4)
    image_tokens = torch.ones(1, 2, 4)

    before = anima.blocks[1].cross_attn.forward(query, None, context)
    with patched_cross_attention(anima, adapter, image_tokens, weight=0.5):
        patched = anima.blocks[1].cross_attn.forward(query, None, context)
    after = anima.blocks[1].cross_attn.forward(query, None, context)

    assert torch.allclose(before, torch.full_like(query, 2.0))
    assert torch.allclose(patched, torch.full_like(query, 5.0))
    assert torch.allclose(after, before)


def _tiny_adapter() -> IPAdapterSigLIP:
    return IPAdapterSigLIP(
        siglip_dim=4,
        siglip_shallow_dim=4,
        dit_dim=8,
        num_blocks=2,
        num_queries=2,
        resampler_depth=1,
        resampler_heads=2,
        resampler_dim=8,
        resampler_dim_head=4,
        intermediate_dim=4,
        intermediate_layers=1,
        intermediate_heads=2,
        ip_heads=2,
        time_embed_dim=6,
        use_intermediate_encoder=True,
    )


class _FakeCrossAttn:
    def __init__(self, value: float) -> None:
        self.value = value

    def forward(
        self,
        x: torch.Tensor,
        attn_params,
        context: torch.Tensor,
        rope_cos_sin=None,
    ) -> torch.Tensor:
        del attn_params, context, rope_cos_sin
        return torch.full_like(x, self.value)


class _FakeBlock:
    def __init__(self, value: float) -> None:
        self.cross_attn = _FakeCrossAttn(value)


class _FakeAnima:
    def __init__(self) -> None:
        self.blocks = [_FakeBlock(1.0), _FakeBlock(2.0)]


class _FakeAdapter(IPAdapterSigLIP):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.num_blocks = 2

    def forward_block(
        self,
        block_idx: int,
        query: torch.Tensor,
        image_tokens: torch.Tensor,
        weight: float = 1.0,
    ) -> torch.Tensor:
        del query
        return image_tokens * float(block_idx + 1) * weight * 3.0
