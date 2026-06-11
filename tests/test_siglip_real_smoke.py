from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import save_file

from siglip_checkpoint import SigLIPCheckpointError, load_siglip_adapter
from siglip_model import IPAdapterSigLIP
from training.siglip_real_smoke import (
    freeze_module,
    load_trainable_adapter,
    PREPARED_ROW_CACHE_LIMIT,
    save_adapter_checkpoint,
    trainable_parameter_count,
    verify_checkpoint,
)
from training.siglip_smoke_data import load_pair_rows, resolve_pair_paths
from training.siglip_smoke_patch import patched_cross_attention
from training.siglip_smoke_runtime import validate_config
from training.siglip_smoke_types import (
    MAX_PILOT_ROWS,
    MAX_PILOT_STEPS,
    CheckpointVerification,
    SmokeConfig,
    SmokeInputError,
    SmokeSummary,
)


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


def test_load_trainable_adapter_can_continue_from_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "siglip.safetensors"
    save_adapter_checkpoint(_tiny_adapter(), checkpoint)
    config = _config(tmp_path, steps=1, max_rows=1)
    config = replace(config, init_checkpoint_path=checkpoint)

    adapter = load_trainable_adapter(config, torch.device("cpu"), torch.bfloat16)
    first_param = next(adapter.parameters())

    assert adapter.training is True
    assert trainable_parameter_count(adapter) > 0
    assert first_param.dtype is torch.float32


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


def test_validate_config_accepts_bounded_pilot_limits(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=16, max_rows=128)

    validate_config(config)

    assert config.steps > 8
    assert config.max_rows > 64
    assert PREPARED_ROW_CACHE_LIMIT >= 128


def test_validate_config_rejects_unbounded_pilot_steps(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=MAX_PILOT_STEPS + 1, max_rows=4)

    try:
        validate_config(config)
    except SmokeInputError as error:
        assert f"steps must be <= {MAX_PILOT_STEPS}" in str(error)
    else:
        raise AssertionError("unbounded pilot steps should fail")


def test_validate_config_rejects_unbounded_pilot_rows(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=1, max_rows=MAX_PILOT_ROWS + 1)

    try:
        validate_config(config)
    except SmokeInputError as error:
        assert f"max_rows must be <= {MAX_PILOT_ROWS}" in str(error)
    else:
        raise AssertionError("unbounded pilot rows should fail")


def test_validate_config_rejects_missing_init_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path, steps=1, max_rows=1)
    config = replace(config, init_checkpoint_path=tmp_path / "missing.safetensors")

    try:
        validate_config(config)
    except SmokeInputError as error:
        assert "init checkpoint not found" in str(error)
    else:
        raise AssertionError("missing init checkpoint should fail")


def test_smoke_summary_records_deterministic_loss_history() -> None:
    summary = SmokeSummary(
        steps=3,
        rows_loaded=3,
        first_loss=0.3,
        final_loss=0.1,
        mean_loss=0.2,
        finite_loss=True,
        loss_history=(0.3, 0.2, 0.1),
        trainable_parameters=10,
        frozen_base_parameters=20,
        checkpoint=CheckpointVerification(
            output_path="checkpoints/example.safetensors",
            loadable=True,
            pe_checkpoint_rejected=True,
        ),
    )

    assert summary.loss_history == (0.3, 0.2, 0.1)
    expected_mean = sum(summary.loss_history) / len(summary.loss_history)
    assert abs(summary.mean_loss - expected_mean) < 1e-9


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


def _config(tmp_path: Path, *, steps: int, max_rows: int) -> SmokeConfig:
    manifest = tmp_path / "pairs.jsonl"
    image_root = tmp_path / "images"
    image_root.mkdir()
    dit_path = tmp_path / "dit.safetensors"
    text_path = tmp_path / "text.safetensors"
    vae_path = tmp_path / "vae.safetensors"
    for path in (manifest, dit_path, text_path, vae_path):
        path.write_text("", encoding="utf-8")
    return SmokeConfig(
        manifest_path=manifest,
        image_root=image_root,
        output_path=tmp_path / "out.safetensors",
        dit_path=dit_path,
        text_encoder_path=text_path,
        vae_path=vae_path,
        pe_checkpoint_path=tmp_path / "pe.safetensors",
        siglip_model_id="google/siglip2-base-patch16-512",
        device="cpu",
        steps=steps,
        resolution=256,
        lr=1e-5,
        seed=20260610,
        max_rows=max_rows,
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
