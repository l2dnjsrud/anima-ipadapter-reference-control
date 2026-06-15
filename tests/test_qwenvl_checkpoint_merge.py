from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools.merge_qwenvl_checkpoints import (
    CheckpointMergeError,
    merge_checkpoint_files,
    merge_checkpoint_states,
)


def test_merge_checkpoint_states_interpolates_float_tensors() -> None:
    base = {
        "float": torch.tensor([1.0, 3.0]),
        "metadata": torch.tensor([7], dtype=torch.int64),
    }
    update = {
        "float": torch.tensor([5.0, 11.0]),
        "metadata": torch.tensor([7], dtype=torch.int64),
    }

    merged, summary = merge_checkpoint_states(base, update, alpha=0.25)

    assert torch.allclose(merged["float"], torch.tensor([2.0, 5.0]))
    assert torch.equal(merged["metadata"], base["metadata"])
    assert summary.float_tensor_count == 1
    assert summary.skipped_non_float_count == 1


def test_merge_checkpoint_states_rejects_key_mismatch() -> None:
    base = {"only_base": torch.tensor([1.0])}
    update = {"only_update": torch.tensor([1.0])}

    with pytest.raises(CheckpointMergeError, match="key mismatch"):
        merge_checkpoint_states(base, update, alpha=0.5)


def test_merge_checkpoint_states_rejects_shape_mismatch() -> None:
    base = {"weight": torch.zeros(1, 2)}
    update = {"weight": torch.zeros(2, 1)}

    with pytest.raises(CheckpointMergeError, match="shape mismatch"):
        merge_checkpoint_states(base, update, alpha=0.5)


def test_merge_checkpoint_files_writes_checkpoint_and_summary(tmp_path: Path) -> None:
    base_path = tmp_path / "base.safetensors"
    update_path = tmp_path / "update.safetensors"
    output_dir = tmp_path / "out"
    summary_path = tmp_path / "summary.json"
    save_file({"weight": torch.tensor([0.0, 2.0])}, str(base_path))
    save_file({"weight": torch.tensor([10.0, 12.0])}, str(update_path))

    summary = merge_checkpoint_files(
        base_checkpoint_path=base_path,
        update_checkpoint_path=update_path,
        output_dir=output_dir,
        output_prefix="merged",
        alphas=(0.4,),
        summary_path=summary_path,
    )

    assert summary_path.is_file()
    assert len(summary.outputs) == 1
    output_path = Path(summary.outputs[0].output_path)
    assert output_path.is_file()
    merged = load_file(str(output_path))
    assert torch.allclose(merged["weight"], torch.tensor([4.0, 6.0]))
