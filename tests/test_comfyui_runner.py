from __future__ import annotations

from pathlib import Path

from PIL import Image

from runner import GenerationOptions, RunnerPaths, build_command, newest_png


def test_build_command_resolves_relative_paths_against_anima_root(tmp_path: Path) -> None:
    anima_root = tmp_path / "anima_lora"
    output_dir = tmp_path / "outputs"
    reference = tmp_path / "ref.png"
    paths = RunnerPaths(
        python_executable=anima_root / ".venv/bin/python",
        anima_root=anima_root,
        dit=Path("models/diffusion_models/anima-base-v1.0.safetensors"),
        text_encoder=Path("models/text_encoders/qwen_3_06b_base.safetensors"),
        vae=Path("models/vae/qwen_image_vae.safetensors"),
        checkpoint=tmp_path / "adapter.safetensors",
        output_dir=output_dir,
    )
    options = GenerationOptions(
        prompt="panel layout reference",
        negative_prompt="low quality",
        seed=1234,
        height=960,
        width=1120,
        steps=20,
        guidance_scale=3.5,
        flow_shift=3.0,
        ip_scale=1.0,
        attn_mode="flash",
        match_reference_size=True,
    )

    command = build_command(paths, options, reference)

    assert command[0] == str(anima_root / ".venv/bin/python")
    assert command[1] == "inference.py"
    assert command[command.index("--dit") + 1] == str(
        anima_root / "models/diffusion_models/anima-base-v1.0.safetensors"
    )
    assert command[command.index("--ip_adapter_weight") + 1] == str(
        tmp_path / "adapter.safetensors"
    )
    assert command[command.index("--image_size") + 1 : command.index("--image_size") + 3] == [
        "960",
        "1120",
    ]
    assert "--ip_image_match_size" in command


def test_newest_png_ignores_older_outputs(tmp_path: Path) -> None:
    older = tmp_path / "older.png"
    newer = tmp_path / "newer.png"
    Image.new("RGB", (8, 8), "black").save(older)
    after_ns = older.stat().st_mtime_ns + 1
    Image.new("RGB", (8, 8), "white").save(newer)

    selected = newest_png(tmp_path, after_ns=after_ns)

    assert selected == newer
