from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence


SUPPORTED_ATTN_MODES: Final[tuple[str, ...]] = (
    "flash",
    "torch",
    "sageattn",
    "flex",
    "xformers",
)


@dataclass(frozen=True, slots=True)
class RunnerPaths:
    python_executable: Path
    anima_root: Path
    dit: Path
    text_encoder: Path
    vae: Path
    checkpoint: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    prompt: str
    negative_prompt: str
    seed: int
    height: int
    width: int
    steps: int
    guidance_scale: float
    flow_shift: float
    ip_scale: float
    attn_mode: str
    match_reference_size: bool


@dataclass(frozen=True, slots=True)
class RunnerConfigError(RuntimeError):
    label: str
    path: Path

    def __str__(self) -> str:
        return f"{self.label} not found: {self.path}"


@dataclass(frozen=True, slots=True)
class UnsupportedAttnModeError(RuntimeError):
    attn_mode: str

    def __str__(self) -> str:
        modes = ", ".join(SUPPORTED_ATTN_MODES)
        return f"Unsupported attention mode {self.attn_mode!r}; expected one of: {modes}"


@dataclass(frozen=True, slots=True)
class OutputImageNotFoundError(RuntimeError):
    output_dir: Path
    after_ns: int

    def __str__(self) -> str:
        return f"No PNG output found in {self.output_dir} after timestamp {self.after_ns}"


@dataclass(frozen=True, slots=True)
class CommandFailedError(RuntimeError):
    returncode: int
    stdout_tail: str
    stderr_tail: str

    def __str__(self) -> str:
        return (
            f"Anima inference failed with exit code {self.returncode}\n"
            f"stdout tail:\n{self.stdout_tail}\n"
            f"stderr tail:\n{self.stderr_tail}"
        )


def resolve_against(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def resolved_paths(paths: RunnerPaths) -> RunnerPaths:
    return RunnerPaths(
        python_executable=resolve_against(paths.python_executable.expanduser(), paths.anima_root),
        anima_root=paths.anima_root.expanduser(),
        dit=resolve_against(paths.dit.expanduser(), paths.anima_root),
        text_encoder=resolve_against(paths.text_encoder.expanduser(), paths.anima_root),
        vae=resolve_against(paths.vae.expanduser(), paths.anima_root),
        checkpoint=paths.checkpoint.expanduser(),
        output_dir=paths.output_dir.expanduser(),
    )


def require_runner_inputs(paths: RunnerPaths, reference_image: Path) -> None:
    checks = (
        ("Python executable", paths.python_executable),
        ("Anima root", paths.anima_root),
        ("DiT model", paths.dit),
        ("Text encoder", paths.text_encoder),
        ("VAE", paths.vae),
        ("IP-Adapter checkpoint", paths.checkpoint),
        ("Reference image", reference_image),
    )
    for label, path in checks:
        if not path.exists():
            raise RunnerConfigError(label=label, path=path)


def build_command(
    paths: RunnerPaths,
    options: GenerationOptions,
    reference_image: Path,
) -> list[str]:
    if options.attn_mode not in SUPPORTED_ATTN_MODES:
        raise UnsupportedAttnModeError(attn_mode=options.attn_mode)

    resolved = resolved_paths(paths)
    command = [
        str(resolved.python_executable),
        "inference.py",
        "--dit",
        str(resolved.dit),
        "--text_encoder",
        str(resolved.text_encoder),
        "--vae",
        str(resolved.vae),
        "--vae_chunk_size",
        "64",
        "--vae_disable_cache",
        "--attn_mode",
        options.attn_mode,
        "--prompt",
        options.prompt,
        "--negative_prompt",
        options.negative_prompt,
        "--seed",
        str(options.seed),
        "--infer_steps",
        str(options.steps),
        "--guidance_scale",
        str(options.guidance_scale),
        "--flow_shift",
        str(options.flow_shift),
        "--image_size",
        str(options.height),
        str(options.width),
        "--save_path",
        str(resolved.output_dir),
        "--ip_adapter_weight",
        str(resolved.checkpoint),
        "--ip_image",
        str(reference_image),
        "--ip_scale",
        str(options.ip_scale),
    ]
    if options.match_reference_size:
        command.append("--ip_image_match_size")
    return command


def build_subprocess_env(anima_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    root_text = str(anima_root)
    env["PYTHONPATH"] = (
        root_text
        if current_pythonpath is None or current_pythonpath == ""
        else f"{root_text}{os.pathsep}{current_pythonpath}"
    )
    return env


def run_command(command: Sequence[str], cwd: Path, env: Mapping[str, str]) -> None:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise CommandFailedError(
            returncode=result.returncode,
            stdout_tail=result.stdout[-4000:],
            stderr_tail=result.stderr[-4000:],
        )


def newest_png(output_dir: Path, after_ns: int) -> Path:
    candidates = [
        path
        for path in output_dir.glob("*.png")
        if path.is_file() and path.stat().st_mtime_ns >= after_ns
    ]
    if not candidates:
        raise OutputImageNotFoundError(output_dir=output_dir, after_ns=after_ns)
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)
