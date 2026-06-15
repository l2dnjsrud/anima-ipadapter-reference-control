from __future__ import annotations

from pathlib import Path

from tools import siglip_auto_caption_artifacts
from tools.siglip_auto_caption_artifacts import copy_output_image, copy_reference
from tools.siglip_auto_caption_types import AutoPromptRow, EvalConfig, JsonValue, Sample


def _sample() -> Sample:
    return Sample(
        label="auto00",
        ref_id="SG-001/portrait",
        seed=20260650,
        prompt_row=AutoPromptRow(
            ref_id="SG-001/portrait",
            tgt_id="SG-001/portrait",
            source_prompt="source",
            prompt="prompt",
            selected_attributes=(),
        ),
    )


def test_copy_reference_uploads_when_comfy_input_is_not_writable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "dataset"
    image_path = data_root / "SG-001" / "portrait.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"jpeg")
    calls: list[tuple[str, Path, str]] = []

    def fake_upload(base_url: str, source: Path, image_name: str) -> str:
        calls.append((base_url, source, image_name))
        return "uploaded/reference.jpg"

    monkeypatch.setattr(
        siglip_auto_caption_artifacts,
        "_can_prepare_writable_directory",
        lambda _path: False,
        raising=False,
    )
    monkeypatch.setattr(
        siglip_auto_caption_artifacts,
        "upload_image",
        fake_upload,
        raising=False,
    )
    config = EvalConfig(
        data_root=data_root,
        base_url="http://127.0.0.1:8102",
        out_dir=tmp_path / "eval",
        comfy_input=tmp_path / "locked_input",
    )

    image_name = copy_reference(_sample(), config)

    assert image_name == "uploaded/reference.jpg"
    assert calls == [
        (
            "http://127.0.0.1:8102",
            image_path,
            "eval_auto00.jpg",
        )
    ]


def test_copy_output_image_downloads_when_output_file_is_not_readable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, str, str, str]] = []

    def fake_view(
        base_url: str,
        *,
        filename: str,
        subfolder: str,
        image_type: str,
    ) -> bytes:
        calls.append((base_url, filename, subfolder, image_type))
        return b"png"

    monkeypatch.setattr(
        siglip_auto_caption_artifacts,
        "view_image_bytes",
        fake_view,
        raising=False,
    )
    config = EvalConfig(
        base_url="http://127.0.0.1:8102",
        out_dir=tmp_path / "eval",
        comfy_output=tmp_path / "missing_output",
    )
    config.out_dir.mkdir()
    image_info: dict[str, JsonValue] = {
        "filename": "ComfyUI_00001_.png",
        "subfolder": "",
        "type": "output",
    }

    image_path = copy_output_image(image_info, "auto00_no_ip", config)

    assert image_path.read_bytes() == b"png"
    assert calls == [
        ("http://127.0.0.1:8102", "ComfyUI_00001_.png", "", "output")
    ]
