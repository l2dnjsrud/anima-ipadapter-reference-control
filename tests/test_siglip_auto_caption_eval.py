from __future__ import annotations

import json
from pathlib import Path

from tools.siglip_auto_caption_eval import (
    AutoPromptRow,
    Sample,
    Variant,
    adapter_prompt,
    load_auto_prompt_rows,
    no_ip_prompt,
)


def test_load_auto_prompt_rows_assigns_stable_labels_and_seeds(tmp_path: Path) -> None:
    manifest_path = tmp_path / "prompts.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "ref_id": "SG-001/portrait",
                "tgt_id": "SG-001/portrait",
                "source_prompt": "source",
                "prompt": "auto prompt",
                "selected_attributes": ["young scholar with glasses"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_auto_prompt_rows(manifest_path)

    assert rows == (
        Sample(
            label="auto00",
            ref_id="SG-001/portrait",
            seed=20260650,
            prompt_row=AutoPromptRow(
                ref_id="SG-001/portrait",
                tgt_id="SG-001/portrait",
                source_prompt="source",
                prompt="auto prompt",
                selected_attributes=("young scholar with glasses",),
            ),
        ),
    )


def test_adapter_prompt_patches_model_path_to_cfg_and_scheduler() -> None:
    sample = Sample(
        label="auto00",
        ref_id="SG-001/portrait",
        seed=20260650,
        prompt_row=AutoPromptRow(
            ref_id="SG-001/portrait",
            tgt_id="SG-001/portrait",
            source_prompt="source",
            prompt="auto prompt",
            selected_attributes=("young scholar with glasses",),
        ),
    )
    variant = Variant("siglip_pe_space_w14", "adapter.safetensors", 1.4)

    prompt = adapter_prompt(sample, "reference.jpg", variant, output_prefix="eval/test")

    assert prompt["2"]["inputs"]["ipadapter_name"] == "adapter.safetensors"
    assert prompt["3"]["class_type"] == "AnimaSigLIPEncodeImage"
    assert prompt["4"]["class_type"] == "AnimaSigLIPIPAdapterApply"
    assert prompt["10"]["inputs"]["model"] == ["4", 0]
    assert prompt["12"]["inputs"]["model"] == ["4", 0]
    assert prompt["7"]["inputs"]["clip_l"] == "auto prompt"


def test_no_ip_prompt_keeps_sampler_on_base_model() -> None:
    sample = Sample(
        label="auto00",
        ref_id="SG-001/portrait",
        seed=20260650,
        prompt_row=AutoPromptRow(
            ref_id="SG-001/portrait",
            tgt_id="SG-001/portrait",
            source_prompt="source",
            prompt="auto prompt",
            selected_attributes=(),
        ),
    )

    prompt = no_ip_prompt(sample, Variant("no_ip", None, 0.0), output_prefix="eval/test")

    assert prompt["10"]["inputs"]["model"] == ["18", 0]
    assert prompt["12"]["inputs"]["model"] == ["18", 0]
    assert prompt["17"]["inputs"]["filename_prefix"] == "eval/test/auto00_no_ip"
