from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    "anima_ipadapter_reference_generate.json",
    "anima_ipadapter_contactsheet_ref03_ersde.json",
)


def _generate_node(workflow_name: str):
    with (ROOT / "workflows" / workflow_name).open() as workflow_file:
        workflow = json.load(workflow_file)
    return next(node for node in workflow["nodes"] if node["type"] == "AnimaIPAdapterGenerate")


def test_workflow_widgets_keep_comfy_seed_control_aligned() -> None:
    for workflow_name in WORKFLOWS:
        node = _generate_node(workflow_name)
        values = node["widgets_values"]

        assert values[2] == 20260610
        assert values[3] == "fixed"
        assert isinstance(values[4], int)
        assert isinstance(values[5], int)


def test_workflow_uses_model_selector_names_instead_of_raw_paths() -> None:
    forbidden_input_names = {
        "dit_path",
        "text_encoder_path",
        "vae_path",
        "checkpoint_path",
        "anima_root",
        "python_executable",
    }

    for workflow_name in WORKFLOWS:
        node = _generate_node(workflow_name)
        input_names = {input_slot["name"] for input_slot in node["inputs"]}
        values_text = "\n".join(str(value) for value in node["widgets_values"])

        assert forbidden_input_names.isdisjoint(input_names)
        assert {"dit_name", "text_encoder_name", "vae_name", "ipadapter_name"} <= input_names
        assert "/home/wktwin/" not in values_text
        assert "/data/ai/" not in values_text
