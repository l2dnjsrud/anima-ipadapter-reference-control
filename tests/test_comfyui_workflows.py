from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_WORKFLOWS = (
    "anima_ipadapter_reference_generate.json",
    "anima_ipadapter_contactsheet_ref03_ersde.json",
)
NATIVE_WORKFLOW = "anima_ipadapter_pe_native_reference.json"


def _load_workflow(workflow_name: str) -> dict:
    with (ROOT / "workflows" / workflow_name).open() as workflow_file:
        return json.load(workflow_file)


def _generate_node(workflow_name: str):
    workflow = _load_workflow(workflow_name)
    return next(node for node in workflow["nodes"] if node["type"] == "AnimaIPAdapterGenerate")


def test_workflow_widgets_keep_comfy_seed_control_aligned() -> None:
    for workflow_name in LEGACY_WORKFLOWS:
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

    for workflow_name in LEGACY_WORKFLOWS:
        node = _generate_node(workflow_name)
        input_names = {input_slot["name"] for input_slot in node["inputs"]}
        values_text = "\n".join(str(value) for value in node["widgets_values"])

        assert forbidden_input_names.isdisjoint(input_names)
        assert {"dit_name", "text_encoder_name", "vae_name", "ipadapter_name"} <= input_names
        assert "/home/wktwin/" not in values_text
        assert "/data/ai/" not in values_text


def test_native_workflow_is_normal_comfyui_graph() -> None:
    workflow = _load_workflow(NATIVE_WORKFLOW)
    node_types = {node["type"] for node in workflow["nodes"]}

    required = {
        "LoadImage",
        "AnimaPEIPAdapterLoader",
        "AnimaPEEncodeImage",
        "AnimaPEIPAdapterApply",
        "UNETLoader",
        "CLIPLoader",
        "CLIPTextEncodeFlux",
        "EmptySD3LatentImage",
        "RandomNoise",
        "CFGGuider",
        "KSamplerSelect",
        "BasicScheduler",
        "SamplerCustomAdvanced",
        "VAELoader",
        "VAEDecode",
        "SaveImage",
    }

    assert required <= node_types
    assert "AnimaIPAdapterGenerate" not in node_types


def test_native_workflow_links_apply_model_to_sampler_path() -> None:
    workflow = _load_workflow(NATIVE_WORKFLOW)
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    apply_node = next(node for node in workflow["nodes"] if node["type"] == "AnimaPEIPAdapterApply")
    output_link_ids = set(apply_node["outputs"][0]["links"])
    destinations = {(links[link_id][3], links[link_id][4]) for link_id in output_link_ids}

    assert any(nodes[node_id]["type"] == "CFGGuider" and slot == 0 for node_id, slot in destinations)
    assert any(nodes[node_id]["type"] == "BasicScheduler" and slot == 0 for node_id, slot in destinations)

    input_sources = {
        input_slot["name"]: links[input_slot["link"]][1]
        for input_slot in apply_node["inputs"]
        if input_slot.get("link") is not None
    }
    assert nodes[input_sources["model"]]["type"] == "UNETLoader"
    assert nodes[input_sources["adapter"]]["type"] == "AnimaPEIPAdapterLoader"
    assert nodes[input_sources["features"]]["type"] == "AnimaPEEncodeImage"
