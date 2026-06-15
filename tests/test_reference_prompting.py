from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from tools import build_reference_prompt_manifest
from tools.reference_prompt_manifest import (
    ManifestPromptRow,
    MissingReferenceImageError,
    OutputAlreadyExistsError,
    ReferencePromptSourceRow,
    build_reference_prompt_rows,
    validate_reference_source_images,
    write_reference_prompt_rows,
)
from tools.reference_prompting import (
    AttributeCandidate,
    CandidateScoreMismatchError,
    build_reference_prompt,
    default_attribute_candidates,
    select_top_attributes,
)


RUNNER = CliRunner()


class FakeScorer:
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(self, image_path: Path, candidate_texts: tuple[str, ...]) -> tuple[float, ...]:
        return tuple(self._scores[text] for text in candidate_texts)


def test_select_top_attributes_keeps_best_candidate_per_category() -> None:
    candidates = (
        AttributeCandidate("identity", "old bearded martial arts master"),
        AttributeCandidate("identity", "young scholar with glasses"),
        AttributeCandidate("palette", "deep blue robe and cool palace lighting"),
        AttributeCandidate("palette", "warm orange firelit background"),
    )
    scores = (0.41, 0.83, 0.75, 0.32)

    selected = select_top_attributes(candidates, scores, max_per_category=1)

    assert [item.text for item in selected] == [
        "young scholar with glasses",
        "deep blue robe and cool palace lighting",
    ]


def test_select_top_attributes_rejects_score_count_mismatch() -> None:
    candidates = (
        AttributeCandidate("identity", "old bearded martial arts master"),
        AttributeCandidate("palette", "warm orange firelit background"),
    )

    try:
        select_top_attributes(candidates, (0.2,))
    except CandidateScoreMismatchError as exc:
        assert exc.candidates == 2
        assert exc.scores == 1
    else:
        raise AssertionError("candidate and score count mismatch should fail")


def test_default_attribute_candidates_cover_v3_reference_categories() -> None:
    candidates = default_attribute_candidates()
    categories = {candidate.category for candidate in candidates}

    assert len(categories) >= 10
    assert len(candidates) >= 80
    assert {
        "age_facial_hair",
        "hair_color_style",
        "expression",
        "framing",
        "outfit_color",
        "accessory_prop",
        "non_human_trait",
        "lighting_palette",
    } <= categories


def test_build_reference_prompt_uses_retrieved_attributes_without_generic_noise() -> None:
    selected = (
        AttributeCandidate("identity", "bald old monk with long white gray beard"),
        AttributeCandidate("expression", "angry stern face"),
    )

    prompt = build_reference_prompt(
        "mrcolor_panel_style, character panel, close-up panel, single panel",
        selected,
    )

    assert "bald old monk with long white gray beard" in prompt
    assert "angry stern face" in prompt
    assert "mrcolor_panel_style" in prompt
    assert "single panel, single panel" not in prompt


def test_build_reference_prompt_falls_back_to_safe_single_character_prompt() -> None:
    prompt = build_reference_prompt("", ())

    assert "solo character portrait" in prompt
    assert "full color manhwa comic panel" in prompt


def test_build_reference_prompt_rows_scores_images_and_writes_jsonl(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "SG-001" / "portrait.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (720, 960), (120, 80, 40)).save(image_path)
    rows = (
        ReferencePromptSourceRow(
            ref_id="SG-001/portrait",
            tgt_id="SG-001/portrait",
            prompt="mrcolor_panel_style, character panel, close-up panel",
        ),
    )
    candidates = (
        AttributeCandidate("identity", "old bearded martial arts master"),
        AttributeCandidate("identity", "young scholar with glasses"),
        AttributeCandidate("palette", "warm orange firelit background"),
    )
    scorer = FakeScorer(
        {
            "old bearded martial arts master": 0.21,
            "young scholar with glasses": 0.91,
            "warm orange firelit background": 0.82,
        }
    )

    prompt_rows = build_reference_prompt_rows(
        rows,
        dataset_root=dataset_root,
        scorer=scorer,
        candidates=candidates,
    )
    output_path = tmp_path / "reference_prompts.jsonl"
    write_reference_prompt_rows(prompt_rows, output_path)

    assert prompt_rows == (
        ManifestPromptRow(
            ref_id="SG-001/portrait",
            tgt_id="SG-001/portrait",
            source_prompt="mrcolor_panel_style, character panel, close-up panel",
            prompt=prompt_rows[0].prompt,
            selected_attributes=(
                "young scholar with glasses",
                "warm orange firelit background",
            ),
        ),
    )
    loaded = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert loaded[0]["selected_attributes"] == [
        "young scholar with glasses",
        "warm orange firelit background",
    ]


def test_write_reference_prompt_rows_refuses_existing_output_by_default(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "reference_prompts.jsonl"
    output_path.write_text("existing\n", encoding="utf-8")
    rows = (
        ManifestPromptRow(
            ref_id="SG-001/portrait",
            tgt_id="SG-001/portrait",
            source_prompt="source",
            prompt="prompt",
            selected_attributes=("old bearded martial arts master",),
        ),
    )

    try:
        write_reference_prompt_rows(rows, output_path)
    except OutputAlreadyExistsError as exc:
        assert exc.output_path == output_path
    else:
        raise AssertionError("existing prompt manifest should not be overwritten")

    assert output_path.read_text(encoding="utf-8") == "existing\n"


def test_prompt_manifest_cli_refuses_existing_output_before_loading_scorer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "SG-001" / "portrait.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (720, 960), (120, 80, 40)).save(image_path)
    manifest_path = tmp_path / "source.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "ref_id": "SG-001/portrait",
                "tgt_id": "SG-001/portrait",
                "prompt": "mrcolor_panel_style, character panel",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "reference_prompts.jsonl"
    output_path.write_text("existing\n", encoding="utf-8")

    def fail_if_loaded(config: build_reference_prompt_manifest.Qwen3VLScorerConfig):
        raise AssertionError(f"scorer should not load for existing output: {config.model_id}")

    monkeypatch.setattr(
        build_reference_prompt_manifest,
        "Qwen3VLReferenceTextScorer",
        fail_if_loaded,
    )

    result = RUNNER.invoke(
        build_reference_prompt_manifest.app,
        [str(manifest_path), str(dataset_root), str(output_path)],
    )

    assert result.exit_code == 1
    assert "already exists" in result.stderr
    assert output_path.read_text(encoding="utf-8") == "existing\n"


def test_write_reference_prompt_rows_can_force_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "reference_prompts.jsonl"
    output_path.write_text("existing\n", encoding="utf-8")
    rows = (
        ManifestPromptRow(
            ref_id="SG-001/portrait",
            tgt_id="SG-001/portrait",
            source_prompt="source",
            prompt="prompt",
            selected_attributes=("old bearded martial arts master",),
        ),
    )

    write_reference_prompt_rows(rows, output_path, overwrite=True)

    loaded = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert loaded[0]["prompt"] == "prompt"


def test_build_reference_prompt_rows_rejects_missing_reference_image(tmp_path: Path) -> None:
    rows = (
        ReferencePromptSourceRow(
            ref_id="SG-001/missing",
            tgt_id="SG-001/missing",
            prompt="mrcolor_panel_style, character panel",
        ),
    )
    scorer = FakeScorer({"old bearded martial arts master": 0.1})
    candidates = (AttributeCandidate("identity", "old bearded martial arts master"),)

    try:
        build_reference_prompt_rows(
            rows,
            dataset_root=tmp_path,
            scorer=scorer,
            candidates=candidates,
        )
    except MissingReferenceImageError as exc:
        assert str(exc.image_path).endswith("SG-001/missing.jpg")
    else:
        raise AssertionError("missing reference image should fail")


def test_validate_reference_source_images_fails_before_scoring(tmp_path: Path) -> None:
    rows = (
        ReferencePromptSourceRow(
            ref_id="SG-999/not_there",
            tgt_id="SG-999/not_there",
            prompt="mrcolor_panel_style, character panel",
        ),
    )

    try:
        validate_reference_source_images(rows, tmp_path)
    except MissingReferenceImageError as exc:
        assert "not_there.jpg" in str(exc)
    else:
        raise AssertionError("missing reference image validation should fail")
