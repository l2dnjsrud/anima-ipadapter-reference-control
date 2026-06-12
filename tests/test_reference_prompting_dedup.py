from __future__ import annotations

from tools.reference_prompting import AttributeCandidate, select_top_attributes


def test_select_top_attributes_deduplicates_repeated_candidate_text() -> None:
    candidates = (
        AttributeCandidate("identity_role", "old bearded martial arts master"),
        AttributeCandidate("age_facial_hair", "old bearded martial arts master"),
        AttributeCandidate("expression", "stern serious expression"),
    )

    selected = select_top_attributes(candidates, (0.99, 0.98, 0.5))

    assert [candidate.text for candidate in selected] == [
        "old bearded martial arts master",
        "stern serious expression",
    ]
