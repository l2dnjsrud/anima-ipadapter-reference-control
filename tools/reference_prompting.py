from __future__ import annotations

from dataclasses import dataclass
from typing import Final


DEFAULT_PREFIX: Final = (
    "masterpiece, best quality, score_7, safe, mrcolor_panel_style, "
    "solo character portrait"
)
DEFAULT_STYLE: Final = (
    "full color manhwa comic panel, clean ink lines, cel shaded anime style, "
    "dramatic lighting, simple palace background, no text"
)
DEFAULT_DETAIL: Final = "wuxia martial artist, upper body, traditional robe"


class CandidateScoreMismatchError(Exception):
    def __init__(self, candidates: int, scores: int) -> None:
        self.candidates = candidates
        self.scores = scores
        super().__init__(str(self))

    def __str__(self) -> str:
        return (
            "candidate and score counts differ: "
            f"candidates={self.candidates}, scores={self.scores}"
        )


@dataclass(frozen=True, slots=True)
class AttributeCandidate:
    category: str
    text: str


@dataclass(frozen=True, slots=True)
class ScoredAttribute:
    candidate: AttributeCandidate
    score: float

    @property
    def text(self) -> str:
        return self.candidate.text


def select_top_attributes(
    candidates: tuple[AttributeCandidate, ...],
    scores: tuple[float, ...],
    *,
    max_per_category: int = 1,
    min_score: float | None = None,
) -> tuple[AttributeCandidate, ...]:
    if len(candidates) != len(scores):
        raise CandidateScoreMismatchError(len(candidates), len(scores))
    ranked = sorted(
        (
            ScoredAttribute(candidate=candidate, score=score)
            for candidate, score in zip(candidates, scores, strict=True)
            if min_score is None or score >= min_score
        ),
        key=lambda item: (-item.score, item.candidate.category, item.candidate.text),
    )
    selected: list[AttributeCandidate] = []
    category_counts: dict[str, int] = {}
    for item in ranked:
        current = category_counts.get(item.candidate.category, 0)
        if current >= max_per_category:
            continue
        selected.append(item.candidate)
        category_counts[item.candidate.category] = current + 1
    return tuple(selected)


def build_reference_prompt(
    source_caption: str,
    selected_attributes: tuple[AttributeCandidate, ...],
    *,
    prefix: str = DEFAULT_PREFIX,
    style: str = DEFAULT_STYLE,
    fallback_detail: str = DEFAULT_DETAIL,
) -> str:
    details = tuple(candidate.text for candidate in selected_attributes)
    if not details:
        details = (_clean_source_caption(source_caption) or fallback_detail,)
    return _join_prompt_parts((prefix, *details, style))


def default_attribute_candidates() -> tuple[AttributeCandidate, ...]:
    return (
        AttributeCandidate("identity", "long black-haired wuxia swordsman"),
        AttributeCandidate("identity", "angry martial artist close-up"),
        AttributeCandidate("identity", "old bearded martial arts master"),
        AttributeCandidate("identity", "bald old monk with long white gray beard"),
        AttributeCandidate("identity", "young scholar with glasses"),
        AttributeCandidate("identity", "red-haired noble woman in ornate dress"),
        AttributeCandidate("identity", "middle-aged court official with black hat"),
        AttributeCandidate("identity", "pale purple-skinned villain"),
        AttributeCandidate("identity", "green non-human demon face"),
        AttributeCandidate("species", "green monster face with red glowing eye"),
        AttributeCandidate("species", "human martial arts character"),
        AttributeCandidate("hair_face", "long flowing black hair"),
        AttributeCandidate("hair_face", "red hair and pale makeup"),
        AttributeCandidate("hair_face", "black mustache and official face"),
        AttributeCandidate("hair_face", "white gray beard and thick eyebrows"),
        AttributeCandidate("hair_face", "sharp eyebrows and stern eyes"),
        AttributeCandidate("hair_face", "open screaming mouth"),
        AttributeCandidate("costume", "black martial robe with red trim"),
        AttributeCandidate("costume", "blue gray scholar robe and official hat"),
        AttributeCandidate("costume", "ornate red and gold palace dress"),
        AttributeCandidate("costume", "black official hat and formal robe"),
        AttributeCandidate("costume", "tan traditional robe"),
        AttributeCandidate("costume", "dark armored robe"),
        AttributeCandidate("expression", "stern serious expression"),
        AttributeCandidate("expression", "angry tense expression"),
        AttributeCandidate("expression", "surprised exaggerated face"),
        AttributeCandidate("expression", "calm seated expression"),
        AttributeCandidate("composition", "upper body close-up portrait"),
        AttributeCandidate("composition", "side profile portrait"),
        AttributeCandidate("composition", "seated full-body indoor panel"),
        AttributeCandidate("composition", "large speech bubble comic panel"),
        AttributeCandidate("composition", "arm thrust forward action pose"),
        AttributeCandidate("palette", "warm orange firelit background"),
        AttributeCandidate("palette", "cool blue gray palace lighting"),
        AttributeCandidate("palette", "red gold indoor palace colors"),
        AttributeCandidate("palette", "dark night palace background"),
    )


def _clean_source_caption(source_caption: str) -> str:
    ignored = {
        "mrcolor_panel_style",
        "full color manga panel",
        "clean webtoon coloring",
        "manhwa panel art",
        "character panel",
        "close-up panel",
        "action panel",
        "single panel",
    }
    parts = [
        part.strip()
        for part in source_caption.split(",")
        if part.strip() and part.strip() not in ignored
    ]
    return ", ".join(parts)


def _join_prompt_parts(parts: tuple[str, ...]) -> str:
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        for token in part.split(","):
            clean = token.strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            result.append(clean)
    return ", ".join(result)
