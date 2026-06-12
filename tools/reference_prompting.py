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
    selected_texts: set[str] = set()
    for item in ranked:
        if item.candidate.text in selected_texts:
            continue
        current = category_counts.get(item.candidate.category, 0)
        if current >= max_per_category:
            continue
        selected.append(item.candidate)
        selected_texts.add(item.candidate.text)
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
        AttributeCandidate("identity_role", "long black-haired wuxia swordsman"),
        AttributeCandidate("identity_role", "angry martial artist close-up"),
        AttributeCandidate("identity_role", "old bearded martial arts master"),
        AttributeCandidate("identity_role", "bald old monk with prayer beads"),
        AttributeCandidate("identity_role", "young scholar with glasses"),
        AttributeCandidate("identity_role", "red-haired noble woman in ornate dress"),
        AttributeCandidate("identity_role", "middle-aged court official with black hat"),
        AttributeCandidate("identity_role", "masked dark-robed assassin"),
        AttributeCandidate("identity_role", "stern palace guard in armor"),
        AttributeCandidate("identity_role", "elegant sword master in white robe"),
        AttributeCandidate("age_facial_hair", "old bearded martial arts master"),
        AttributeCandidate("age_facial_hair", "bald old monk with long white gray beard"),
        AttributeCandidate("age_facial_hair", "white gray beard and thick eyebrows"),
        AttributeCandidate("age_facial_hair", "black mustache and official face"),
        AttributeCandidate("age_facial_hair", "young clean-shaven warrior"),
        AttributeCandidate("age_facial_hair", "middle-aged man with short beard"),
        AttributeCandidate("age_facial_hair", "elderly wrinkled face"),
        AttributeCandidate("age_facial_hair", "thin pointed goatee"),
        AttributeCandidate("hair_color_style", "long flowing black hair"),
        AttributeCandidate("hair_color_style", "red hair and pale makeup"),
        AttributeCandidate("hair_color_style", "white silver hair"),
        AttributeCandidate("hair_color_style", "messy black hair"),
        AttributeCandidate("hair_color_style", "hair tied in topknot"),
        AttributeCandidate("hair_color_style", "short cropped black hair"),
        AttributeCandidate("hair_color_style", "long loose white hair"),
        AttributeCandidate("hair_color_style", "dark hair with ornate hairpin"),
        AttributeCandidate("gender_presentation", "human martial arts character"),
        AttributeCandidate("gender_presentation", "male wuxia swordsman"),
        AttributeCandidate("gender_presentation", "female noble court character"),
        AttributeCandidate("gender_presentation", "androgynous elegant warrior"),
        AttributeCandidate("gender_presentation", "elder male master"),
        AttributeCandidate("gender_presentation", "young female swordswoman"),
        AttributeCandidate("expression", "stern serious expression"),
        AttributeCandidate("expression", "angry tense expression"),
        AttributeCandidate("expression", "surprised exaggerated face"),
        AttributeCandidate("expression", "calm seated expression"),
        AttributeCandidate("expression", "open screaming mouth"),
        AttributeCandidate("expression", "sharp eyebrows and stern eyes"),
        AttributeCandidate("expression", "cold emotionless stare"),
        AttributeCandidate("expression", "soft gentle smile"),
        AttributeCandidate("expression", "wide shocked eyes"),
        AttributeCandidate("framing", "upper body close-up portrait"),
        AttributeCandidate("framing", "side profile portrait"),
        AttributeCandidate("framing", "seated full-body indoor panel"),
        AttributeCandidate("framing", "large speech bubble comic panel"),
        AttributeCandidate("framing", "arm thrust forward action pose"),
        AttributeCandidate("framing", "front-facing portrait crop"),
        AttributeCandidate("framing", "three-quarter view portrait"),
        AttributeCandidate("framing", "full-body standing pose"),
        AttributeCandidate("outfit_color", "black martial robe with red trim"),
        AttributeCandidate("outfit_color", "blue gray scholar robe and official hat"),
        AttributeCandidate("outfit_color", "ornate red and gold palace dress"),
        AttributeCandidate("outfit_color", "black official hat and formal robe"),
        AttributeCandidate("outfit_color", "tan traditional robe"),
        AttributeCandidate("outfit_color", "dark armored robe"),
        AttributeCandidate("outfit_color", "white flowing martial robe"),
        AttributeCandidate("outfit_color", "green scholar robe"),
        AttributeCandidate("outfit_color", "purple villain robe"),
        AttributeCandidate("outfit_color", "gold embroidered court robe"),
        AttributeCandidate("accessory_prop", "black official hat"),
        AttributeCandidate("accessory_prop", "round scholar glasses"),
        AttributeCandidate("accessory_prop", "folding fan in hand"),
        AttributeCandidate("accessory_prop", "sword hilt visible"),
        AttributeCandidate("accessory_prop", "red prayer beads necklace"),
        AttributeCandidate("accessory_prop", "ornate hairpin"),
        AttributeCandidate("accessory_prop", "fur collar cloak"),
        AttributeCandidate("accessory_prop", "metal shoulder armor"),
        AttributeCandidate("weapon_action", "drawn sword action pose"),
        AttributeCandidate("weapon_action", "raised hand martial arts gesture"),
        AttributeCandidate("weapon_action", "holding a staff"),
        AttributeCandidate("weapon_action", "clenched fist foreground"),
        AttributeCandidate("weapon_action", "flowing robe in motion"),
        AttributeCandidate("non_human_trait", "green monster face with red glowing eye"),
        AttributeCandidate("non_human_trait", "green non-human demon face"),
        AttributeCandidate("non_human_trait", "pale purple-skinned villain"),
        AttributeCandidate("non_human_trait", "red glowing demonic eye"),
        AttributeCandidate("non_human_trait", "sharp fangs visible"),
        AttributeCandidate("non_human_trait", "horned demon silhouette"),
        AttributeCandidate("lighting_palette", "warm orange firelit background"),
        AttributeCandidate("lighting_palette", "cool blue gray palace lighting"),
        AttributeCandidate("lighting_palette", "red gold indoor palace colors"),
        AttributeCandidate("lighting_palette", "dark night palace background"),
        AttributeCandidate("lighting_palette", "misty blue moonlight"),
        AttributeCandidate("lighting_palette", "bright daylight courtyard"),
        AttributeCandidate("lighting_palette", "green eerie cave lighting"),
        AttributeCandidate("lighting_palette", "purple villain aura lighting"),
        AttributeCandidate("background_setting", "simple palace background"),
        AttributeCandidate("background_setting", "indoor throne hall background"),
        AttributeCandidate("background_setting", "outdoor martial arts courtyard"),
        AttributeCandidate("background_setting", "night palace rooftop"),
        AttributeCandidate("background_setting", "wooden temple interior"),
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
