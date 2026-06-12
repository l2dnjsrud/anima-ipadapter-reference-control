from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

Label = Literal["positive", "negative"]

DIRECT_GREEN_KEYWORDS: Final = ("green monster", "green non-human", "green-skinned demon", "green demon")
RED_EYE_KEYWORDS: Final = ("red glowing demonic eye",)
PALE_KEYWORDS: Final = ("pale purple-skinned villain",)
FANG_PROFILE_KEYWORDS: Final = ("sharp fangs visible", "side profile portrait")
HUMAN_KEYWORDS: Final = ("human martial arts character", "young clean-shaven warrior", "female noble court character")
OLD_HEADWEAR_KEYWORDS: Final = ("old bearded martial arts master", "middle-aged court official with black hat", "black official hat", "upper body close-up portrait")
SIDECAR_KEYWORDS: Final = DIRECT_GREEN_KEYWORDS + RED_EYE_KEYWORDS + PALE_KEYWORDS + FANG_PROFILE_KEYWORDS


@dataclass(frozen=True, slots=True)
class C066InputError(Exception):
    detail: str


@dataclass(frozen=True, slots=True)
class GreenMetrics:
    green_ratio: float
    strong_green_ratio: float
    red_ratio: float


@dataclass(frozen=True, slots=True)
class C066CandidateRow:
    image_id: str
    label: Label
    source_bucket: str
    candidate_source: str
    matched_keywords: tuple[str, ...]
    selected_attributes: tuple[str, ...]
    caption: str
    image_path: str
    caption_path: str
    green_ratio: float
    strong_green_ratio: float
    red_ratio: float
    source_split: str
    heldout_excluded: bool
    path_exists: bool


@dataclass(frozen=True, slots=True)
class C066PairRow:
    pair_id: str
    label: Label
    anchor_id: str
    candidate_id: str
    anchor_group: str
    candidate_group: str


@dataclass(frozen=True, slots=True)
class C066Config:
    dataset_root: Path
    train_manifest_path: Path
    heldout_manifest_path: Path
    gate_summary_path: Path
    c065_pair_manifest_path: Path | None
    output_manifest_path: Path
    output_summary_path: Path
    output_pair_manifest_path: Path
    max_per_bucket: int = 40
    green_ratio_min: float = 0.02
    strong_green_ratio_min: float = 0.002


@dataclass(frozen=True, slots=True)
class C066Summary:
    total_candidates: int
    positive_candidates: int
    negative_candidates: int
    direct_green_positive_count: int
    direct_green_pixel_candidate_count: int
    non_human_positive_count: int
    heldout_rows_used: int
    missing_paths: int
    sidecar_caption_keyword_hits: int
    source_buckets: dict[str, int]
    pair_rows: int
