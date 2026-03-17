"""Core data models for AutoClip."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class WordToken(BaseModel, frozen=True):
    """A single word with precise timestamps from ASR."""

    id: str
    start_sec: float
    end_sec: float
    text: str
    probability: float = 1.0


class CaptionSegment(BaseModel, frozen=True):
    """A segment of transcription containing multiple words."""

    start_sec: float
    end_sec: float
    text: str
    words: tuple[WordToken, ...] = ()
    no_speech_prob: float = 0.0


class RemovalReason(StrEnum):
    """Internal 5-category classification of disfluencies."""

    STUTTER = "stutter"
    REPEAT = "repeat"
    FILLER = "filler"
    FALSE_START = "false_start"
    LONG_PAUSE = "long_pause"


# Mapping from internal reasons to CLI categories
CLI_CATEGORY_MAP: dict[RemovalReason, str] = {
    RemovalReason.STUTTER: "repeat",
    RemovalReason.REPEAT: "repeat",
    RemovalReason.FILLER: "filler",
    RemovalReason.FALSE_START: "false-start",
    RemovalReason.LONG_PAUSE: "pause",
}

# Mapping from CLI category to internal reasons
CLI_TO_INTERNAL: dict[str, list[RemovalReason]] = {
    "filler": [RemovalReason.FILLER],
    "repeat": [RemovalReason.STUTTER, RemovalReason.REPEAT],
    "false-start": [RemovalReason.FALSE_START],
    "pause": [RemovalReason.LONG_PAUSE],
}

ALL_CLI_CATEGORIES: list[str] = ["filler", "repeat", "false-start", "pause"]


class AutoRemovalCandidate(BaseModel, frozen=True):
    """A candidate word for automatic removal."""

    word_id: str
    text: str
    reason: RemovalReason
    confidence: float = Field(ge=0.0, le=1.0)
    start_sec: float
    end_sec: float


class AnalysisCandidate(AutoRemovalCandidate, frozen=True):
    """A removal candidate with source tracking for analysis reports."""

    source: Literal["keyword", "heuristic", "llm"]


class AppliedParams(BaseModel, frozen=True):
    """Parameters applied during the clean operation."""

    threshold: float
    categories: list[str]


class AnalysisResult(BaseModel, frozen=True):
    """Full analysis result for persistence and report generation."""

    source: str
    original_duration_sec: float
    detected_language: str
    words: tuple[WordToken, ...]
    candidates: tuple[AnalysisCandidate, ...]
    applied_params: AppliedParams


class Segment(BaseModel, frozen=True):
    """A retained time segment for export."""

    start_sec: float
    end_sec: float


class RemovalEntry(BaseModel, frozen=True):
    """A removal entry for JSON output."""

    word_id: str
    text: str
    reason: str
    confidence: float
    start_sec: float
    end_sec: float


class CleanResult(BaseModel, frozen=True):
    """Result of the clean operation, written as JSON metadata."""

    source: str
    original_duration_sec: float
    cleaned_duration_sec: float
    reduction_percent: float
    removals: tuple[RemovalEntry, ...] = ()
    removal_counts: dict[str, int] = Field(default_factory=dict)
    retained_segments: tuple[Segment, ...] = ()
    detected_language: str = "unknown"


# Valid CLI category literals for Click validation
CliCategory = Literal["filler", "repeat", "false-start", "pause"]
