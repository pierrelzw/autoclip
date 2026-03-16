"""Tests for data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autoclip.models import (
    ALL_CLI_CATEGORIES,
    CLI_CATEGORY_MAP,
    CLI_TO_INTERNAL,
    AutoRemovalCandidate,
    CaptionSegment,
    CleanResult,
    RemovalEntry,
    RemovalReason,
    Segment,
    WordToken,
)


class TestWordToken:
    def test_create(self) -> None:
        token = WordToken(id="w_0", start_sec=0.0, end_sec=0.5, text="hello")
        assert token.id == "w_0"
        assert token.start_sec == 0.0
        assert token.end_sec == 0.5
        assert token.text == "hello"
        assert token.probability == 1.0

    def test_frozen(self) -> None:
        token = WordToken(id="w_0", start_sec=0.0, end_sec=0.5, text="hello")
        with pytest.raises(ValidationError):
            token.text = "world"  # type: ignore[misc]


class TestCaptionSegment:
    def test_create_with_words(self) -> None:
        w1 = WordToken(id="w_0", start_sec=0.0, end_sec=0.3, text="hello")
        w2 = WordToken(id="w_1", start_sec=0.3, end_sec=0.6, text="world")
        seg = CaptionSegment(
            start_sec=0.0, end_sec=0.6, text="hello world", words=(w1, w2)
        )
        assert len(seg.words) == 2
        assert seg.no_speech_prob == 0.0

    def test_empty_words(self) -> None:
        seg = CaptionSegment(start_sec=0.0, end_sec=1.0, text="test")
        assert seg.words == ()


class TestRemovalReason:
    def test_values(self) -> None:
        assert RemovalReason.STUTTER.value == "stutter"
        assert RemovalReason.REPEAT.value == "repeat"
        assert RemovalReason.FILLER.value == "filler"
        assert RemovalReason.FALSE_START.value == "false_start"
        assert RemovalReason.LONG_PAUSE.value == "long_pause"


class TestAutoRemovalCandidate:
    def test_create(self) -> None:
        c = AutoRemovalCandidate(
            word_id="w_3",
            text="um",
            reason=RemovalReason.FILLER,
            confidence=1.0,
            start_sec=2.0,
            end_sec=2.3,
        )
        assert c.reason == RemovalReason.FILLER

    def test_confidence_validation(self) -> None:
        with pytest.raises(ValidationError):
            AutoRemovalCandidate(
                word_id="w_0",
                text="x",
                reason=RemovalReason.FILLER,
                confidence=1.5,
                start_sec=0.0,
                end_sec=0.1,
            )


class TestSegment:
    def test_create(self) -> None:
        s = Segment(start_sec=1.0, end_sec=5.0)
        assert s.end_sec - s.start_sec == 4.0


class TestCleanResult:
    def test_create(self) -> None:
        result = CleanResult(
            source="test.mp4",
            original_duration_sec=60.0,
            cleaned_duration_sec=55.0,
            reduction_percent=8.33,
            removal_counts={"filler": 5, "repeat": 2},
            detected_language="en",
        )
        assert result.reduction_percent == 8.33
        assert result.removal_counts["filler"] == 5


class TestCategoryMappings:
    def test_cli_category_map_covers_all_reasons(self) -> None:
        for reason in RemovalReason:
            assert reason in CLI_CATEGORY_MAP

    def test_cli_to_internal_covers_all_categories(self) -> None:
        for cat in ALL_CLI_CATEGORIES:
            assert cat in CLI_TO_INTERNAL

    def test_stutter_and_repeat_map_to_same_cli_category(self) -> None:
        assert CLI_CATEGORY_MAP[RemovalReason.STUTTER] == "repeat"
        assert CLI_CATEGORY_MAP[RemovalReason.REPEAT] == "repeat"

    def test_removal_entry(self) -> None:
        entry = RemovalEntry(
            word_id="w_0",
            text="um",
            reason="filler",
            confidence=1.0,
            start_sec=0.0,
            end_sec=0.3,
        )
        assert entry.word_id == "w_0"
