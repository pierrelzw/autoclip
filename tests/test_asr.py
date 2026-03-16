"""Tests for ASR provider with fixture data."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from autoclip.providers.asr.whisper_local import (
    WhisperLocalProvider,
    _filter_hallucinations,
    _to_caption_segments,
)


def _make_segment(
    text: str,
    start: float,
    end: float,
    no_speech_prob: float = 0.0,
    words: list[object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        start=start,
        end=end,
        no_speech_prob=no_speech_prob,
        words=words or [],
    )


def _make_word(text: str, start: float, end: float, prob: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(word=text, start=start, end=end, probability=prob)


class TestFilterHallucinations:
    def test_removes_high_no_speech_prob(self) -> None:
        segs = [
            _make_segment("hello", 0.0, 1.0, no_speech_prob=0.1),
            _make_segment("phantom", 1.0, 2.0, no_speech_prob=0.8),
            _make_segment("world", 2.0, 3.0, no_speech_prob=0.2),
        ]
        filtered = _filter_hallucinations(segs)
        assert len(filtered) == 2
        assert filtered[0].text == "hello"
        assert filtered[1].text == "world"

    def test_removes_consecutive_duplicates(self) -> None:
        segs = [
            _make_segment("hello world", 0.0, 1.0),
            _make_segment("hello world", 1.0, 2.0),
            _make_segment("goodbye", 2.0, 3.0),
        ]
        filtered = _filter_hallucinations(segs)
        assert len(filtered) == 2

    def test_keeps_non_consecutive_duplicates(self) -> None:
        segs = [
            _make_segment("hello", 0.0, 1.0),
            _make_segment("world", 1.0, 2.0),
            _make_segment("hello", 2.0, 3.0),
        ]
        filtered = _filter_hallucinations(segs)
        assert len(filtered) == 3

    def test_empty_input(self) -> None:
        assert _filter_hallucinations([]) == []


class TestToCaptionSegments:
    def test_conversion(self) -> None:
        words = [
            _make_word("hello", 0.0, 0.4),
            _make_word("world", 0.5, 0.9),
        ]
        segs = [_make_segment("hello world", 0.0, 1.0, words=words)]
        result = _to_caption_segments(segs)
        assert len(result) == 1
        assert len(result[0].words) == 2
        assert result[0].words[0].text == "hello"
        assert result[0].words[1].start_sec == 0.5

    def test_empty_words_filtered(self) -> None:
        words = [
            _make_word("hello", 0.0, 0.4),
            _make_word("  ", 0.4, 0.5),  # whitespace-only
            _make_word("world", 0.5, 0.9),
        ]
        segs = [_make_segment("hello world", 0.0, 1.0, words=words)]
        result = _to_caption_segments(segs)
        assert len(result[0].words) == 2


class TestWhisperLocalProvider:
    @patch("autoclip.providers.asr.whisper_local.WhisperLocalProvider._get_model")
    def test_transcribe(self, mock_get_model: MagicMock) -> None:
        words = [_make_word("hello", 0.0, 0.4), _make_word("world", 0.5, 0.9)]
        seg = _make_segment("hello world", 0.0, 1.0, words=words)
        info = SimpleNamespace(language="en")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], info)
        mock_get_model.return_value = mock_model

        provider = WhisperLocalProvider(model_size="base")
        segments, lang = provider.transcribe("audio.wav")

        assert lang == "en"
        assert len(segments) == 1
        assert segments[0].text == "hello world"

    @patch("autoclip.providers.asr.whisper_local.WhisperLocalProvider._get_model")
    def test_auto_language(self, mock_get_model: MagicMock) -> None:
        info = SimpleNamespace(language="zh")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], info)
        mock_get_model.return_value = mock_model

        provider = WhisperLocalProvider()
        _, lang = provider.transcribe("audio.wav", language="auto")
        assert lang == "zh"

        # Verify language=None was passed to model
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] is None
