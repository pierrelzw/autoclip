"""Comprehensive tests for the fine-cut engine."""

from __future__ import annotations

from autoclip.models import (
    AutoRemovalCandidate,
    CaptionSegment,
    RemovalReason,
    WordToken,
)
from autoclip.processing.finecut import (
    apply_removals,
    detect_fillers,
    detect_pauses,
    merge_retained_segments,
    normalize_whisper_words,
    parse_cleanup_response,
)
from autoclip.processing.prompts import build_cleanup_prompt


def _w(id: str, start: float, end: float, text: str) -> WordToken:
    """Helper to create WordToken."""
    return WordToken(id=id, start_sec=start, end_sec=end, text=text)


class TestNormalizeWhisperWords:
    def test_basic(self) -> None:
        w1 = WordToken(id="", start_sec=0.0, end_sec=0.3, text="hello")
        w2 = WordToken(id="", start_sec=0.3, end_sec=0.6, text="world")
        seg = CaptionSegment(
            start_sec=0.0, end_sec=0.6, text="hello world", words=(w1, w2)
        )
        result = normalize_whisper_words([seg])
        assert len(result) == 2
        assert result[0].id == "w_0"
        assert result[1].id == "w_1"
        assert result[0].text == "hello"

    def test_multiple_segments(self) -> None:
        w1 = WordToken(id="", start_sec=0.0, end_sec=0.3, text="a")
        w2 = WordToken(id="", start_sec=1.0, end_sec=1.3, text="b")
        seg1 = CaptionSegment(start_sec=0.0, end_sec=0.3, text="a", words=(w1,))
        seg2 = CaptionSegment(start_sec=1.0, end_sec=1.3, text="b", words=(w2,))
        result = normalize_whisper_words([seg1, seg2])
        assert len(result) == 2
        assert result[0].id == "w_0"
        assert result[1].id == "w_1"

    def test_empty(self) -> None:
        assert normalize_whisper_words([]) == []


class TestDetectFillers:
    def test_english_fillers(self) -> None:
        words = [
            _w("w_0", 0.0, 0.3, "um"),
            _w("w_1", 0.3, 0.6, "I"),
            _w("w_2", 0.6, 0.9, "uh"),
            _w("w_3", 0.9, 1.2, "think"),
        ]
        candidates = detect_fillers(words, "en")
        assert len(candidates) == 2
        assert all(c.confidence == 1.0 for c in candidates)
        assert all(c.reason == RemovalReason.FILLER for c in candidates)
        assert candidates[0].word_id == "w_0"
        assert candidates[1].word_id == "w_2"

    def test_chinese_fillers(self) -> None:
        words = [
            _w("w_0", 0.0, 0.3, "嗯"),
            _w("w_1", 0.3, 0.6, "我"),
            _w("w_2", 0.6, 0.9, "啊"),
        ]
        candidates = detect_fillers(words, "zh")
        assert len(candidates) == 2

    def test_no_fillers(self) -> None:
        words = [_w("w_0", 0.0, 0.3, "hello"), _w("w_1", 0.3, 0.6, "world")]
        candidates = detect_fillers(words, "en")
        assert len(candidates) == 0

    def test_case_insensitive(self) -> None:
        words = [_w("w_0", 0.0, 0.3, "Um"), _w("w_1", 0.3, 0.6, "UH")]
        candidates = detect_fillers(words, "en")
        assert len(candidates) == 2

    def test_ambiguous_words_not_matched(self) -> None:
        words = [
            _w("w_0", 0.0, 0.3, "like"),
            _w("w_1", 0.3, 0.6, "so"),
            _w("w_2", 0.6, 0.9, "就是"),
        ]
        candidates = detect_fillers(words, "en")
        assert len(candidates) == 0


class TestDetectPauses:
    def test_long_pause(self) -> None:
        words = [
            _w("w_0", 0.0, 1.0, "hello"),
            _w("w_1", 2.0, 3.0, "world"),  # 1.0s gap
        ]
        result_words, candidates = detect_pauses(words, long_pause_ms=500)
        assert len(candidates) == 1
        assert candidates[0].reason == RemovalReason.LONG_PAUSE
        assert candidates[0].confidence == 1.0
        assert candidates[0].start_sec == 1.0
        assert candidates[0].end_sec == 2.0
        # Pause token inserted between words
        assert len(result_words) == 3
        assert result_words[1].text == "[PAUSE]"

    def test_short_gap_no_pause(self) -> None:
        words = [
            _w("w_0", 0.0, 1.0, "hello"),
            _w("w_1", 1.2, 2.0, "world"),  # 0.2s gap
        ]
        result_words, candidates = detect_pauses(words, long_pause_ms=500)
        assert len(candidates) == 0
        assert len(result_words) == 2

    def test_multiple_pauses(self) -> None:
        words = [
            _w("w_0", 0.0, 1.0, "a"),
            _w("w_1", 2.0, 3.0, "b"),  # 1.0s gap
            _w("w_2", 4.0, 5.0, "c"),  # 1.0s gap
        ]
        _, candidates = detect_pauses(words, long_pause_ms=500)
        assert len(candidates) == 2

    def test_single_word(self) -> None:
        words = [_w("w_0", 0.0, 1.0, "hello")]
        result_words, candidates = detect_pauses(words)
        assert len(candidates) == 0
        assert len(result_words) == 1

    def test_empty(self) -> None:
        result_words, candidates = detect_pauses([])
        assert result_words == []
        assert candidates == []


class TestParseCleanupResponse:
    def test_valid_json(self) -> None:
        response = '[{"word_id": "w_3", "text": "I", "reason": "repeat", "confidence": 0.9, "start_sec": 1.0, "end_sec": 1.3}]'
        candidates = parse_cleanup_response(response)
        assert len(candidates) == 1
        assert candidates[0].reason == RemovalReason.REPEAT
        assert candidates[0].confidence == 0.9

    def test_empty_array(self) -> None:
        candidates = parse_cleanup_response("[]")
        assert len(candidates) == 0

    def test_malformed_json(self) -> None:
        candidates = parse_cleanup_response("not json")
        assert len(candidates) == 0

    def test_json_in_code_block(self) -> None:
        response = '```json\n[{"word_id": "w_0", "text": "the", "reason": "stutter", "confidence": 0.85, "start_sec": 0.0, "end_sec": 0.2}]\n```'
        candidates = parse_cleanup_response(response)
        assert len(candidates) == 1
        assert candidates[0].reason == RemovalReason.STUTTER

    def test_unknown_reason_skipped(self) -> None:
        response = '[{"word_id": "w_0", "text": "x", "reason": "unknown_type", "confidence": 0.9, "start_sec": 0.0, "end_sec": 0.1}]'
        candidates = parse_cleanup_response(response)
        assert len(candidates) == 0

    def test_false_start(self) -> None:
        response = '[{"word_id": "w_5", "text": "Actually", "reason": "false_start", "confidence": 0.92, "start_sec": 2.0, "end_sec": 2.5}]'
        candidates = parse_cleanup_response(response)
        assert len(candidates) == 1
        assert candidates[0].reason == RemovalReason.FALSE_START

    def test_non_array_returns_empty(self) -> None:
        candidates = parse_cleanup_response('{"error": "something"}')
        assert len(candidates) == 0


class TestApplyRemovals:
    def test_basic_filtering(self) -> None:
        words = [
            _w("w_0", 0.0, 0.3, "um"),
            _w("w_1", 0.3, 0.6, "I"),
            _w("w_2", 0.6, 0.9, "think"),
        ]
        candidates = [
            AutoRemovalCandidate(
                word_id="w_0", text="um", reason=RemovalReason.FILLER,
                confidence=1.0, start_sec=0.0, end_sec=0.3,
            ),
        ]
        retained, applied = apply_removals(words, candidates, threshold=0.7)
        assert len(retained) == 2
        assert len(applied) == 1
        assert retained[0].text == "I"

    def test_below_threshold_not_removed(self) -> None:
        words = [_w("w_0", 0.0, 0.3, "the"), _w("w_1", 0.3, 0.6, "thing")]
        candidates = [
            AutoRemovalCandidate(
                word_id="w_0", text="the", reason=RemovalReason.STUTTER,
                confidence=0.5, start_sec=0.0, end_sec=0.3,
            ),
        ]
        retained, applied = apply_removals(words, candidates, threshold=0.7)
        assert len(retained) == 2
        assert len(applied) == 0

    def test_false_start_min_threshold(self) -> None:
        words = [_w("w_0", 0.0, 0.5, "Actually")]
        candidates = [
            AutoRemovalCandidate(
                word_id="w_0", text="Actually", reason=RemovalReason.FALSE_START,
                confidence=0.80, start_sec=0.0, end_sec=0.5,
            ),
        ]
        # Default threshold 0.7, but false_start uses max(0.7, 0.85) = 0.85
        retained, applied = apply_removals(words, candidates, threshold=0.7)
        assert len(retained) == 1  # Not removed (0.80 < 0.85)
        assert len(applied) == 0

    def test_false_start_user_threshold_above_floor(self) -> None:
        words = [_w("w_0", 0.0, 0.5, "Actually")]
        candidates = [
            AutoRemovalCandidate(
                word_id="w_0", text="Actually", reason=RemovalReason.FALSE_START,
                confidence=0.87, start_sec=0.0, end_sec=0.5,
            ),
        ]
        # User threshold 0.9 > 0.85 floor, so effective = 0.9
        retained, _applied = apply_removals(words, candidates, threshold=0.9)
        assert len(retained) == 1  # Not removed (0.87 < 0.9)

    def test_category_filter(self) -> None:
        words = [
            _w("w_0", 0.0, 0.3, "um"),
            _w("w_1", 0.3, 0.6, "I"),
            _w("w_2", 0.6, 0.9, "I"),
        ]
        candidates = [
            AutoRemovalCandidate(
                word_id="w_0", text="um", reason=RemovalReason.FILLER,
                confidence=1.0, start_sec=0.0, end_sec=0.3,
            ),
            AutoRemovalCandidate(
                word_id="w_2", text="I", reason=RemovalReason.REPEAT,
                confidence=0.9, start_sec=0.6, end_sec=0.9,
            ),
        ]
        # Only remove fillers
        retained, applied = apply_removals(
            words, candidates, threshold=0.7, categories=["filler"]
        )
        assert len(applied) == 1
        assert applied[0].reason == RemovalReason.FILLER
        assert len(retained) == 2  # w_1 and w_2 kept


class TestMergeRetainedSegments:
    def test_adjacent_words(self) -> None:
        words = [
            _w("w_0", 0.0, 0.3, "hello"),
            _w("w_1", 0.35, 0.6, "world"),  # 0.05s gap, within threshold
        ]
        segments = merge_retained_segments(words)
        assert len(segments) == 1
        assert segments[0].start_sec == 0.0
        assert segments[0].end_sec == 0.6

    def test_gap_creates_new_segment(self) -> None:
        words = [
            _w("w_0", 0.0, 1.0, "hello"),
            _w("w_1", 3.0, 4.0, "world"),  # 2.0s gap
        ]
        segments = merge_retained_segments(words)
        assert len(segments) == 2
        assert segments[0].end_sec == 1.0
        assert segments[1].start_sec == 3.0

    def test_empty(self) -> None:
        assert merge_retained_segments([]) == []

    def test_single_word(self) -> None:
        words = [_w("w_0", 1.0, 2.0, "hello")]
        segments = merge_retained_segments(words)
        assert len(segments) == 1
        assert segments[0].start_sec == 1.0

    def test_filters_synthetic_tokens(self) -> None:
        words = [
            _w("w_0", 0.0, 1.0, "hello"),
            _w("pause_0", 1.0, 2.0, "[PAUSE]"),
            _w("w_1", 3.0, 4.0, "world"),
        ]
        segments = merge_retained_segments(words)
        assert len(segments) == 2
        # Pause token filtered, so segments are based on real words only


class TestBuildCleanupPrompt:
    def test_basic_prompt(self) -> None:
        words = [
            _w("w_0", 0.0, 0.3, "I"),
            _w("w_1", 0.3, 0.6, "I"),
            _w("w_2", 0.6, 1.0, "went"),
        ]
        prompt = build_cleanup_prompt(words)
        assert "w_0" in prompt
        assert "w_1" in prompt
        assert "stutter" in prompt
        assert "repeat" in prompt
        assert "false_start" in prompt
        assert "filler" not in prompt.split("## Word List")[0] or "Do NOT identify filler" in prompt

    def test_skips_synthetic_tokens(self) -> None:
        words = [
            _w("w_0", 0.0, 1.0, "hello"),
            _w("pause_0", 1.0, 2.0, "[PAUSE]"),
            _w("w_1", 2.0, 3.0, "world"),
        ]
        prompt = build_cleanup_prompt(words)
        assert "[PAUSE]" not in prompt
        assert "pause_0" not in prompt
