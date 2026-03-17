"""Fine-cut engine: normalize, detect fillers/pauses, apply removals, merge segments."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from autoclip.models import (
    AnalysisCandidate,
    AutoRemovalCandidate,
    CaptionSegment,
    RemovalReason,
    Segment,
    WordToken,
)

logger = logging.getLogger(__name__)

# Language-specific filler word lists (unambiguous only)
FILLER_WORDS: dict[str, set[str]] = {
    "en": {"um", "uh", "uh-huh", "uh huh", "uhh", "umm", "hmm", "hm", "mm"},
    "zh": {"嗯", "啊", "呃", "额", "唔", "嗯嗯", "啊啊"},
}

# Merge gap threshold for retained segments
MERGE_GAP_SEC = 0.1


def normalize_whisper_words(segments: list[CaptionSegment]) -> list[WordToken]:
    """Convert CaptionSegments into a flat list of WordTokens with sequential IDs.

    Args:
        segments: CaptionSegment list from ASR provider.

    Returns:
        Flat list of WordTokens with IDs w_0, w_1, ...
    """
    tokens: list[WordToken] = []
    idx = 0
    for seg in segments:
        for word in seg.words:
            tokens.append(
                WordToken(
                    id=f"w_{idx}",
                    start_sec=word.start_sec,
                    end_sec=word.end_sec,
                    text=word.text,
                    probability=word.probability,
                )
            )
            idx += 1
    return tokens


def detect_fillers(
    words: list[WordToken],
    language: str = "en",
) -> list[AnalysisCandidate]:
    """Detect filler words via keyword matching.

    Args:
        words: Normalized word tokens.
        language: Language code for filler word list.

    Returns:
        List of filler removal candidates with confidence=1.0 and source="keyword".
    """
    # Combine all known filler sets (fallback to en)
    filler_set = FILLER_WORDS.get(language, FILLER_WORDS["en"])

    candidates: list[AnalysisCandidate] = []
    for word in words:
        normalized_text = word.text.lower().strip()
        if normalized_text in filler_set:
            candidates.append(
                AnalysisCandidate(
                    word_id=word.id,
                    text=word.text,
                    reason=RemovalReason.FILLER,
                    confidence=1.0,
                    start_sec=word.start_sec,
                    end_sec=word.end_sec,
                    source="keyword",
                )
            )

    return candidates


def detect_pauses(
    words: list[WordToken],
    long_pause_ms: int = 500,
) -> tuple[list[WordToken], list[AnalysisCandidate]]:
    """Detect long pauses between words and insert synthetic [PAUSE] tokens.

    Args:
        words: Normalized word tokens.
        long_pause_ms: Minimum gap in ms to be considered a long pause.

    Returns:
        Tuple of (words with pause tokens inserted, pause removal candidates).
    """
    if len(words) < 2:
        return list(words), []

    threshold_sec = long_pause_ms / 1000.0
    result_words: list[WordToken] = [words[0]]
    candidates: list[AnalysisCandidate] = []
    pause_idx = 0

    for i in range(1, len(words)):
        gap = words[i].start_sec - words[i - 1].end_sec
        if gap >= threshold_sec:
            pause_id = f"pause_{pause_idx}"
            pause_token = WordToken(
                id=pause_id,
                start_sec=words[i - 1].end_sec,
                end_sec=words[i].start_sec,
                text="[PAUSE]",
            )
            result_words.append(pause_token)
            candidates.append(
                AnalysisCandidate(
                    word_id=pause_id,
                    text="[PAUSE]",
                    reason=RemovalReason.LONG_PAUSE,
                    confidence=1.0,
                    start_sec=words[i - 1].end_sec,
                    end_sec=words[i].start_sec,
                    source="heuristic",
                )
            )
            pause_idx += 1
        result_words.append(words[i])

    return result_words, candidates


def parse_cleanup_response(response: str) -> list[AnalysisCandidate]:
    """Parse LLM cleanup response into removal candidates.

    Expected format: JSON array of objects with word_id, reason, confidence.

    Args:
        response: Raw LLM response text.

    Returns:
        List of AnalysisCandidate with source="llm". Empty on parse failure (fail-safe).
    """
    # Try to extract JSON from response (may be wrapped in markdown code blocks)
    text = response.strip()
    if "```" in text:
        # Extract content between code fences
        parts = text.split("```")
        for part in parts[1:]:
            # Skip language identifier line
            lines = part.strip().split("\n", 1)
            if len(lines) > 1:
                text = lines[1].strip()
                break
            else:
                text = lines[0].strip()
                break

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON, returning empty candidates")
        return []

    if not isinstance(data, list):
        logger.warning("LLM response is not a JSON array, returning empty candidates")
        return []

    # Map LLM reason strings to internal RemovalReason
    reason_map: dict[str, RemovalReason] = {
        "stutter": RemovalReason.STUTTER,
        "repeat": RemovalReason.REPEAT,
        "false_start": RemovalReason.FALSE_START,
    }

    candidates: list[AnalysisCandidate] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        word_id = item.get("word_id", "")
        reason_str = item.get("reason", "")
        confidence = item.get("confidence", 0.0)

        reason = reason_map.get(reason_str)
        if reason is None:
            logger.debug("Unknown LLM reason '%s' for word %s, skipping", reason_str, word_id)
            continue

        try:
            candidates.append(
                AnalysisCandidate(
                    word_id=str(word_id),
                    text=item.get("text", ""),
                    reason=reason,
                    confidence=float(confidence),
                    start_sec=float(item.get("start_sec", 0.0)),
                    end_sec=float(item.get("end_sec", 0.0)),
                    source="llm",
                )
            )
        except (ValueError, TypeError) as e:
            logger.debug("Skipping malformed candidate: %s", e)
            continue

    return candidates


def apply_removals(
    words: list[WordToken],
    candidates: Sequence[AutoRemovalCandidate],
    threshold: float = 0.7,
    categories: list[str] | None = None,
) -> tuple[list[WordToken], list[AutoRemovalCandidate]]:
    """Filter candidates by threshold and categories, return retained words and applied removals.

    false-start uses max(threshold, 0.85) as its effective threshold.

    Args:
        words: All word tokens (including pause tokens).
        candidates: All removal candidates (filler + pause + LLM).
        threshold: User confidence threshold.
        categories: CLI categories to filter. None means all.

    Returns:
        Tuple of (retained words, applied removals).
    """
    from autoclip.models import CLI_TO_INTERNAL

    # Determine which internal reasons are active
    active_reasons: set[RemovalReason] = set()
    if categories is None:
        active_reasons = set(RemovalReason)
    else:
        for cat in categories:
            for reason in CLI_TO_INTERNAL.get(cat, []):
                active_reasons.add(reason)

    # Filter candidates
    applied: list[AutoRemovalCandidate] = []
    removal_ids: set[str] = set()

    for c in candidates:
        if c.reason not in active_reasons:
            continue

        # Determine effective threshold
        if c.reason == RemovalReason.FALSE_START:
            effective_threshold = max(threshold, 0.85)
        else:
            effective_threshold = threshold

        if c.confidence >= effective_threshold:
            applied.append(c)
            removal_ids.add(c.word_id)

    # Filter words
    retained = [w for w in words if w.id not in removal_ids]

    return retained, applied


def merge_retained_segments(words: list[WordToken]) -> list[Segment]:
    """Merge consecutive retained words into Segments.

    Words with gap <= MERGE_GAP_SEC are merged into a single segment.

    Args:
        words: Retained word tokens (after removal).

    Returns:
        List of Segment objects for export.
    """
    if not words:
        return []

    # Filter out synthetic pause tokens
    real_words = [w for w in words if not w.text.startswith("[")]

    if not real_words:
        return []

    segments: list[Segment] = []
    seg_start = real_words[0].start_sec
    seg_end = real_words[0].end_sec

    for i in range(1, len(real_words)):
        gap = real_words[i].start_sec - seg_end
        if gap <= MERGE_GAP_SEC:
            seg_end = real_words[i].end_sec
        else:
            segments.append(Segment(start_sec=seg_start, end_sec=seg_end))
            seg_start = real_words[i].start_sec
            seg_end = real_words[i].end_sec

    segments.append(Segment(start_sec=seg_start, end_sec=seg_end))

    return segments
