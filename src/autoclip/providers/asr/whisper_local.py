"""faster-whisper ASR provider with VAD and hallucination filtering."""

from __future__ import annotations

import logging
from typing import Any

from autoclip.models import CaptionSegment, WordToken

logger = logging.getLogger(__name__)


class WhisperLocalProvider:
    """ASR provider using faster-whisper with word timestamps."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        vad_filter: bool = True,
        min_silence_duration_ms: int = 500,
        hallucination_threshold: float = 0.9,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._min_silence_duration_ms = min_silence_duration_ms
        self._hallucination_threshold = hallucination_threshold
        self._model: Any = None

    def _get_model(self) -> Any:
        """Lazy-load the whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading whisper model '%s' (this may take a moment on first run)...",
                self._model_size,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> tuple[list[CaptionSegment], str]:
        """Transcribe audio with word-level timestamps.

        Args:
            audio_path: Path to WAV audio file.
            language: Language code or None/"auto" for auto-detection.

        Returns:
            Tuple of (CaptionSegment list, detected language).
        """
        model = self._get_model()

        lang_arg = None if language in (None, "auto") else language

        vad_params = (
            {"min_silence_duration_ms": self._min_silence_duration_ms}
            if self._vad_filter
            else None
        )

        segments_iter, info = model.transcribe(
            audio_path,
            language=lang_arg,
            beam_size=self._beam_size,
            word_timestamps=True,
            vad_filter=self._vad_filter,
            vad_parameters=vad_params,
        )

        detected_language = str(info.language)
        raw_segments = list(segments_iter)

        # Post-filter: remove hallucinations
        filtered = _filter_hallucinations(
            raw_segments, no_speech_threshold=self._hallucination_threshold,
        )

        # Convert to our models
        caption_segments = _to_caption_segments(filtered)

        logger.info(
            "Transcribed %d segments, language=%s",
            len(caption_segments),
            detected_language,
        )

        return caption_segments, detected_language


def _filter_hallucinations(
    segments: list[Any],
    no_speech_threshold: float = 0.9,
) -> list[Any]:
    """Remove hallucinated segments: duplicates and high no_speech_prob.

    Args:
        segments: Raw faster-whisper segments.
        no_speech_threshold: Segments with no_speech_prob above this are dropped.
            Default 0.9 — the VAD filter already handles non-speech; this only
            catches near-certain hallucinations.
    """
    filtered: list[Any] = []
    prev_text: str | None = None

    for seg in segments:
        text = seg.text.strip()
        no_speech_prob = getattr(seg, "no_speech_prob", 0.0)

        # Filter: high no-speech probability
        if no_speech_prob > no_speech_threshold:
            logger.info(
                "Filtered hallucination [%.1fs-%.1fs] no_speech_prob=%.2f: %s",
                seg.start, seg.end, no_speech_prob, text,
            )
            continue

        # Log segments that would have been filtered by the old 0.6 threshold
        if no_speech_prob > 0.6:
            logger.debug(
                "Kept segment [%.1fs-%.1fs] no_speech_prob=%.2f (above old 0.6 threshold): %s",
                seg.start, seg.end, no_speech_prob, text,
            )

        # Filter: consecutive duplicate text
        if text == prev_text:
            logger.debug("Filtered duplicate segment: %s", text)
            continue

        filtered.append(seg)
        prev_text = text

    return filtered


def _to_caption_segments(segments: list[Any]) -> list[CaptionSegment]:
    """Convert faster-whisper segments to CaptionSegment models."""
    result: list[CaptionSegment] = []

    for seg in segments:
        words_raw = getattr(seg, "words", None) or []
        word_tokens = tuple(
            WordToken(
                id="",  # IDs assigned during normalization
                start_sec=float(w.start),
                end_sec=float(w.end),
                text=w.word.strip(),
                probability=float(getattr(w, "probability", 1.0)),
            )
            for w in words_raw
            if w.word.strip()
        )

        result.append(
            CaptionSegment(
                start_sec=float(seg.start),
                end_sec=float(seg.end),
                text=seg.text.strip(),
                words=word_tokens,
                no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0)),
            )
        )

    return result
