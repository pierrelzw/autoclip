"""mlx-whisper ASR provider for Apple Silicon Macs."""

from __future__ import annotations

import logging
from typing import Any

from autoclip.models import CaptionSegment, WordToken

logger = logging.getLogger(__name__)

# Map short model names to mlx-community HuggingFace repos
_MODEL_NAME_MAP: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def _resolve_model_repo(model_name: str) -> str:
    """Resolve a short model name to an mlx-community repo path.

    If the name contains '/', it is treated as a full HuggingFace repo path.
    Otherwise, it is mapped via the built-in table.
    """
    if "/" in model_name:
        return model_name
    return _MODEL_NAME_MAP.get(model_name, f"mlx-community/whisper-{model_name}-mlx")


class MLXWhisperProvider:
    """ASR provider using mlx-whisper on Apple Silicon."""

    def __init__(
        self,
        model_size: str = "large-v3",
        beam_size: int = 5,
        vad_filter: bool = True,
        min_silence_duration_ms: int = 500,
        hallucination_threshold: float = 0.9,
    ) -> None:
        self._model_size = model_size
        self._model_repo = _resolve_model_repo(model_size)
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._min_silence_duration_ms = min_silence_duration_ms
        self._hallucination_threshold = hallucination_threshold
        self._loaded = False

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
        import mlx_whisper

        if not self._loaded:
            logger.info(
                "Loading mlx-whisper model '%s' (this may take a moment on first run)...",
                self._model_repo,
            )
            self._loaded = True

        lang_arg = None if language in (None, "auto") else language

        kwargs: dict[str, Any] = {
            "path_or_hf_repo": self._model_repo,
            "word_timestamps": True,
        }
        if lang_arg is not None:
            kwargs["language"] = lang_arg

        result = mlx_whisper.transcribe(audio_path, **kwargs)

        detected_language = str(result.get("language", "unknown"))
        raw_segments: list[dict[str, Any]] = result.get("segments", [])

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
    segments: list[dict[str, Any]],
    no_speech_threshold: float = 0.9,
) -> list[dict[str, Any]]:
    """Remove hallucinated segments: duplicates and high no_speech_prob.

    Args:
        segments: Raw mlx-whisper segment dicts.
        no_speech_threshold: Segments with no_speech_prob above this are dropped.
            Default 0.9 — only catches near-certain hallucinations.
    """
    filtered: list[dict[str, Any]] = []
    prev_text: str | None = None

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        no_speech_prob = float(seg.get("no_speech_prob", 0.0))
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))

        # Filter: high no-speech probability
        if no_speech_prob > no_speech_threshold:
            logger.info(
                "Filtered hallucination [%.1fs-%.1fs] no_speech_prob=%.2f: %s",
                start, end, no_speech_prob, text,
            )
            continue

        # Log segments that would have been filtered by the old 0.6 threshold
        if no_speech_prob > 0.6:
            logger.debug(
                "Kept segment [%.1fs-%.1fs] no_speech_prob=%.2f (above old 0.6 threshold): %s",
                start, end, no_speech_prob, text,
            )

        # Filter: consecutive duplicate text
        if text == prev_text:
            logger.debug("Filtered duplicate segment: %s", text)
            continue

        filtered.append(seg)
        prev_text = text

    return filtered


def _to_caption_segments(segments: list[dict[str, Any]]) -> list[CaptionSegment]:
    """Convert mlx-whisper segment dicts to CaptionSegment models."""
    result: list[CaptionSegment] = []

    for seg in segments:
        words_raw: list[dict[str, Any]] = seg.get("words", [])
        word_tokens = tuple(
            WordToken(
                id="",  # IDs assigned during normalization
                start_sec=float(w.get("start", 0.0)),
                end_sec=float(w.get("end", 0.0)),
                text=str(w.get("word", "")).strip(),
                probability=float(w.get("probability", 1.0)),
            )
            for w in words_raw
            if str(w.get("word", "")).strip()
        )

        result.append(
            CaptionSegment(
                start_sec=float(seg.get("start", 0.0)),
                end_sec=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")).strip(),
                words=word_tokens,
                no_speech_prob=float(seg.get("no_speech_prob", 0.0)),
            )
        )

    return result
