"""Protocol definitions for ASR and LLM providers."""

from __future__ import annotations

from typing import Protocol

from autoclip.models import CaptionSegment


class ASRProvider(Protocol):
    """Protocol for ASR (Automatic Speech Recognition) providers."""

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> tuple[list[CaptionSegment], str]:
        """Transcribe audio with word-level timestamps.

        Args:
            audio_path: Path to WAV audio file (16kHz mono).
            language: Language code or None for auto-detection.

        Returns:
            Tuple of (list of CaptionSegments, detected language code).
        """
        ...


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """Send a prompt and get a text response.

        Args:
            prompt: The prompt text.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            The LLM's text response.
        """
        ...
