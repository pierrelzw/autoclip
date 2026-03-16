"""Provider factory for ASR and LLM providers."""

from __future__ import annotations

import importlib
import logging
import platform

from autoclip.config import AppConfig
from autoclip.providers.types import ASRProvider, LLMProvider

logger = logging.getLogger(__name__)


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (macOS ARM64)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _is_installed(module: str) -> bool:
    """Check if a Python module is importable."""
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def _create_whisper_provider(config: AppConfig) -> ASRProvider:
    """Create a faster-whisper provider."""
    from autoclip.providers.asr.whisper_local import WhisperLocalProvider

    asr_cfg = config.asr
    return WhisperLocalProvider(
        model_size=asr_cfg.model,
        beam_size=asr_cfg.beam_size,
        vad_filter=asr_cfg.vad_filter,
        min_silence_duration_ms=asr_cfg.min_silence_duration_ms,
    )


def _create_mlx_whisper_provider(config: AppConfig) -> ASRProvider:
    """Create an mlx-whisper provider."""
    from autoclip.providers.asr.mlx_whisper_local import MLXWhisperProvider

    asr_cfg = config.asr
    return MLXWhisperProvider(
        model_size=asr_cfg.model,
        beam_size=asr_cfg.beam_size,
        vad_filter=asr_cfg.vad_filter,
        min_silence_duration_ms=asr_cfg.min_silence_duration_ms,
    )


def create_asr_provider(config: AppConfig) -> ASRProvider:
    """Create an ASR provider based on configuration.

    Provider selection:
        - "whisper": faster-whisper (requires faster-whisper package)
        - "mlx-whisper": mlx-whisper (requires mlx-whisper on Apple Silicon)
        - "auto": picks mlx-whisper on Apple Silicon if available, else faster-whisper

    Args:
        config: Application configuration.

    Returns:
        An ASRProvider instance.

    Raises:
        ValueError: If the provider is not supported or required package is missing.
    """
    asr_cfg = config.asr
    provider = asr_cfg.provider

    if provider == "whisper":
        if not _is_installed("faster_whisper"):
            raise ValueError(
                "faster-whisper is not installed. "
                "Install it with: pip install 'autoclip[whisper]'"
            )
        return _create_whisper_provider(config)

    if provider == "mlx-whisper":
        if not _is_installed("mlx_whisper"):
            raise ValueError(
                "mlx-whisper is not installed. "
                "Install it with: pip install 'autoclip[mlx-whisper]'"
            )
        return _create_mlx_whisper_provider(config)

    if provider == "auto":
        # Prefer mlx-whisper on Apple Silicon
        if _is_apple_silicon() and _is_installed("mlx_whisper"):
            logger.info("Auto-selected mlx-whisper (Apple Silicon detected)")
            return _create_mlx_whisper_provider(config)

        if _is_installed("faster_whisper"):
            logger.info("Auto-selected faster-whisper")
            return _create_whisper_provider(config)

        raise ValueError(
            "No ASR engine available. Install one of:\n"
            "  pip install 'autoclip[whisper]'       # faster-whisper\n"
            "  pip install 'autoclip[mlx-whisper]'   # Apple Silicon only"
        )

    raise ValueError(
        f"Unknown ASR provider: '{provider}'. "
        "Available: whisper, mlx-whisper, auto"
    )


def create_llm_provider(config: AppConfig) -> LLMProvider:
    """Create an LLM provider based on configuration.

    Args:
        config: Application configuration.

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    llm_cfg = config.llm

    if llm_cfg.provider == "ollama":
        from autoclip.providers.llm.ollama_local import OllamaProvider

        return OllamaProvider(
            model=llm_cfg.model,
            base_url=llm_cfg.base_url,
        )

    if llm_cfg.provider == "openai":
        from autoclip.providers.llm.openai_cloud import OpenAIProvider

        return OpenAIProvider(
            model=llm_cfg.model,
            api_key=llm_cfg.api_key,
        )

    raise ValueError(
        f"Unknown LLM provider: '{llm_cfg.provider}'. Available: ollama, openai"
    )
