"""Provider factory for ASR and LLM providers."""

from __future__ import annotations

from autoclip.config import AppConfig
from autoclip.providers.types import ASRProvider, LLMProvider


def create_asr_provider(config: AppConfig) -> ASRProvider:
    """Create an ASR provider based on configuration.

    Args:
        config: Application configuration.

    Returns:
        An ASRProvider instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    asr_cfg = config.asr

    if asr_cfg.provider == "whisper":
        from autoclip.providers.asr.whisper_local import WhisperLocalProvider

        return WhisperLocalProvider(
            model_size=asr_cfg.model,
            beam_size=asr_cfg.beam_size,
            vad_filter=asr_cfg.vad_filter,
            min_silence_duration_ms=asr_cfg.min_silence_duration_ms,
        )

    raise ValueError(
        f"Unknown ASR provider: '{asr_cfg.provider}'. Available: whisper"
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
