"""Ollama LLM provider via OpenAI-compatible endpoint."""

from __future__ import annotations

import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class OllamaProvider:
    """LLM provider using Ollama through its OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434/v1",
    ) -> None:
        self._model = model
        self._client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollama doesn't require a real key
        )

    def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """Send prompt to Ollama and return response text.

        Args:
            prompt: The prompt text.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            The LLM response text.

        Raises:
            RuntimeError: If Ollama is unreachable or returns an error.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise RuntimeError(
                f"Ollama request failed. Is Ollama running? "
                f"Start with: ollama serve\n"
                f"Error: {e}"
            ) from e

        content = response.choices[0].message.content
        return content or ""
