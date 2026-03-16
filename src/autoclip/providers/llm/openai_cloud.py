"""OpenAI cloud LLM provider."""

from __future__ import annotations

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """LLM provider using the OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = "",
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key is required. Set it via:\n"
                "  - Config: llm.api_key in autoclip.yaml\n"
                "  - Environment: OPENAI_API_KEY"
            )

        self._model = model
        self._client = OpenAI(api_key=resolved_key)

    def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """Send prompt to OpenAI and return response text.

        Args:
            prompt: The prompt text.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            The LLM response text.

        Raises:
            RuntimeError: If the API call fails.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

        content = response.choices[0].message.content
        return content or ""
