"""Tests for LLM providers with mocked responses."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoclip.config import AppConfig
from autoclip.providers.registry import create_llm_provider


class TestOllamaProvider:
    @patch("autoclip.providers.llm.ollama_local.OpenAI")
    def test_successful_completion(self, mock_openai_cls: MagicMock) -> None:
        from autoclip.providers.llm.ollama_local import OllamaProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '[{"word_id": "w_0", "reason": "stutter"}]'
        mock_client.chat.completions.create.return_value = mock_response

        provider = OllamaProvider(model="qwen2.5:7b-instruct")
        result = provider.complete("test prompt")
        assert "stutter" in result

    @patch("autoclip.providers.llm.ollama_local.OpenAI")
    def test_connection_failure(self, mock_openai_cls: MagicMock) -> None:
        from autoclip.providers.llm.ollama_local import OllamaProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")

        provider = OllamaProvider()
        with pytest.raises(RuntimeError, match="Ollama request failed"):
            provider.complete("test")


class TestOpenAIProvider:
    @patch("autoclip.providers.llm.openai_cloud.OpenAI")
    def test_successful_completion(self, mock_openai_cls: MagicMock) -> None:
        from autoclip.providers.llm.openai_cloud import OpenAIProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[]"
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(model="gpt-4o-mini", api_key="test-key")
        result = provider.complete("test prompt")
        assert result == "[]"

    def test_missing_api_key(self) -> None:
        from autoclip.providers.llm.openai_cloud import OpenAIProvider

        with patch.dict("os.environ", {}, clear=True), pytest.raises(ValueError, match="API key is required"):
            OpenAIProvider(api_key="")

    @patch("autoclip.providers.llm.openai_cloud.OpenAI")
    def test_api_failure(self, mock_openai_cls: MagicMock) -> None:
        from autoclip.providers.llm.openai_cloud import OpenAIProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Rate limited")

        provider = OpenAIProvider(api_key="test-key")
        with pytest.raises(RuntimeError, match="API call failed"):
            provider.complete("test")


class TestRegistry:
    def test_create_ollama(self) -> None:
        config = AppConfig()
        with patch("autoclip.providers.llm.ollama_local.OpenAI"):
            provider = create_llm_provider(config)
        assert provider is not None

    def test_create_openai(self) -> None:
        config = AppConfig(llm=AppConfig.model_fields["llm"].default.model_copy(
            update={"provider": "openai", "api_key": "test-key"}
        ))
        with patch("autoclip.providers.llm.openai_cloud.OpenAI"):
            provider = create_llm_provider(config)
        assert provider is not None

    def test_unknown_provider(self) -> None:
        config = AppConfig(llm=AppConfig.model_fields["llm"].default.model_copy(
            update={"provider": "unknown"}
        ))
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider(config)
