"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from autoclip.config import (
    AppConfig,
    ASRConfig,
    CleanConfig,
    LLMConfig,
    OutputConfig,
    _deep_merge,
    load_config,
)


class TestDefaults:
    def test_default_asr(self) -> None:
        cfg = ASRConfig()
        assert cfg.provider == "whisper"
        assert cfg.model == "large-v3"
        assert cfg.language == "auto"
        assert cfg.vad_filter is True
        assert cfg.beam_size == 5

    def test_default_llm(self) -> None:
        cfg = LLMConfig()
        assert cfg.provider == "ollama"
        assert cfg.model == "qwen2.5:7b-instruct"
        assert cfg.temperature == 0.1

    def test_default_clean(self) -> None:
        cfg = CleanConfig()
        assert cfg.auto_apply_threshold == 0.7
        assert cfg.long_pause_ms == 500
        assert "filler" in cfg.categories

    def test_default_output(self) -> None:
        cfg = OutputConfig()
        assert cfg.dir == "./output"

    def test_default_app_config(self) -> None:
        cfg = AppConfig()
        assert cfg.asr.model == "large-v3"
        assert cfg.llm.provider == "ollama"


class TestValidation:
    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValidationError):
            CleanConfig(auto_apply_threshold=1.5)

    def test_valid_threshold(self) -> None:
        cfg = CleanConfig(auto_apply_threshold=0.9)
        assert cfg.auto_apply_threshold == 0.9


class TestDeepMerge:
    def test_simple_merge(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"asr": {"model": "large-v3", "language": "auto"}}
        override = {"asr": {"model": "base"}}
        result = _deep_merge(base, override)
        assert result == {"asr": {"model": "base", "language": "auto"}}

    def test_base_not_mutated(self) -> None:
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}


class TestLoadConfig:
    def test_with_cli_overrides(self) -> None:
        overrides = {"llm": {"provider": "openai"}, "clean": {"auto_apply_threshold": 0.9}}
        cfg = load_config(cli_overrides=overrides, project_path=Path("/nonexistent"))
        assert cfg.llm.provider == "openai"
        assert cfg.clean.auto_apply_threshold == 0.9

    def test_no_config_files(self) -> None:
        cfg = load_config(project_path=Path("/nonexistent"))
        assert cfg.asr.model == "large-v3"

    def test_with_yaml_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "autoclip.yaml"
        config_file.write_text("asr:\n  model: base\n")
        cfg = load_config(project_path=tmp_path)
        assert cfg.asr.model == "base"
