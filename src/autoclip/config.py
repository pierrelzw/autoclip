"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ASRConfig(BaseModel, frozen=True):
    """ASR provider configuration.

    Provider options:
        - "whisper": faster-whisper (CPU/CUDA)
        - "mlx-whisper": mlx-whisper (Apple Silicon only)
        - "auto": auto-detect best available engine
    """

    provider: str = "auto"
    model: str = "large-v3"
    language: str = "auto"
    vad_filter: bool = True
    beam_size: int = 5
    min_silence_duration_ms: int = 500


class LLMConfig(BaseModel, frozen=True):
    """LLM provider configuration."""

    provider: str = "ollama"
    model: str = "qwen2.5:7b-instruct"
    base_url: str = "http://localhost:11434/v1"
    temperature: float = 0.1
    api_key: str = ""


class CleanConfig(BaseModel, frozen=True):
    """Cleaning behavior configuration."""

    auto_apply_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    categories: list[str] = Field(
        default_factory=lambda: ["filler", "repeat", "false-start", "pause"]
    )
    long_pause_ms: int = 500


class OutputConfig(BaseModel, frozen=True):
    """Output configuration."""

    dir: str = "./output"


class AppConfig(BaseModel, frozen=True):
    """Top-level application configuration."""

    asr: ASRConfig = ASRConfig()
    llm: LLMConfig = LLMConfig()
    clean: CleanConfig = CleanConfig()
    output: OutputConfig = OutputConfig()


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict if not found."""
    if not path.is_file():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts, override wins on conflicts."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    cli_overrides: dict[str, Any] | None = None,
    project_path: Path | None = None,
) -> AppConfig:
    """Load configuration with precedence: CLI > project > user > defaults.

    Args:
        cli_overrides: Dict of CLI argument overrides (nested structure).
        project_path: Path to project directory for project-level config.

    Returns:
        Validated AppConfig.
    """
    user_config_path = Path.home() / ".config" / "autoclip" / "config.yaml"
    user_data = _load_yaml(user_config_path)

    project_config_path = (project_path or Path.cwd()) / "autoclip.yaml"
    project_data = _load_yaml(project_config_path)

    merged = _deep_merge(user_data, project_data)
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    return AppConfig(**merged)
