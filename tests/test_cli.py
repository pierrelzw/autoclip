"""Tests for CLI integration."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from autoclip.cli import (
    _build_clean_result,
    _check_dependencies,
    main,
)
from autoclip.config import AppConfig
from autoclip.models import (
    AutoRemovalCandidate,
    RemovalReason,
    Segment,
)


class TestBuildCleanResult:
    def test_basic_result(self) -> None:
        segments = [Segment(start_sec=0.0, end_sec=5.0), Segment(start_sec=6.0, end_sec=10.0)]
        removals = [
            AutoRemovalCandidate(
                word_id="w_0", text="um", reason=RemovalReason.FILLER,
                confidence=1.0, start_sec=5.0, end_sec=5.5,
            ),
            AutoRemovalCandidate(
                word_id="w_1", text="uh", reason=RemovalReason.FILLER,
                confidence=1.0, start_sec=5.5, end_sec=6.0,
            ),
        ]
        result = _build_clean_result("test.mp4", 12.0, segments, removals, "en")
        assert result.source == "test.mp4"
        assert result.original_duration_sec == 12.0
        assert result.cleaned_duration_sec == 9.0
        assert result.reduction_percent == 25.0
        assert result.removal_counts == {"filler": 2}
        assert result.detected_language == "en"
        assert len(result.removals) == 2

    def test_stutter_and_repeat_aggregated(self) -> None:
        segments = [Segment(start_sec=0.0, end_sec=5.0)]
        removals = [
            AutoRemovalCandidate(
                word_id="w_0", text="the", reason=RemovalReason.STUTTER,
                confidence=0.9, start_sec=0.0, end_sec=0.2,
            ),
            AutoRemovalCandidate(
                word_id="w_1", text="the", reason=RemovalReason.REPEAT,
                confidence=0.85, start_sec=0.2, end_sec=0.4,
            ),
        ]
        result = _build_clean_result("test.mp4", 10.0, segments, removals, "en")
        # Both stutter and repeat map to CLI "repeat"
        assert result.removal_counts == {"repeat": 2}

    def test_zero_duration(self) -> None:
        result = _build_clean_result("test.mp4", 0.0, [], [], "en")
        assert result.reduction_percent == 0.0


class TestCheckDependencies:
    @patch("autoclip.cli.check_ffmpeg")
    def test_passes_when_available(self, mock_check: MagicMock) -> None:
        config = AppConfig(llm=AppConfig.model_fields["llm"].default.model_copy(
            update={"provider": "openai"}
        ))
        _check_dependencies(config)
        mock_check.assert_called_once()

    @patch("autoclip.cli.check_ffmpeg")
    def test_raises_when_missing(self, mock_check: MagicMock) -> None:
        mock_check.side_effect = FileNotFoundError("ffmpeg not found")
        config = AppConfig()
        with pytest.raises(FileNotFoundError):
            _check_dependencies(config)


class TestCLICleanCommand:
    @patch("autoclip.cli._check_dependencies")
    @patch("autoclip.cli.probe_video")
    @patch("autoclip.cli.extract_audio")
    @patch("autoclip.cli.create_asr_provider")
    @patch("autoclip.cli.create_llm_provider")
    def test_no_speech_exit(
        self,
        mock_llm: MagicMock,
        mock_asr: MagicMock,
        mock_extract: MagicMock,
        mock_probe: MagicMock,
        mock_deps: MagicMock,
        tmp_path: object,
    ) -> None:
        # Create a dummy video file
        video = os.path.join(str(tmp_path), "test.mp4")
        with open(video, "w") as f:
            f.write("dummy")

        mock_probe.return_value = SimpleNamespace(
            duration_sec=10.0, video_codec="h264", audio_codec="aac",
            width=1920, height=1080, fps=30.0,
        )
        mock_extract.return_value = "/tmp/audio.wav"
        mock_asr_instance = MagicMock()
        mock_asr_instance.transcribe.return_value = ([], "en")
        mock_asr.return_value = mock_asr_instance

        runner = CliRunner()
        result = runner.invoke(main, ["clean", video])
        assert "No speech detected" in result.output

    @patch("autoclip.cli._check_dependencies")
    @patch("autoclip.cli.probe_video")
    @patch("autoclip.cli.extract_audio")
    @patch("autoclip.cli.create_asr_provider")
    @patch("autoclip.cli.create_llm_provider")
    def test_no_disfluencies_exit(
        self,
        mock_llm: MagicMock,
        mock_asr: MagicMock,
        mock_extract: MagicMock,
        mock_probe: MagicMock,
        mock_deps: MagicMock,
        tmp_path: object,
    ) -> None:
        from autoclip.models import CaptionSegment, WordToken

        video = os.path.join(str(tmp_path), "test.mp4")
        with open(video, "w") as f:
            f.write("dummy")

        mock_probe.return_value = SimpleNamespace(
            duration_sec=10.0, video_codec="h264", audio_codec="aac",
            width=1920, height=1080, fps=30.0,
        )
        mock_extract.return_value = "/tmp/audio.wav"

        # Return a clean transcript (no fillers/pauses/disfluencies)
        words = (
            WordToken(id="", start_sec=0.0, end_sec=0.5, text="hello"),
            WordToken(id="", start_sec=0.5, end_sec=1.0, text="world"),
        )
        seg = CaptionSegment(
            start_sec=0.0, end_sec=1.0, text="hello world", words=words
        )
        mock_asr_instance = MagicMock()
        mock_asr_instance.transcribe.return_value = ([seg], "en")
        mock_asr.return_value = mock_asr_instance

        mock_llm_instance = MagicMock()
        mock_llm_instance.complete.return_value = "[]"
        mock_llm.return_value = mock_llm_instance

        runner = CliRunner()
        result = runner.invoke(main, ["clean", video])
        assert "already clean" in result.output

    @patch("autoclip.cli._check_dependencies")
    @patch("autoclip.cli.probe_video")
    @patch("autoclip.cli.extract_audio")
    @patch("autoclip.cli.create_asr_provider")
    @patch("autoclip.cli.create_llm_provider")
    def test_preview_mode(
        self,
        mock_llm: MagicMock,
        mock_asr: MagicMock,
        mock_extract: MagicMock,
        mock_probe: MagicMock,
        mock_deps: MagicMock,
        tmp_path: object,
    ) -> None:
        from autoclip.models import CaptionSegment, WordToken

        video = os.path.join(str(tmp_path), "test.mp4")
        with open(video, "w") as f:
            f.write("dummy")

        mock_probe.return_value = SimpleNamespace(
            duration_sec=10.0, video_codec="h264", audio_codec="aac",
            width=1920, height=1080, fps=30.0,
        )
        mock_extract.return_value = "/tmp/audio.wav"

        words = (
            WordToken(id="", start_sec=0.0, end_sec=0.3, text="um"),
            WordToken(id="", start_sec=0.3, end_sec=0.6, text="hello"),
            WordToken(id="", start_sec=0.6, end_sec=1.0, text="world"),
        )
        seg = CaptionSegment(
            start_sec=0.0, end_sec=1.0, text="um hello world", words=words
        )
        mock_asr_instance = MagicMock()
        mock_asr_instance.transcribe.return_value = ([seg], "en")
        mock_asr.return_value = mock_asr_instance

        mock_llm_instance = MagicMock()
        mock_llm_instance.complete.return_value = "[]"
        mock_llm.return_value = mock_llm_instance

        runner = CliRunner()
        result = runner.invoke(main, ["clean", video, "--preview"])
        assert "Disfluency Analysis" in result.output
        assert "Summary" in result.output

    def test_invalid_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["clean", "/nonexistent/video.mp4"])
        assert result.exit_code != 0

    @patch("autoclip.cli._check_dependencies")
    @patch("autoclip.cli.probe_video")
    @patch("autoclip.cli.extract_audio")
    @patch("autoclip.cli.create_asr_provider")
    @patch("autoclip.cli.create_llm_provider")
    @patch("autoclip.cli.export_clean_video")
    def test_full_export(
        self,
        mock_export: MagicMock,
        mock_llm: MagicMock,
        mock_asr: MagicMock,
        mock_extract: MagicMock,
        mock_probe: MagicMock,
        mock_deps: MagicMock,
        tmp_path: object,
    ) -> None:
        from autoclip.models import CaptionSegment, WordToken

        video = os.path.join(str(tmp_path), "test.mp4")
        with open(video, "w") as f:
            f.write("dummy")

        out_dir = os.path.join(str(tmp_path), "output")

        mock_probe.return_value = SimpleNamespace(
            duration_sec=10.0, video_codec="h264", audio_codec="aac",
            width=1920, height=1080, fps=30.0,
        )
        mock_extract.return_value = "/tmp/audio.wav"

        words = (
            WordToken(id="", start_sec=0.0, end_sec=0.3, text="um"),
            WordToken(id="", start_sec=0.3, end_sec=0.6, text="hello"),
            WordToken(id="", start_sec=0.6, end_sec=1.0, text="world"),
        )
        seg = CaptionSegment(
            start_sec=0.0, end_sec=1.0, text="um hello world", words=words
        )
        mock_asr_instance = MagicMock()
        mock_asr_instance.transcribe.return_value = ([seg], "en")
        mock_asr.return_value = mock_asr_instance

        mock_llm_instance = MagicMock()
        mock_llm_instance.complete.return_value = "[]"
        mock_llm.return_value = mock_llm_instance

        mock_export.return_value = os.path.join(out_dir, "test_clean.mp4")

        runner = CliRunner()
        result = runner.invoke(main, ["clean", video, "-o", out_dir])
        assert "Export complete" in result.output
        # JSON report should be written
        json_path = os.path.join(out_dir, "test_clean.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            report = json.load(f)
        assert report["source"] == video
        assert "filler" in report["removal_counts"]

    def test_invalid_category(self) -> None:
        runner = CliRunner()
        # Use a real file to get past file-not-found check
        result = runner.invoke(main, ["clean", "/dev/null", "--categories", "nonexistent"])
        assert "Unknown category" in result.output


class TestCLIVersion:
    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert "0.1.0" in result.output
