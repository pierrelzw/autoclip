"""Tests for analysis report feature: models, persistence, HTML generation."""

from __future__ import annotations

import json
import os
import re
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from autoclip.cli import main
from autoclip.models import (
    ALL_CLI_CATEGORIES,
    AnalysisCandidate,
    AnalysisResult,
    AppliedParams,
    RemovalReason,
    WordToken,
)
from autoclip.processing.finecut import apply_removals, merge_retained_segments
from autoclip.reporting.analysis import write_analysis_json
from autoclip.reporting.html import generate_report_html


def _make_words() -> tuple[WordToken, ...]:
    return (
        WordToken(id="w_0", start_sec=0.0, end_sec=0.3, text="um"),
        WordToken(id="w_1", start_sec=0.3, end_sec=0.6, text="hello"),
        WordToken(id="w_2", start_sec=0.6, end_sec=1.0, text="world"),
    )


def _make_candidates() -> tuple[AnalysisCandidate, ...]:
    return (
        AnalysisCandidate(
            word_id="w_0",
            text="um",
            reason=RemovalReason.FILLER,
            confidence=1.0,
            start_sec=0.0,
            end_sec=0.3,
            source="keyword",
        ),
    )


def _make_analysis_result() -> AnalysisResult:
    return AnalysisResult(
        source="test.mp4",
        original_duration_sec=10.0,
        detected_language="en",
        words=_make_words(),
        candidates=_make_candidates(),
        applied_params=AppliedParams(
            threshold=0.7,
            categories=list(ALL_CLI_CATEGORIES),
        ),
    )


def _make_rich_analysis_result() -> AnalysisResult:
    """Fixture covering filler, pause, and LLM sources + PAUSE token."""
    words = (
        WordToken(id="w_0", start_sec=0.0, end_sec=0.3, text="um"),
        WordToken(id="w_1", start_sec=0.3, end_sec=0.8, text="hello"),
        WordToken(id="w_2", start_sec=0.8, end_sec=1.0, text="hello"),
        WordToken(id="w_3", start_sec=1.0, end_sec=1.5, text="world"),
        WordToken(id="pause_0", start_sec=1.5, end_sec=2.5, text="[PAUSE]"),
        WordToken(id="w_4", start_sec=2.5, end_sec=3.0, text="today"),
    )
    candidates = (
        AnalysisCandidate(
            word_id="w_0", text="um", reason=RemovalReason.FILLER,
            confidence=1.0, start_sec=0.0, end_sec=0.3, source="keyword",
        ),
        AnalysisCandidate(
            word_id="w_2", text="hello", reason=RemovalReason.REPEAT,
            confidence=0.85, start_sec=0.8, end_sec=1.0, source="llm",
        ),
        AnalysisCandidate(
            word_id="pause_0", text="[PAUSE]", reason=RemovalReason.LONG_PAUSE,
            confidence=1.0, start_sec=1.5, end_sec=2.5, source="heuristic",
        ),
    )
    return AnalysisResult(
        source="test.mp4",
        original_duration_sec=10.0,
        detected_language="en",
        words=words,
        candidates=candidates,
        applied_params=AppliedParams(
            threshold=0.7,
            categories=list(ALL_CLI_CATEGORIES),
        ),
    )


class TestAnalysisResultSerialization:
    """Task 6.1: Test AnalysisResult model serialization round-trip."""

    def test_round_trip(self) -> None:
        original = _make_analysis_result()
        json_str = original.model_dump_json()
        restored = AnalysisResult.model_validate_json(json_str)

        assert restored.source == original.source
        assert restored.original_duration_sec == original.original_duration_sec
        assert restored.detected_language == original.detected_language
        assert len(restored.words) == len(original.words)
        assert len(restored.candidates) == len(original.candidates)
        assert restored.applied_params.threshold == original.applied_params.threshold
        assert restored.applied_params.categories == original.applied_params.categories

    def test_candidate_source_preserved(self) -> None:
        original = _make_analysis_result()
        json_str = original.model_dump_json()
        restored = AnalysisResult.model_validate_json(json_str)
        assert restored.candidates[0].source == "keyword"

    def test_json_is_valid(self) -> None:
        result = _make_analysis_result()
        data = json.loads(result.model_dump_json())
        assert data["source"] == "test.mp4"
        assert data["candidates"][0]["source"] == "keyword"


class TestRemovalStatusDerivation:
    """Task 6.2: Test removal status derivation from candidates + applied_params."""

    def test_derivation_matches_pipeline(self) -> None:
        words = list(_make_words())
        candidates = list(_make_candidates())
        params = AppliedParams(
            threshold=0.7,
            categories=list(ALL_CLI_CATEGORIES),
        )

        # Use pipeline function
        _retained, applied = apply_removals(
            words, candidates, threshold=params.threshold, categories=params.categories
        )

        # Derive from candidates + params (same logic)
        applied_ids = {c.word_id for c in applied}
        assert "w_0" in applied_ids
        assert "w_1" not in applied_ids

    def test_threshold_filters_low_confidence(self) -> None:
        words = [
            WordToken(id="w_0", start_sec=0.0, end_sec=0.3, text="the"),
        ]
        candidates = [
            AnalysisCandidate(
                word_id="w_0", text="the", reason=RemovalReason.STUTTER,
                confidence=0.5, start_sec=0.0, end_sec=0.3, source="llm",
            ),
        ]
        _, applied = apply_removals(words, candidates, threshold=0.7)
        assert len(applied) == 0


def _make_cli_test_fixtures(tmp_path: object) -> tuple[str, str]:
    """Create a dummy video file and output dir for CLI tests."""
    video = os.path.join(str(tmp_path), "test.mp4")
    with open(video, "w") as f:
        f.write("dummy")
    out_dir = os.path.join(str(tmp_path), "output")
    return video, out_dir


def _setup_cli_mocks(
    mock_deps: MagicMock,
    mock_probe: MagicMock,
    mock_extract: MagicMock,
    mock_asr: MagicMock,
    mock_llm: MagicMock,
    mock_export: MagicMock | None = None,
    out_dir: str | None = None,
) -> None:
    """Configure common mocks for CLI tests."""
    from autoclip.models import CaptionSegment, WordToken

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

    if mock_export and out_dir:
        mock_export.return_value = os.path.join(out_dir, "test_clean.mp4")


class TestAnalysisJsonExportMode:
    """Task 6.3: Test analysis.json is written in export mode."""

    @patch("autoclip.cli._check_dependencies")
    @patch("autoclip.cli.probe_video")
    @patch("autoclip.cli.extract_audio")
    @patch("autoclip.cli.create_asr_provider")
    @patch("autoclip.cli.create_llm_provider")
    @patch("autoclip.cli.export_clean_video")
    def test_analysis_json_written_on_export(
        self,
        mock_export: MagicMock,
        mock_llm: MagicMock,
        mock_asr: MagicMock,
        mock_extract: MagicMock,
        mock_probe: MagicMock,
        mock_deps: MagicMock,
        tmp_path: object,
    ) -> None:
        video, out_dir = _make_cli_test_fixtures(tmp_path)
        _setup_cli_mocks(mock_deps, mock_probe, mock_extract, mock_asr, mock_llm, mock_export, out_dir)

        runner = CliRunner()
        result = runner.invoke(main, ["clean", video, "-o", out_dir])
        assert result.exit_code == 0

        analysis_path = os.path.join(out_dir, "test_analysis.json")
        assert os.path.exists(analysis_path)
        with open(analysis_path) as f:
            data = json.load(f)
        assert "words" in data
        assert "candidates" in data
        assert "applied_params" in data


class TestAnalysisJsonPreviewMode:
    """Task 6.4: Test analysis.json is written in preview mode."""

    @patch("autoclip.cli._check_dependencies")
    @patch("autoclip.cli.probe_video")
    @patch("autoclip.cli.extract_audio")
    @patch("autoclip.cli.create_asr_provider")
    @patch("autoclip.cli.create_llm_provider")
    def test_analysis_json_written_on_preview(
        self,
        mock_llm: MagicMock,
        mock_asr: MagicMock,
        mock_extract: MagicMock,
        mock_probe: MagicMock,
        mock_deps: MagicMock,
        tmp_path: object,
    ) -> None:
        video, out_dir = _make_cli_test_fixtures(tmp_path)
        _setup_cli_mocks(mock_deps, mock_probe, mock_extract, mock_asr, mock_llm)

        runner = CliRunner()
        result = runner.invoke(main, ["clean", video, "--preview", "-o", out_dir])
        assert result.exit_code == 0

        analysis_path = os.path.join(out_dir, "test_analysis.json")
        assert os.path.exists(analysis_path)


class TestHtmlReportFlag:
    """Task 6.5: Test HTML report generated only when --report flag present."""

    @patch("autoclip.cli._check_dependencies")
    @patch("autoclip.cli.probe_video")
    @patch("autoclip.cli.extract_audio")
    @patch("autoclip.cli.create_asr_provider")
    @patch("autoclip.cli.create_llm_provider")
    @patch("autoclip.cli.export_clean_video")
    def test_no_report_without_flag(
        self,
        mock_export: MagicMock,
        mock_llm: MagicMock,
        mock_asr: MagicMock,
        mock_extract: MagicMock,
        mock_probe: MagicMock,
        mock_deps: MagicMock,
        tmp_path: object,
    ) -> None:
        video, out_dir = _make_cli_test_fixtures(tmp_path)
        _setup_cli_mocks(mock_deps, mock_probe, mock_extract, mock_asr, mock_llm, mock_export, out_dir)

        runner = CliRunner()
        runner.invoke(main, ["clean", video, "-o", out_dir])

        report_path = os.path.join(out_dir, "test_report.html")
        assert not os.path.exists(report_path)

    @patch("autoclip.cli._check_dependencies")
    @patch("autoclip.cli.probe_video")
    @patch("autoclip.cli.extract_audio")
    @patch("autoclip.cli.create_asr_provider")
    @patch("autoclip.cli.create_llm_provider")
    @patch("autoclip.cli.export_clean_video")
    def test_report_with_flag(
        self,
        mock_export: MagicMock,
        mock_llm: MagicMock,
        mock_asr: MagicMock,
        mock_extract: MagicMock,
        mock_probe: MagicMock,
        mock_deps: MagicMock,
        tmp_path: object,
    ) -> None:
        video, out_dir = _make_cli_test_fixtures(tmp_path)
        _setup_cli_mocks(mock_deps, mock_probe, mock_extract, mock_asr, mock_llm, mock_export, out_dir)

        runner = CliRunner()
        result = runner.invoke(main, ["clean", video, "-o", out_dir, "--report"])
        assert result.exit_code == 0

        report_path = os.path.join(out_dir, "test_report.html")
        assert os.path.exists(report_path)
        with open(report_path) as f:
            html = f.read()
        assert "AutoClip Analysis Report" in html


class TestHtmlReportSelfContained:
    """Task 6.6: Test HTML report is self-contained (no external resource references)."""

    def test_no_external_resources(self) -> None:
        analysis = _make_analysis_result()
        html = generate_report_html(analysis, "test.mp4")

        # No external CSS/JS/font links
        assert 'href="http' not in html
        assert 'src="http' not in html
        assert "@import" not in html
        # Has inline style and script
        assert "<style>" in html
        assert "<script>" in html
        # Data is embedded
        assert "const data =" in html

    def test_contains_all_sections(self) -> None:
        analysis = _make_analysis_result()
        html = generate_report_html(analysis, "test.mp4")

        assert "AutoClip Analysis Report" in html
        assert "Transcript" in html
        assert "Timeline" in html
        assert "Video Preview" in html
        assert "Summary" in html


class TestWriteAnalysisJson:
    """Additional tests for write_analysis_json."""

    def test_creates_parent_dirs(self, tmp_path: object) -> None:
        result = _make_analysis_result()
        path = os.path.join(str(tmp_path), "sub", "dir", "analysis.json")
        write_analysis_json(result, path)
        assert os.path.exists(path)

    def test_valid_json_output(self, tmp_path: object) -> None:
        result = _make_analysis_result()
        path = os.path.join(str(tmp_path), "analysis.json")
        write_analysis_json(result, path)
        with open(path) as f:
            data = json.load(f)
        assert data["source"] == "test.mp4"


class TestHtmlJsSyntax:
    """Generated HTML's embedded JS must be syntactically valid."""

    def test_js_syntax_valid_simple_fixture(self) -> None:
        analysis = _make_analysis_result()
        html = generate_report_html(analysis, "test.mp4")
        self._assert_js_valid(html)

    def test_js_syntax_valid_rich_fixture(self) -> None:
        analysis = _make_rich_analysis_result()
        html = generate_report_html(analysis, "test.mp4")
        self._assert_js_valid(html)

    def test_js_syntax_with_tricky_path(self) -> None:
        """Paths with quotes/backslashes must not break JS syntax."""
        analysis = _make_rich_analysis_result()
        html = generate_report_html(analysis, "path/with'quote/video.mp4")
        self._assert_js_valid(html)

    @staticmethod
    def _assert_js_valid(html: str) -> None:
        scripts = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
        assert scripts, "No <script> blocks found in HTML"
        combined_js = "\n".join(scripts)
        result = subprocess.run(
            ["node", "--check"],
            input=combined_js,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"JS syntax error:\n{result.stderr}"


class TestHtmlStatsConsistency:
    """HTML report's derived stats must match pipeline output."""

    def test_reduction_matches_pipeline(self) -> None:
        analysis = _make_rich_analysis_result()

        # Pipeline side: use apply_removals + merge to get real cleaned duration
        retained, _applied = apply_removals(
            list(analysis.words),
            list(analysis.candidates),
            threshold=analysis.applied_params.threshold,
            categories=analysis.applied_params.categories,
        )
        segments = merge_retained_segments(retained)
        pipeline_cleaned = sum(s.end_sec - s.start_sec for s in segments)

        # HTML side: replicate the JS logic in Python
        removal_ids = {c.word_id for c in _applied}
        retained_real = [
            w for w in analysis.words
            if not w.text.startswith("[") and w.id not in removal_ids
        ]
        html_cleaned = _merge_duration(retained_real)

        assert abs(pipeline_cleaned - html_cleaned) < 0.01, (
            f"Pipeline cleaned={pipeline_cleaned:.3f}s vs "
            f"HTML logic cleaned={html_cleaned:.3f}s"
        )

    def test_applied_count_matches_pipeline(self) -> None:
        analysis = _make_rich_analysis_result()
        _retained, applied = apply_removals(
            list(analysis.words),
            list(analysis.candidates),
            threshold=analysis.applied_params.threshold,
            categories=analysis.applied_params.categories,
        )
        # All 3 candidates should be applied (filler@1.0, repeat@0.85, pause@1.0)
        assert len(applied) == 3


def _merge_duration(words: list[WordToken]) -> float:
    """Replicate JS merge-adjacent-retained logic: gap <= 0.1s merges."""
    if not words:
        return 0.0
    total = 0.0
    seg_start = words[0].start_sec
    seg_end = words[0].end_sec
    for w in words[1:]:
        if w.start_sec - seg_end <= 0.1:
            seg_end = w.end_sec
        else:
            total += seg_end - seg_start
            seg_start = w.start_sec
            seg_end = w.end_sec
    total += seg_end - seg_start
    return total


class TestHtmlPauseRendering:
    """Removed PAUSE tokens must be rendered with the correct classes."""

    def test_removed_pause_has_removed_class(self) -> None:
        analysis = _make_rich_analysis_result()
        html = generate_report_html(analysis, "test.mp4")
        # JS template builds: 'pause-pill removed' for removed pauses
        assert "pause-pill removed" in html

    def test_removed_filler_has_badge(self) -> None:
        analysis = _make_rich_analysis_result()
        html = generate_report_html(analysis, "test.mp4")
        assert "badge-filler" in html

    def test_llm_candidate_has_badge(self) -> None:
        analysis = _make_rich_analysis_result()
        html = generate_report_html(analysis, "test.mp4")
        assert "badge-llm_suggested" in html

    def test_pause_pill_style_defined(self) -> None:
        """CSS must define both .pause-pill and .pause-pill.removed styles."""
        analysis = _make_rich_analysis_result()
        html = generate_report_html(analysis, "test.mp4")
        assert ".pause-pill.removed" in html
        assert ".pause-pill {" in html or ".pause-pill{" in html
