"""Tests for FFmpeg export logic."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from autoclip.media.ffmpeg import (
    _build_concat_demuxer,
    _build_concat_filter,
    export_clean_video,
)
from autoclip.models import Segment


class TestConcatFilter:
    def test_single_segment_no_fade(self) -> None:
        """Single segment should have no fades (no boundaries)."""
        segments = [Segment(start_sec=1.0, end_sec=5.0)]
        cmd = _build_concat_filter(segments, "in.mp4")
        filter_str = " ".join(cmd)
        assert "concat=n=1" in filter_str
        # Single segment: no fade in (first) and no fade out (last)
        assert "afade=t=in" not in filter_str
        assert "afade=t=out" not in filter_str

    def test_two_segments_have_fades(self) -> None:
        segments = [
            Segment(start_sec=0.0, end_sec=2.0),
            Segment(start_sec=3.0, end_sec=5.0),
        ]
        cmd = _build_concat_filter(segments, "in.mp4")
        filter_str = " ".join(cmd)
        assert "concat=n=2" in filter_str
        # First segment: no fade in, but fade out
        # Second segment: fade in, no fade out
        assert "afade=t=out" in filter_str
        assert "afade=t=in" in filter_str

    def test_three_segments(self) -> None:
        segments = [
            Segment(start_sec=0.0, end_sec=1.0),
            Segment(start_sec=2.0, end_sec=3.0),
            Segment(start_sec=4.0, end_sec=5.0),
        ]
        cmd = _build_concat_filter(segments, "in.mp4")
        filter_str = " ".join(cmd)
        assert "concat=n=3" in filter_str


class TestConcatDemuxer:
    @patch("autoclip.media.ffmpeg.subprocess.run")
    def test_creates_segments_and_list(self, mock_run: MagicMock, tmp_path: object) -> None:
        mock_run.return_value = MagicMock()
        segments = [
            Segment(start_sec=0.0, end_sec=1.0),
            Segment(start_sec=2.0, end_sec=3.0),
        ]
        tmp_dir = str(tmp_path)
        cmd, list_path = _build_concat_demuxer(segments, "in.mp4", tmp_dir)

        # Should have created per-segment trim commands
        assert mock_run.call_count == 2
        # Should have created a concat list file
        assert os.path.exists(list_path)
        with open(list_path) as f:
            content = f.read()
        assert "seg_0000" in content
        assert "seg_0001" in content
        # Final command should be concat demuxer
        assert "-f" in cmd
        assert "concat" in cmd

    @patch("autoclip.media.ffmpeg.subprocess.run")
    def test_segment_trim_failure(self, mock_run: MagicMock, tmp_path: object) -> None:
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr="error")
        segments = [Segment(start_sec=0.0, end_sec=1.0)]
        with pytest.raises(RuntimeError, match="segment trim failed"):
            _build_concat_demuxer(segments, "in.mp4", str(tmp_path))


class TestExportCleanVideo:
    def test_empty_segments_raises(self) -> None:
        with pytest.raises(ValueError, match="No segments"):
            export_clean_video("in.mp4", [], "out.mp4")

    @patch("autoclip.media.ffmpeg.subprocess.run")
    def test_uses_concat_filter_for_small_count(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock()
        segments = [Segment(start_sec=i, end_sec=i + 0.5) for i in range(50)]
        export_clean_video("in.mp4", segments, "/tmp/out.mp4")
        # Single FFmpeg call for concat filter
        assert mock_run.call_count == 1

    @patch("autoclip.media.ffmpeg.subprocess.run")
    @patch("autoclip.media.ffmpeg.shutil.rmtree")
    def test_uses_demuxer_for_large_count(
        self, mock_rmtree: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = MagicMock()
        segments = [Segment(start_sec=i, end_sec=i + 0.5) for i in range(51)]
        export_clean_video("in.mp4", segments, "/tmp/out.mp4")
        # 51 trim calls + 1 concat call = 52
        assert mock_run.call_count == 52
        # Temp dir should be cleaned up
        mock_rmtree.assert_called_once()

    @patch("autoclip.media.ffmpeg.subprocess.run")
    def test_creates_output_directory(self, mock_run: MagicMock, tmp_path: object) -> None:
        mock_run.return_value = MagicMock()
        out_path = os.path.join(str(tmp_path), "subdir", "out.mp4")
        segments = [Segment(start_sec=0.0, end_sec=1.0)]
        export_clean_video("in.mp4", segments, out_path)
        assert os.path.isdir(os.path.join(str(tmp_path), "subdir"))
