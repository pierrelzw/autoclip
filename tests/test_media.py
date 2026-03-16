"""Tests for media utilities."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from autoclip.media.download import download_video, is_url
from autoclip.media.ffmpeg import (
    _build_concat_filter,
    check_ffmpeg,
    export_clean_video,
    extract_audio,
)
from autoclip.media.probe import probe_video
from autoclip.models import Segment


class TestProbeVideo:
    MOCK_FFPROBE_OUTPUT = json.dumps({
        "format": {"duration": "120.5"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30000/1001",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
            },
        ],
    })

    @patch("autoclip.media.probe.subprocess.run")
    def test_successful_probe(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout=self.MOCK_FFPROBE_OUTPUT)
        meta = probe_video("test.mp4")
        assert meta.duration_sec == 120.5
        assert meta.video_codec == "h264"
        assert meta.audio_codec == "aac"
        assert meta.width == 1920
        assert meta.height == 1080
        assert abs(meta.fps - 29.97) < 0.01

    @patch("autoclip.media.probe.subprocess.run")
    def test_ffprobe_not_found(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(FileNotFoundError, match="ffprobe not found"):
            probe_video("test.mp4")

    @patch("autoclip.media.probe.subprocess.run")
    def test_ffprobe_failure(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe", stderr="error")
        with pytest.raises(RuntimeError, match="ffprobe failed"):
            probe_video("test.mp4")


class TestCheckFfmpeg:
    @patch("autoclip.media.ffmpeg.shutil.which")
    def test_both_available(self, mock_which: MagicMock) -> None:
        mock_which.return_value = "/usr/bin/ffmpeg"
        check_ffmpeg()  # Should not raise

    @patch("autoclip.media.ffmpeg.shutil.which")
    def test_ffmpeg_missing(self, mock_which: MagicMock) -> None:
        mock_which.return_value = None
        with pytest.raises(FileNotFoundError, match="ffmpeg not found"):
            check_ffmpeg()


class TestExtractAudio:
    @patch("autoclip.media.ffmpeg.subprocess.run")
    def test_extract_to_temp(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock()
        result = extract_audio("video.mp4")
        assert result.endswith(".wav")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "-ar" in cmd
        assert "16000" in cmd

    @patch("autoclip.media.ffmpeg.subprocess.run")
    def test_extract_to_path(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock()
        result = extract_audio("video.mp4", "/tmp/out.wav")
        assert result == "/tmp/out.wav"

    @patch("autoclip.media.ffmpeg.subprocess.run")
    def test_extract_failure(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr="err")
        with pytest.raises(RuntimeError, match="extraction failed"):
            extract_audio("video.mp4")


class TestBuildConcatFilter:
    def test_single_segment(self) -> None:
        segments = [Segment(start_sec=0.0, end_sec=5.0)]
        cmd = _build_concat_filter(segments, "input.mp4")
        assert "concat=n=1:v=1:a=1" in " ".join(cmd)

    def test_multiple_segments(self) -> None:
        segments = [
            Segment(start_sec=0.0, end_sec=2.0),
            Segment(start_sec=3.0, end_sec=5.0),
        ]
        cmd = _build_concat_filter(segments, "input.mp4")
        filter_str = " ".join(cmd)
        assert "concat=n=2:v=1:a=1" in filter_str
        assert "afade" in filter_str


class TestExportCleanVideo:
    def test_no_segments_raises(self) -> None:
        with pytest.raises(ValueError, match="No segments"):
            export_clean_video("input.mp4", [], "output.mp4")

    @patch("autoclip.media.ffmpeg.subprocess.run")
    def test_small_segment_count(self, mock_run: MagicMock, tmp_path: object) -> None:
        mock_run.return_value = MagicMock()
        segments = [Segment(start_sec=0.0, end_sec=5.0)]
        result = export_clean_video("input.mp4", segments, "/tmp/out.mp4")
        assert result == "/tmp/out.mp4"
        # Should use concat filter (single subprocess call)
        assert mock_run.call_count == 1


class TestIsUrl:
    def test_http_url(self) -> None:
        assert is_url("https://youtube.com/watch?v=abc") is True

    def test_youtu_be(self) -> None:
        assert is_url("youtu.be/abc123") is True

    def test_local_file(self) -> None:
        assert is_url("video.mp4") is False
        assert is_url("/path/to/video.mp4") is False

    def test_www_youtube(self) -> None:
        assert is_url("www.youtube.com/watch?v=abc") is True


class TestDownloadVideo:
    @patch.dict("sys.modules", {"yt_dlp": MagicMock()})
    def test_successful_download(self) -> None:
        import sys
        mock_ytdlp = sys.modules["yt_dlp"]
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {"title": "test", "ext": "mp4"}
        mock_ydl.prepare_filename.return_value = "/tmp/test.mp4"
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ytdlp.YoutubeDL.return_value = mock_ydl

        result = download_video("https://youtube.com/watch?v=abc")
        assert result == "/tmp/test.mp4"

    @patch.dict("sys.modules", {"yt_dlp": MagicMock()})
    def test_download_failure(self) -> None:
        import sys
        mock_ytdlp = sys.modules["yt_dlp"]
        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = Exception("network error")
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ytdlp.YoutubeDL.return_value = mock_ydl

        with pytest.raises(RuntimeError, match="download failed"):
            download_video("https://youtube.com/watch?v=abc")
