"""FFprobe wrapper for extracting video metadata."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VideoMeta:
    """Video metadata extracted by ffprobe."""

    duration_sec: float
    video_codec: str
    audio_codec: str
    width: int
    height: int
    fps: float


def probe_video(path: str) -> VideoMeta:
    """Extract video metadata using ffprobe.

    Args:
        path: Path to the video file.

    Returns:
        VideoMeta with duration, codecs, resolution, and fps.

    Raises:
        FileNotFoundError: If ffprobe is not in PATH.
        RuntimeError: If ffprobe fails or output is unparseable.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "ffprobe not found in PATH. Install FFmpeg: https://ffmpeg.org/download.html"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}") from e

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream: dict[str, Any] = {}
    audio_stream: dict[str, Any] = {}
    for s in streams:
        if s.get("codec_type") == "video" and not video_stream:
            video_stream = s
        elif s.get("codec_type") == "audio" and not audio_stream:
            audio_stream = s

    duration = float(fmt.get("duration", 0))
    video_codec = str(video_stream.get("codec_name", "unknown"))
    audio_codec = str(audio_stream.get("codec_name", "unknown"))
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    # Parse fps from r_frame_rate (e.g., "30000/1001")
    fps_str = str(video_stream.get("r_frame_rate", "0/1"))
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    else:
        fps = float(fps_str)

    return VideoMeta(
        duration_sec=duration,
        video_codec=video_codec,
        audio_codec=audio_codec,
        width=width,
        height=height,
        fps=fps,
    )
