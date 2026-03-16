"""FFmpeg utilities for audio extraction and video export."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

from autoclip.models import Segment

logger = logging.getLogger(__name__)

FADE_DURATION_SEC = 0.018  # ~18ms afade to mask timestamp imprecision


def check_ffmpeg() -> None:
    """Verify ffmpeg and ffprobe are available in PATH.

    Raises:
        FileNotFoundError: If ffmpeg or ffprobe is missing.
    """
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise FileNotFoundError(
                f"{tool} not found in PATH. Install FFmpeg: https://ffmpeg.org/download.html"
            )


def extract_audio(video_path: str, output_path: str | None = None) -> str:
    """Extract audio from video as WAV 16kHz mono.

    Args:
        video_path: Path to input video.
        output_path: Optional output path. If None, creates a temp file.

    Returns:
        Path to the extracted WAV file.

    Raises:
        RuntimeError: If FFmpeg fails.
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg audio extraction failed: {e.stderr}") from e

    return output_path


def _build_concat_filter(
    segments: list[Segment],
    input_path: str,
) -> list[str]:
    """Build FFmpeg command using trim+concat complex filter (for <= 50 segments)."""
    filter_parts: list[str] = []
    concat_inputs: list[str] = []

    for i, seg in enumerate(segments):
        duration = seg.end_sec - seg.start_sec
        # Video trim
        vf = (
            f"[0:v]trim=start={seg.start_sec}:duration={duration},"
            f"setpts=PTS-STARTPTS[v{i}]"
        )
        # Audio trim with afade
        af = (
            f"[0:a]atrim=start={seg.start_sec}:duration={duration},"
            f"asetpts=PTS-STARTPTS"
        )
        # Add fade in at start of each segment (except first)
        if i > 0:
            af += f",afade=t=in:st=0:d={FADE_DURATION_SEC}"
        # Add fade out at end of each segment (except last)
        if i < len(segments) - 1:
            af += f",afade=t=out:st={duration - FADE_DURATION_SEC}:d={FADE_DURATION_SEC}"
        af += f"[a{i}]"

        filter_parts.append(vf)
        filter_parts.append(af)
        concat_inputs.append(f"[v{i}][a{i}]")

    n = len(segments)
    concat = "".join(concat_inputs) + f"concat=n={n}:v=1:a=1[outv][outa]"
    filter_parts.append(concat)

    filter_complex = ";".join(filter_parts)

    return [
        "ffmpeg",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
    ]


def _build_concat_demuxer(
    segments: list[Segment],
    input_path: str,
    tmp_dir: str,
) -> tuple[list[str], str]:
    """Build FFmpeg commands using concat demuxer (for > 50 segments).

    Returns the final concat command and the list file path.
    """
    list_path = os.path.join(tmp_dir, "concat_list.txt")
    entries: list[str] = []

    for i, seg in enumerate(segments):
        duration = seg.end_sec - seg.start_sec
        seg_path = os.path.join(tmp_dir, f"seg_{i:04d}.mp4")

        # Build per-segment trim command
        afade_filter = ""
        if i > 0:
            afade_filter += f"afade=t=in:st=0:d={FADE_DURATION_SEC}"
        if i < len(segments) - 1:
            fade_out_start = max(0, duration - FADE_DURATION_SEC)
            if afade_filter:
                afade_filter += f",afade=t=out:st={fade_out_start}:d={FADE_DURATION_SEC}"
            else:
                afade_filter += f"afade=t=out:st={fade_out_start}:d={FADE_DURATION_SEC}"

        cmd = [
            "ffmpeg",
            "-ss", str(seg.start_sec),
            "-i", input_path,
            "-t", str(duration),
        ]
        if afade_filter:
            cmd += ["-af", afade_filter]
        cmd += ["-y", seg_path]

        logger.debug("Trimming segment %d/%d", i + 1, len(segments))
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg segment trim failed (segment {i}): {e.stderr}"
            ) from e

        entries.append(f"file '{seg_path}'")

    with open(list_path, "w") as f:
        f.write("\n".join(entries))

    concat_cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
    ]

    return concat_cmd, list_path


def export_clean_video(
    input_path: str,
    segments: list[Segment],
    output_path: str,
) -> str:
    """Export cleaned video by concatenating retained segments.

    Uses trim+concat filter for <= 50 segments, concat demuxer for > 50.

    Args:
        input_path: Path to original video.
        segments: List of retained Segment objects.
        output_path: Path for output video.

    Returns:
        Path to the exported video.

    Raises:
        RuntimeError: If FFmpeg fails.
        ValueError: If no segments provided.
    """
    if not segments:
        raise ValueError("No segments to export")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_path):
        logger.warning("Output file already exists, overwriting: %s", output_path)

    if len(segments) <= 50:
        cmd = _build_concat_filter(segments, input_path)
        cmd += ["-y", output_path]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg export failed: {e.stderr}") from e
    else:
        tmp_dir = tempfile.mkdtemp(prefix="autoclip_")
        try:
            concat_cmd, _ = _build_concat_demuxer(segments, input_path, tmp_dir)
            concat_cmd += ["-y", output_path]
            try:
                subprocess.run(concat_cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"FFmpeg concat failed: {e.stderr}") from e
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_path
