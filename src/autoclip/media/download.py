"""yt-dlp wrapper for downloading videos from YouTube URLs."""

from __future__ import annotations

import logging
import re
import tempfile

logger = logging.getLogger(__name__)

# Patterns that indicate a URL (not a local file)
URL_PATTERNS = (
    re.compile(r"^https?://"),
    re.compile(r"^(www\.)?youtube\.com/"),
    re.compile(r"^youtu\.be/"),
)


def is_url(input_path: str) -> bool:
    """Check if input looks like a URL rather than a local file path."""
    return any(p.match(input_path) for p in URL_PATTERNS)


def download_video(url: str) -> str:
    """Download video from URL using yt-dlp.

    Args:
        url: YouTube or other supported URL.

    Returns:
        Path to the downloaded video file.

    Raises:
        RuntimeError: If download fails.
    """
    # Import here to avoid startup cost if not needed
    import yt_dlp

    output_dir = tempfile.mkdtemp(prefix="autoclip_dl_")
    output_template = f"{output_dir}/%(title)s.%(ext)s"

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }

    logger.info("Downloading video from %s", url)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise RuntimeError("yt-dlp returned no info for URL")
            filename = ydl.prepare_filename(info)
            logger.info("Downloaded: %s", filename)
            return str(filename)
    except Exception as e:
        raise RuntimeError(f"Video download failed: {e}") from e
