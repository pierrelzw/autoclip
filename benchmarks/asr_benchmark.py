#!/usr/bin/env python3
"""Benchmark script comparing faster-whisper and mlx-whisper ASR engines.

Usage:
    python benchmarks/asr_benchmark.py <audio_or_video_file> [--model large-v3]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time


def _is_video(path: str) -> bool:
    """Check if a file is a video (not audio-only) by extension."""
    video_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts"}
    return os.path.splitext(path)[1].lower() in video_exts


def _extract_audio(video_path: str) -> str:
    """Extract 16kHz mono WAV from a video file."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        wav_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return wav_path


def _get_audio_duration(path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def _try_import(module: str) -> bool:
    """Check if a module is importable."""
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _run_faster_whisper(
    audio_path: str,
    model_name: str,
) -> dict[str, object]:
    """Run faster-whisper and return metrics."""
    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel(model_name, device="auto", compute_type="auto")
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    segments_iter, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    segments = list(segments_iter)
    t_transcribe = time.perf_counter() - t1

    word_count = sum(len(getattr(s, "words", None) or []) for s in segments)
    words = []
    for s in segments:
        for w in getattr(s, "words", None) or []:
            words.append({"text": w.word.strip(), "start": w.start, "end": w.end})

    return {
        "engine": "faster-whisper",
        "model_load_sec": t_load,
        "transcribe_sec": t_transcribe,
        "total_sec": t_load + t_transcribe,
        "segment_count": len(segments),
        "word_count": word_count,
        "language": str(info.language),
        "words": words,
    }


def _run_mlx_whisper(
    audio_path: str,
    model_name: str,
) -> dict[str, object]:
    """Run mlx-whisper and return metrics."""
    import mlx_whisper

    # Map short model names to mlx-community repo paths
    model_repo = _map_mlx_model_name(model_name)

    t0 = time.perf_counter()
    # mlx-whisper loads model on first transcribe, so we do a dummy to measure load
    # Actually, mlx_whisper.transcribe handles everything in one call
    # We measure total and estimate load separately is not feasible,
    # so we just measure total transcribe time
    t_load = 0.0  # mlx-whisper doesn't separate load from transcribe
    t0 = time.perf_counter()

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_repo,
        word_timestamps=True,
    )
    t_transcribe = time.perf_counter() - t0

    segments = result.get("segments", [])
    language = result.get("language", "unknown")

    word_count = 0
    words: list[dict[str, object]] = []
    for seg in segments:
        seg_words = seg.get("words", [])
        word_count += len(seg_words)
        for w in seg_words:
            words.append({
                "text": str(w.get("word", "")).strip(),
                "start": w.get("start", 0.0),
                "end": w.get("end", 0.0),
            })

    return {
        "engine": "mlx-whisper",
        "model_load_sec": t_load,
        "transcribe_sec": t_transcribe,
        "total_sec": t_transcribe,
        "segment_count": len(segments),
        "word_count": word_count,
        "language": str(language),
        "words": words,
    }


def _map_mlx_model_name(model_name: str) -> str:
    """Map short model names to mlx-community HuggingFace repos."""
    mapping = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large": "mlx-community/whisper-large-mlx",
        "large-v2": "mlx-community/whisper-large-v2-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    }
    if "/" in model_name:
        return model_name
    return mapping.get(model_name, f"mlx-community/whisper-{model_name}-mlx")


def _print_comparison(
    results: list[dict[str, object]],
    audio_duration: float,
) -> None:
    """Print comparison table of benchmark results."""
    print("\n" + "=" * 70)
    print("ASR Benchmark Results")
    print("=" * 70)
    print(f"Audio duration: {audio_duration:.1f}s")
    print()

    # Metrics table
    headers = ["Metric", *[str(r["engine"]) for r in results]]
    rows = [
        ("Model load (s)", *[f"{r['model_load_sec']:.2f}" for r in results]),
        ("Transcribe (s)", *[f"{r['transcribe_sec']:.2f}" for r in results]),
        ("Total (s)", *[f"{r['total_sec']:.2f}" for r in results]),
        (
            "RTF",
            *[
                f"{float(str(r['total_sec'])) / audio_duration:.3f}"
                for r in results
            ],
        ),
        ("Segments", *[str(r["segment_count"]) for r in results]),
        ("Words", *[str(r["word_count"]) for r in results]),
        ("Language", *[str(r["language"]) for r in results]),
    ]

    col_widths = [max(len(str(h)), *(len(str(row[i])) for row in rows)) + 2 for i, h in enumerate(headers)]

    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=False))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("".join(str(v).ljust(w) for v, w in zip(row, col_widths, strict=False)))

    # Word timestamp comparison (first 20 words)
    print("\n" + "-" * 70)
    print("First 20 words comparison:")
    print("-" * 70)
    print(f"{'#':<4} {'Engine 1':<30} {'Engine 2':<30}")
    print(f"{'':4} {'Text':<15} {'Start':>6} {'End':>6}   {'Text':<15} {'Start':>6} {'End':>6}")
    print("-" * 70)

    words_lists = [list(r.get("words", [])) for r in results]  # type: ignore[union-attr]
    max_words = min(20, max((len(wl) for wl in words_lists), default=0))

    for i in range(max_words):
        parts: list[str] = [f"{i + 1:<4}"]
        for wl in words_lists:
            if i < len(wl):
                w = wl[i]
                text = str(w.get("text", ""))[:14]  # type: ignore[union-attr]
                start = float(str(w.get("start", 0)))  # type: ignore[union-attr]
                end = float(str(w.get("end", 0)))  # type: ignore[union-attr]
                parts.append(f"{text:<15} {start:>6.2f} {end:>6.2f}  ")
            else:
                parts.append(f"{'—':<15} {'—':>6} {'—':>6}  ")
        print("".join(parts))

    print("=" * 70)


def main() -> None:
    """Run ASR benchmark."""
    parser = argparse.ArgumentParser(description="ASR engine benchmark")
    parser.add_argument("input", help="Audio or video file path")
    parser.add_argument("--model", default="large-v3", help="Model name (default: large-v3)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Prepare audio
    audio_path = args.input
    tmp_wav: str | None = None
    if _is_video(args.input):
        print(f"Extracting audio from {args.input}...")
        tmp_wav = _extract_audio(args.input)
        audio_path = tmp_wav

    try:
        audio_duration = _get_audio_duration(audio_path)
        print(f"Audio duration: {audio_duration:.1f}s")

        results: list[dict[str, object]] = []

        # Run faster-whisper
        if _try_import("faster_whisper"):
            print("\nRunning faster-whisper...")
            try:
                result = _run_faster_whisper(audio_path, args.model)
                results.append(result)
                print(f"  Done in {result['total_sec']:.2f}s")
            except Exception as e:
                print(f"  faster-whisper failed: {e}", file=sys.stderr)
        else:
            print("\nSkipping faster-whisper (not installed)")

        # Run mlx-whisper
        if _try_import("mlx_whisper"):
            print("\nRunning mlx-whisper...")
            try:
                result = _run_mlx_whisper(audio_path, args.model)
                results.append(result)
                print(f"  Done in {result['total_sec']:.2f}s")
            except Exception as e:
                print(f"  mlx-whisper failed: {e}", file=sys.stderr)
        else:
            print("\nSkipping mlx-whisper (not installed)")

        if not results:
            print("\nError: no ASR engine available. Install faster-whisper or mlx-whisper.", file=sys.stderr)
            sys.exit(1)

        _print_comparison(results, audio_duration)

    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


if __name__ == "__main__":
    main()
