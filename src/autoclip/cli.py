"""Click CLI entry point for AutoClip."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from autoclip.config import AppConfig, load_config
from autoclip.media.download import download_video, is_url
from autoclip.media.ffmpeg import check_ffmpeg, export_clean_video, extract_audio
from autoclip.media.probe import probe_video
from autoclip.models import (
    ALL_CLI_CATEGORIES,
    CLI_CATEGORY_MAP,
    AutoRemovalCandidate,
    CleanResult,
    RemovalEntry,
    Segment,
)
from autoclip.processing.finecut import (
    apply_removals,
    detect_fillers,
    detect_pauses,
    merge_retained_segments,
    normalize_whisper_words,
    parse_cleanup_response,
)
from autoclip.processing.prompts import build_cleanup_prompt
from autoclip.providers.registry import create_asr_provider, create_llm_provider
from autoclip.utils import format_duration, format_timestamp, setup_logging

console = Console()
logger = logging.getLogger(__name__)


def _check_dependencies(config: AppConfig) -> None:
    """Verify external dependencies are available."""
    check_ffmpeg()

    if config.llm.provider == "ollama":
        _check_ollama(config)


def _check_ollama(config: AppConfig) -> None:
    """Check Ollama service and model availability."""
    import httpx

    base_url = config.llm.base_url.replace("/v1", "")
    try:
        with httpx.Client(trust_env=False) as client:
            resp = client.get(f"{base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        console.print(
            "[red]Error:[/red] Cannot connect to Ollama. "
            "Start it with: [bold]ollama serve[/bold]"
        )
        sys.exit(1)

    models = [m.get("name", "") for m in data.get("models", [])]
    model_name = config.llm.model
    # Check both exact match and without tag
    if not any(model_name in m for m in models):
        console.print(
            f"[red]Error:[/red] Ollama model '{model_name}' not found.\n"
            f"Pull it with: [bold]ollama pull {model_name}[/bold]"
        )
        sys.exit(1)


def _build_clean_result(
    source: str,
    original_duration: float,
    segments: list[Segment],
    applied_removals: list[AutoRemovalCandidate],
    detected_language: str,
) -> CleanResult:
    """Build CleanResult from pipeline outputs."""
    cleaned_duration = sum(s.end_sec - s.start_sec for s in segments)
    reduction = (
        ((original_duration - cleaned_duration) / original_duration * 100)
        if original_duration > 0
        else 0.0
    )

    # Count removals by CLI category
    counts: dict[str, int] = {}
    for r in applied_removals:
        cli_cat = CLI_CATEGORY_MAP[r.reason]
        counts[cli_cat] = counts.get(cli_cat, 0) + 1

    removal_entries = tuple(
        RemovalEntry(
            word_id=r.word_id,
            text=r.text,
            reason=CLI_CATEGORY_MAP[r.reason],
            confidence=r.confidence,
            start_sec=r.start_sec,
            end_sec=r.end_sec,
        )
        for r in applied_removals
    )

    return CleanResult(
        source=source,
        original_duration_sec=round(original_duration, 3),
        cleaned_duration_sec=round(cleaned_duration, 3),
        reduction_percent=round(reduction, 2),
        removals=removal_entries,
        removal_counts=counts,
        retained_segments=tuple(segments),
        detected_language=detected_language,
    )


def _print_preview(
    all_candidates: list[AutoRemovalCandidate],
    applied_removals: list[AutoRemovalCandidate],
    original_duration: float,
    cleaned_duration: float,
) -> None:
    """Print preview analysis to console."""
    applied_ids = {r.word_id for r in applied_removals}

    table = Table(title="Disfluency Analysis")
    table.add_column("#", style="dim")
    table.add_column("Timestamp")
    table.add_column("Text")
    table.add_column("Category")
    table.add_column("Confidence")
    table.add_column("Applied", justify="center")

    for i, c in enumerate(all_candidates, 1):
        is_applied = c.word_id in applied_ids
        table.add_row(
            str(i),
            f"{format_timestamp(c.start_sec)}-{format_timestamp(c.end_sec)}",
            c.text,
            CLI_CATEGORY_MAP[c.reason],
            f"{c.confidence:.2f}",
            "[green]Yes[/green]" if is_applied else "[dim]No[/dim]",
        )

    console.print(table)

    reduction = (
        (original_duration - cleaned_duration) / original_duration * 100
        if original_duration > 0
        else 0
    )
    console.print(
        f"\n[bold]Summary:[/bold] {len(applied_removals)} removals applied "
        f"| Duration: {format_duration(original_duration)} -> {format_duration(cleaned_duration)} "
        f"({reduction:.1f}% reduction)"
    )


def _print_export_summary(result: CleanResult, output_path: str, json_path: str) -> None:
    """Print post-export summary."""
    console.print("\n[bold green]Export complete![/bold green]\n")
    console.print(f"  Output:    {output_path}")
    console.print(f"  Report:    {json_path}")
    console.print(
        f"  Duration:  {format_duration(result.original_duration_sec)} -> "
        f"{format_duration(result.cleaned_duration_sec)} "
        f"({result.reduction_percent:.1f}% reduction)"
    )
    if result.removal_counts:
        parts = [f"{cat}: {n}" for cat, n in sorted(result.removal_counts.items())]
        console.print(f"  Removals:  {', '.join(parts)}")


@click.group()
@click.version_option()
def main() -> None:
    """AutoClip — automatically clean spoken-word videos."""


@main.command()
@click.argument("input_path")
@click.option("-o", "--output", "output_dir", default=None, help="Output directory")
@click.option("--threshold", type=float, default=None, help="Confidence threshold (0.0-1.0)")
@click.option("--categories", default=None, help="Comma-separated categories: filler,repeat,false-start,pause")
@click.option("--llm", "llm_provider", default=None, help="LLM provider: ollama or openai")
@click.option("--preview", is_flag=True, help="Analyze only, no export")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def clean(
    input_path: str,
    output_dir: str | None,
    threshold: float | None,
    categories: str | None,
    llm_provider: str | None,
    preview: bool,
    verbose: bool,
) -> None:
    """Clean a video by removing disfluencies.

    INPUT_PATH can be a local video file or YouTube URL.
    """
    setup_logging(verbose)

    # Build config with CLI overrides
    cli_overrides: dict[str, Any] = {}
    if llm_provider:
        cli_overrides["llm"] = {"provider": llm_provider}
    if output_dir:
        cli_overrides["output"] = {"dir": output_dir}
    if threshold is not None:
        cli_overrides["clean"] = {"auto_apply_threshold": threshold}

    config = load_config(cli_overrides=cli_overrides)

    # Parse categories
    cat_list: list[str] | None = None
    if categories:
        cat_list = [c.strip() for c in categories.split(",")]
        for c in cat_list:
            if c not in ALL_CLI_CATEGORIES:
                console.print(f"[red]Error:[/red] Unknown category '{c}'. Valid: {ALL_CLI_CATEGORIES}")
                sys.exit(1)

    effective_threshold = config.clean.auto_apply_threshold

    # Check dependencies
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
        progress.add_task("Checking dependencies...", total=None)
        try:
            _check_dependencies(config)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    # Download if URL
    video_path = input_path
    downloaded = False
    if is_url(input_path):
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
            progress.add_task("Downloading video...", total=None)
            try:
                video_path = download_video(input_path)
                downloaded = True
            except RuntimeError as e:
                console.print(f"[red]Error:[/red] {e}")
                sys.exit(1)

    # Verify file exists
    if not os.path.isfile(video_path):
        console.print(f"[red]Error:[/red] File not found: {video_path}")
        sys.exit(1)

    # Probe video
    try:
        meta = probe_video(video_path)
    except (FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    console.print(f"Input: {input_path} ({format_duration(meta.duration_sec)}, {meta.width}x{meta.height})")

    # Extract audio
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
        progress.add_task("Extracting audio...", total=None)
        try:
            audio_path = extract_audio(video_path)
        except RuntimeError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    try:
        # ASR
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
            progress.add_task("Transcribing (this may take a while)...", total=None)
            asr = create_asr_provider(config)
            caption_segments, detected_language = asr.transcribe(
                audio_path,
                language=config.asr.language if config.asr.language != "auto" else None,
            )

        # Edge case: no speech
        if not caption_segments:
            console.print("[yellow]No speech detected in this video. Nothing to clean.[/yellow]")
            sys.exit(0)

        # Normalize
        words = normalize_whisper_words(caption_segments)
        console.print(f"Detected language: {detected_language} | Words: {len(words)}")

        # Detect fillers
        filler_candidates = detect_fillers(words, detected_language)

        # Detect pauses
        words_with_pauses, pause_candidates = detect_pauses(
            words, long_pause_ms=config.clean.long_pause_ms
        )

        # LLM classification
        llm_candidates: list[AutoRemovalCandidate] = []
        if words:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
                progress.add_task("Classifying disfluencies...", total=None)
                prompt = build_cleanup_prompt(words_with_pauses)
                llm = create_llm_provider(config)
                response = llm.complete(prompt, temperature=config.llm.temperature)
                llm_candidates = parse_cleanup_response(response)

        # Merge all candidates
        all_candidates = filler_candidates + pause_candidates + llm_candidates

        # Apply removals
        retained_words, applied_removals = apply_removals(
            words_with_pauses,
            all_candidates,
            threshold=effective_threshold,
            categories=cat_list,
        )

        # Edge case: no disfluencies
        if not applied_removals:
            console.print("[green]No disfluencies detected. Your video is already clean![/green]")
            sys.exit(0)

        # Merge segments
        segments = merge_retained_segments(retained_words)

        # Build result
        result = _build_clean_result(
            source=input_path,
            original_duration=meta.duration_sec,
            segments=segments,
            applied_removals=applied_removals,
            detected_language=detected_language,
        )

        if preview:
            cleaned_duration = sum(s.end_sec - s.start_sec for s in segments)
            _print_preview(all_candidates, applied_removals, meta.duration_sec, cleaned_duration)
            return

        # Export
        out_dir = config.output.dir
        stem = Path(video_path).stem
        ext = Path(video_path).suffix
        output_path = os.path.join(out_dir, f"{stem}_clean{ext}")
        json_path = os.path.join(out_dir, f"{stem}_clean.json")

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
            progress.add_task("Exporting clean video...", total=None)
            export_clean_video(video_path, segments, output_path)

        # Write JSON
        os.makedirs(out_dir, exist_ok=True)
        with open(json_path, "w") as f:
            f.write(result.model_dump_json(indent=2))

        _print_export_summary(result, output_path, json_path)

    finally:
        # Cleanup temp audio
        if os.path.isfile(audio_path):
            os.unlink(audio_path)
        # Cleanup downloaded video
        if downloaded and os.path.isfile(video_path):
            os.unlink(video_path)
