"""LLM prompt templates for disfluency classification."""

from __future__ import annotations

from autoclip.models import WordToken

CLEANUP_PROMPT = """You are analyzing a speech transcript for oral disfluencies.

Given the following word list with timestamps, identify words that are:
- **stutter**: repeated partial words or sounds (e.g., "I I I went", "th-the")
- **repeat**: repeated full words or short phrases (e.g., "I want I want to")
- **false_start**: abandoned sentence starts where the speaker restarts (e.g., "I was going to— Actually, let me")

Do NOT identify filler words (um, uh, etc.) — those are handled separately.

Return a JSON array. Each entry must have:
- "word_id": the word ID (e.g., "w_5")
- "text": the word text
- "reason": one of "stutter", "repeat", "false_start"
- "confidence": float 0.0-1.0 (how confident you are this is a disfluency)
- "start_sec": word start time
- "end_sec": word end time

If no disfluencies are found, return an empty array: []

Return ONLY the JSON array, no other text.

## Word List

{word_list}
"""


def build_cleanup_prompt(words: list[WordToken]) -> str:
    """Build a cleanup prompt for LLM disfluency classification.

    Only includes real words (not synthetic pause tokens or fillers already detected).

    Args:
        words: Word tokens to classify.

    Returns:
        Formatted prompt string.
    """
    lines: list[str] = []
    for w in words:
        # Skip synthetic tokens
        if w.text.startswith("["):
            continue
        lines.append(f"{w.id} | {w.start_sec:.3f}-{w.end_sec:.3f} | {w.text}")

    word_list = "\n".join(lines)
    return CLEANUP_PROMPT.format(word_list=word_list)
