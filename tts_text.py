"""Clean text for natural, continuous Azure TTS (no awkward SSML pauses)."""

from __future__ import annotations

import re

_STRIP_PATTERNS = [
    (re.compile(r"\\+"), " "),
    (re.compile(r"\*+|_+|`+"), ""),
    (re.compile(r"#{1,6}\s*"), ""),
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),
    (re.compile(r"[:;](?:smile|laugh|pause|wink|grin|sad)[a-z]*[:;]?", re.I), ""),
    (re.compile(r"\(\s*(?:smile|laugh|pause|wink|grin|sad)\s*\)", re.I), ""),
    (re.compile(r"\b(?:smile|laugh|pause|wink|grin|sad)\b", re.I), ""),
    (re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\u2600-\u26FF]"), ""),
    (re.compile(r"[{}[\]|<>]"), " "),
]

# Symbols that make the neural voice insert long unnatural gaps.
_PAUSE_CHARS = re.compile(r"[.!?।,;:'\"()\[\]{}\-–—/\\@#]+")


def sanitize_for_tts(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    for pattern, repl in _STRIP_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)

    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_for_natural_speech(text: str) -> str:
    """One flowing spoken line — like a live call, not sentence-by-sentence."""
    text = re.sub(r"https?://fship\.in/?", "f ship dot in", text, flags=re.I)
    text = re.sub(r"\bfship\.in\b", "f ship dot in", text, flags=re.I)
    text = re.sub(r"support@fship\.in", "support at fship dot in", text, flags=re.I)
    text = _PAUSE_CHARS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_text_for_tts(text: str) -> str:
    cleaned = sanitize_for_tts(text)
    if not cleaned:
        return ""
    return normalize_for_natural_speech(cleaned)


def escape_ssml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def build_ssml(text: str, voice: str) -> str:
    """Minimal SSML — default voice pacing (no prosody rate that slows speech)."""
    body = escape_ssml(prepare_text_for_tts(text))
    if not body:
        return ""
    return (
        f'<speak version="1.0" xml:lang="en-IN">'
        f'<voice name="{voice}">{body}</voice>'
        f"</speak>"
    )
