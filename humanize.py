"""Script transition phrases — varied acks, no repeated 'samajh gayi'."""

from __future__ import annotations

_ACKS = (
    "Theek hai",
    "Achha",
    "Okay",
    "Bilkul",
    "Shukriya",
    "Right",
)


def fallback_transition(next_line: str, *, ack_index: int = 0) -> str:
    """Brief varied acknowledgment + next script line — no fixed 'samajh gayi'."""
    ack = _ACKS[ack_index % len(_ACKS)]
    line = next_line.strip()
    if not line:
        return ack
    return f"{ack} {line}"
