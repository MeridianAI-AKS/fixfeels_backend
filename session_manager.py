"""In-memory session state for script-driven interviews."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from script_loader import ScriptStep


@dataclass
class SessionState:
    session_id: str
    steps: list[ScriptStep]
    step_index: int = 0
    pending_question: str | None = None
    started: bool = False
    awaiting_customer_question: bool = False

    def clear_pending(self) -> None:
        self.pending_question = None

    def is_complete(self) -> bool:
        return self.step_index >= len(self.steps) and self.pending_question is None


_sessions: dict[str, SessionState] = {}


def create_session(steps: list[ScriptStep]) -> SessionState:
    session_id = str(uuid.uuid4())
    state = SessionState(session_id=session_id, steps=steps)
    _sessions[session_id] = state
    return state


def get_session(session_id: str) -> SessionState | None:
    return _sessions.get(session_id)


def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
