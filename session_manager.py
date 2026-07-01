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
    awaiting_customer_question: bool = False
    interview_complete: bool = False
    conversation: list[dict[str, str]] = field(default_factory=list)
    script_answers: list[dict[str, str]] = field(default_factory=list)

    def append_turn(self, role: str, content: str) -> None:
        content = content.strip()
        if content:
            self.conversation.append({"role": role, "content": content})
            if len(self.conversation) > 40:
                self.conversation = self.conversation[-40:]

    def record_script_answer(self, question: str, answer: str) -> None:
        question = question.strip()
        answer = answer.strip()
        if question and answer:
            self.script_answers.append({"question": question, "answer": answer})
            if len(self.script_answers) > 20:
                self.script_answers = self.script_answers[-20:]
        self.clear_pending()

    def clear_pending(self) -> None:
        self.pending_question = None

    def is_script_exhausted(self) -> bool:
        return self.step_index >= len(self.steps)


_sessions: dict[str, SessionState] = {}


def create_session(steps: list[ScriptStep]) -> SessionState:
    session_id = str(uuid.uuid4())
    state = SessionState(session_id=session_id, steps=steps)
    _sessions[session_id] = state
    return state


def get_session(session_id: str) -> SessionState | None:
    return _sessions.get(session_id)
