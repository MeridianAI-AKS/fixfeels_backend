"""Persist Q&A records to local JSON files."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
QA_LOG_PATH = DATA_DIR / "qa_log.json"
_write_lock = threading.Lock()


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_all() -> list[dict[str, Any]]:
    _ensure_data_dir()
    if not QA_LOG_PATH.exists():
        return []
    with QA_LOG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_all(records: list[dict[str, Any]]) -> None:
    _ensure_data_dir()
    with QA_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _append_record(record: dict[str, Any]) -> dict[str, Any]:
    with _write_lock:
        records = _load_all()
        records.append(record)
        _save_all(records)
    return record


def save_script_qa(
    *,
    session_id: str,
    script_question: str,
    user_response: str,
) -> dict[str, Any]:
    """Store a script question asked by Aarti and the user's answer."""
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "script",
        "session_id": session_id,
        "question": script_question,
        "response": user_response,
    }
    return _append_record(record)


def save_off_script_qa(
    *,
    session_id: str,
    user_question: str,
    assistant_response: str,
) -> dict[str, Any]:
    """Store an off-script user question and Aarti's general-knowledge answer."""
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "general",
        "session_id": session_id,
        "question": user_question,
        "response": assistant_response,
    }
    return _append_record(record)
