"""
Aarti Voice Assistant — FastAPI Backend
Script interview from script.docx; LLM answers off-script questions only.
"""

from __future__ import annotations

import json
import time

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from chat_handler import build_opening_reply, process_chat_stream
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    TTS_VOICE,
)
from script_loader import DEFAULT_SCRIPT_PATH, ScriptStep, load_script_steps
from session_manager import SessionState, create_session, get_session
from tts_text import build_ssml

app = FastAPI(title="Aarti Voice Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_STEPS: list[ScriptStep] = []
_speech_token: str | None = None
_speech_token_expires: float = 0.0
_http = requests.Session()
TTS_OUTPUT_FORMAT = "audio-24khz-48kbitrate-mono-mp3"


def reload_script() -> None:
    global SCRIPT_STEPS
    SCRIPT_STEPS = load_script_steps()
    print(f"Loaded {len(SCRIPT_STEPS)} script steps from {DEFAULT_SCRIPT_PATH}")


@app.on_event("startup")
def load_script_on_startup() -> None:
    try:
        reload_script()
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: {e}")
        SCRIPT_STEPS = []

    try:
        get_cached_speech_token()
        print("Speech token warmed for faster TTS.")
    except Exception as e:
        print(f"Warning: could not pre-warm speech token: {e}")


# ── MODELS ──────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    session_id: str | None = None
    interrupt: bool = False


class SessionStartRequest(BaseModel):
    pass


class TTSRequest(BaseModel):
    text: str


# ── SPEECH TOKEN ────────────────────────────────────────────────
def get_cached_speech_token() -> str:
    global _speech_token, _speech_token_expires
    now = time.time()
    if _speech_token and now < _speech_token_expires:
        return _speech_token

    token_resp = _http.post(
        f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
        headers={"Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY},
        timeout=10,
    )
    token_resp.raise_for_status()
    _speech_token = token_resp.text
    _speech_token_expires = now + 540  # 9 minutes
    return _speech_token


# ── ROUTES ──────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    questions = sum(1 for s in SCRIPT_STEPS if s.kind == "question")
    return {
        "status": "ok",
        "voice": TTS_VOICE,
        "script_loaded": len(SCRIPT_STEPS) > 0,
        "script_step_count": len(SCRIPT_STEPS),
        "script_question_count": questions,
    }


@app.post("/api/session/start")
async def start_session(_req: SessionStartRequest | None = None):
    """Start interview using only backend/data/script.docx."""
    try:
        reload_script()
    except (FileNotFoundError, ValueError):
        pass

    if not SCRIPT_STEPS:
        raise HTTPException(
            404,
            "No script found. Add your script.docx to backend/data/script.docx",
        )

    session = create_session(list(SCRIPT_STEPS))
    reply = build_opening_reply(session)
    session.append_turn("assistant", reply)

    return {
        "session_id": session.session_id,
        "reply": reply,
        "source": "script",
        "script_question": session.pending_question,
        "interview_complete": False,
    }


def _validate_chat_request(req: ChatRequest) -> tuple[str, SessionState]:
    if not req.messages:
        raise HTTPException(400, "messages are required")

    user_text = req.messages[-1].content.strip()
    if not user_text:
        raise HTTPException(400, "Empty user message")

    if not req.session_id:
        raise HTTPException(400, "Start the script interview first.")

    session = get_session(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Start a new interview.")
    return user_text, session


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """NDJSON stream: meta → delta tokens → done. Enables speak-while-generating."""
    user_text, session = _validate_chat_request(req)

    def generate():
        try:
            for line in process_chat_stream(
                session,
                user_text,
                req.messages,
                interrupt=req.interrupt,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_KEY,
            ):
                yield line
        except HTTPException as e:
            yield json.dumps({"type": "error", "detail": e.detail}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/api/tts")
async def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is required")

    ssml = build_ssml(text, TTS_VOICE)
    if not ssml:
        raise HTTPException(400, "text is empty after sanitization")

    try:
        token = get_cached_speech_token()
    except Exception as e:
        raise HTTPException(502, f"Failed to get speech token: {e}")

    try:
        audio_resp = _http.post(
            f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": TTS_OUTPUT_FORMAT,
            },
            data=ssml.encode("utf-8"),
            timeout=20,
        )
        audio_resp.raise_for_status()
    except Exception as e:
        raise HTTPException(502, f"TTS synthesis failed: {e}")

    return Response(
        content=audio_resp.content,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=tts.mp3"},
    )


@app.get("/api/stt-token")
async def stt_token():
    try:
        return {"token": get_cached_speech_token(), "region": AZURE_SPEECH_REGION}
    except Exception as e:
        raise HTTPException(502, f"Could not issue speech token: {e}")
