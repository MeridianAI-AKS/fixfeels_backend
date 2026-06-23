"""
Aarti Voice Assistant — FastAPI Backend
Script interview from script.docx; LLM answers off-script questions only.
"""

from __future__ import annotations

import io
import json
import os
import re
import time

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from qa_store import list_qa, save_off_script_qa, save_script_qa
from script_loader import DEFAULT_SCRIPT_PATH, ScriptStep, load_script_steps
from session_manager import SessionState, create_session, get_session

load_dotenv()

app = FastAPI(title="Aarti Voice Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "0cdf84f4c01845cda3c5dd02933bb646")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "westus2")
AZURE_OPENAI_KEY = os.getenv(
    "AZURE_OPENAI_KEY",
    "Wl3MPy9kEHJbj3ymPiDs9H5JIQAtG5LVTvJrRBtzuTGoDvEtrVhKJQQJ99CDACYeBjFXJ3w3AAABACOGhta2",
)
AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://openai-production001.openai.azure.com/openai/deployments/gpt-4o-aiteam/chat/completions?api-version=2025-01-01-preview",
)

TTS_VOICE = "en-IN-Aarti:DragonHDLatestNeural"

TONE_GUIDE = (
    "Use a formal, soft and polite tone like a professional Fship call-centre agent. "
    "Address the customer respectfully with ji, Sir, or Ma'am where natural. "
    "Sound warm, calm and helpful — never robotic or abrupt. "
    "Never mention script, interview, or going back to the script."
)

GENERAL_SYSTEM_PROMPT = (
    "You are Preeti from Fship, a polite Indian voice assistant on a live sales call. "
    f"{TONE_GUIDE} "
    "Answer ONLY what the customer asked — nothing extra. "
    "Be concise: 1 short sentence, maximum 2 if truly needed. "
    "Do not invite them to ask more questions. Do not repeat the agent's pending question. "
    "Answer in Hindi/Hinglish. Do not use bullet points."
)

INVITE_TO_ASK_PROMPT = (
    "You are Preeti from Fship on a live call. "
    f"{TONE_GUIDE} "
    "The customer said they want to ask a question but has not asked it yet. "
    "Reply in ONE short sentence inviting them to ask — e.g. ji bilkul, aap apna sawal puchhiye."
)

VALIDATION_SYSTEM_PROMPT = (
    "You validate customer speech during an Fship shipping sales call in Hindi/Hinglish. "
    f"{TONE_GUIDE}\n"
    "Classify the customer message into exactly one intent:\n"
    "- answer: customer answered the agent's question (haan/nahi, numbers, city, courier, etc.)\n"
    "- wants_to_ask: customer wants to ask something but has NOT asked the actual question yet "
    '(e.g. "mujhe ek sawal karna tha", "kuch puchna tha")\n'
    "- off_script_question: customer asked a specific unrelated question that needs an answer\n"
    "Reply rules:\n"
    "- wants_to_ask: one short sentence inviting them to ask (reply field)\n"
    "- off_script_question: leave reply empty (answer will be generated separately)\n"
    "- answer: reply empty\n"
    'Reply ONLY JSON: {"intent":"answer"|"wants_to_ask"|"off_script_question","reply":""}'
)

SCRIPT_STEPS: list[ScriptStep] = []
_speech_token: str | None = None
_speech_token_expires: float = 0.0


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


# ── MODELS ──────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    session_id: str | None = None


class SessionStartRequest(BaseModel):
    pass


class TTSRequest(BaseModel):
    text: str


# ── SCRIPT ENGINE (instant, no LLM) ─────────────────────────────
def call_openai(messages: list[dict], *, max_tokens: int = 150, temperature: float = 0.7) -> str:
    try:
        resp = requests.post(
            AZURE_OPENAI_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "api-key": AZURE_OPENAI_KEY,
            },
            json={"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        raise HTTPException(502, "OpenAI request timed out")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(502, f"OpenAI error: {e.response.status_code}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


def answer_off_script(user_text: str, history: list[Message]) -> str:
    messages = [{"role": "system", "content": GENERAL_SYSTEM_PROMPT}]
    messages += [{"role": m.role, "content": m.content} for m in history[-4:]]
    messages.append({"role": "user", "content": user_text})
    return call_openai(messages, max_tokens=100, temperature=0.5)


def invite_customer_to_ask(user_text: str) -> str:
    messages = [
        {"role": "system", "content": INVITE_TO_ASK_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return call_openai(messages, max_tokens=60, temperature=0.4)


def resume_with_pending_question(answer: str, pending_question: str) -> str:
    """Soft formal transition back to the pending question — no script wording."""
    answer = answer.rstrip(".!? ").strip()
    return f"{answer}. Ji, {pending_question}"


def _parse_validation_json(raw: str) -> dict | None:
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
        parsed = json.loads(cleaned)
        if parsed.get("intent") in ("answer", "off_script_question", "wants_to_ask"):
            return parsed
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def quick_intent_guess(user_text: str) -> str | None:
    """Fast path — skip LLM when intent is obvious."""
    text = user_text.strip()
    lower = text.lower()

    if re.search(
        r"(sawal|question|baat).*(karna tha|puchna tha|poochhna tha|puchni thi|puchna hai|"
        r"karna hai|puch sakta|puch sakti|puchna chahta|puchna chahti)",
        lower,
    ):
        return "wants_to_ask"
    if re.search(r"^(mujhe|main).*(sawal|question|baat).*(karna|puch)", lower):
        return "wants_to_ask"
    if re.search(r"^(kuch|ek).*(puchna|poochhna|sawal)", lower) and not text.endswith("?"):
        return "wants_to_ask"

    if text.endswith("?"):
        return "off_script_question"
    if re.match(
        r"^(kya|kaise|kyun|kab|kahan|kaun|what|how|why|when|where|who|tell me|explain|"
        r"batao|bataiye|bataye|bata dena|can you|do you)\b",
        text,
        re.I,
    ):
        return "off_script_question"
    if re.search(r"\b(kya|kaise|kyun|kab|kahan|kaun)\b.+\?$", lower):
        return "off_script_question"

    if re.match(
        r"^(haan|haan ji|ji|han|yes|nahi|na|no|around|approx|lagbhag|kar raha|kar rahi|"
        r"use karta|use karti|rehta|rehti|hain|hai)\b",
        text,
        re.I,
    ):
        return "answer"
    if re.match(r"^[\d,\.\s]+$", text):
        return "answer"
    if re.match(r"^\d+\b", text):
        return "answer"
    if re.match(
        r"^(delhi|mumbai|bangalore|bengaluru|chennai|kolkata|hyderabad|pune|ahmedabad|"
        r"delhivery|bluedart|dtdc|ekart|xpressbees|shiprocket|india post|surface|air)\b",
        text,
        re.I,
    ):
        return "answer"

    return None


def validate_user_intent(user_text: str, pending_question: str) -> tuple[str, str | None]:
    """
    Returns (intent, optional_reply).
    intent: answer | wants_to_ask | off_script_question
    """
    guess = quick_intent_guess(user_text)
    if guess == "answer":
        return "answer", None
    if guess == "wants_to_ask":
        return "wants_to_ask", None
    if guess == "off_script_question":
        return "off_script_question", None

    messages = [
        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Script question: {pending_question}\n"
                f"User said: {user_text}"
            ),
        },
    ]
    raw = call_openai(messages, max_tokens=180, temperature=0.0)
    parsed = _parse_validation_json(raw)
    if parsed:
        intent = parsed["intent"]
        reply = (parsed.get("reply") or "").strip() or None
        return intent, reply

    # Safe fallback: treat as answer so interview keeps moving
    return "answer", None


def build_opening_reply(session: SessionState) -> str:
    """Speak intro lines, then stop at the first script question."""
    parts: list[str] = []
    while session.step_index < len(session.steps):
        step = session.steps[session.step_index]
        session.step_index += 1
        if step.kind == "question":
            session.pending_question = step.text
            parts.append(step.text)
            break
        parts.append(step.text)
    return " ".join(parts)


def build_next_reply(session: SessionState) -> tuple[str, bool]:
    """Advance script after user answered. Returns (reply, interview_complete)."""
    parts: list[str] = []
    while session.step_index < len(session.steps):
        step = session.steps[session.step_index]
        session.step_index += 1
        if step.kind == "question":
            session.pending_question = step.text
            parts.append(step.text)
            return " ".join(parts), False
        if step.kind == "closing":
            session.clear_pending()
            parts.append(step.text)
            return " ".join(parts), True
        parts.append(step.text)

    session.clear_pending()
    return " ".join(parts) if parts else "Dhanyawad.", True


def get_cached_speech_token() -> str:
    global _speech_token, _speech_token_expires
    now = time.time()
    if _speech_token and now < _speech_token_expires:
        return _speech_token

    token_resp = requests.post(
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


@app.get("/api/script/questions")
def get_script_questions():
    return {
        "questions": [s.text for s in SCRIPT_STEPS if s.kind == "question"],
        "steps": [{"kind": s.kind, "text": s.text} for s in SCRIPT_STEPS],
        "count": len(SCRIPT_STEPS),
    }


@app.get("/api/qa")
def get_qa_records(limit: int | None = None):
    return {"records": list_qa(limit=limit)}


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
    session.started = True
    reply = build_opening_reply(session)

    return {
        "session_id": session.session_id,
        "reply": reply,
        "source": "script",
        "script_question": session.pending_question,
        "interview_complete": False,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Validates user speech against the pending script question.
    - answer → save & next script question
    - wants_to_ask → invite only, wait for their question
    - off_script_question → concise answer, then resume pending question
    """
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

    # Customer was invited to ask — this message is their actual question
    if session.awaiting_customer_question:
        llm_reply = answer_off_script(user_text, req.messages[:-1])
        save_off_script_qa(
            session_id=session.session_id,
            user_question=user_text,
            assistant_response=llm_reply,
        )
        session.awaiting_customer_question = False
        reply = (
            resume_with_pending_question(llm_reply, session.pending_question)
            if session.pending_question
            else llm_reply
        )
        return {
            "reply": reply,
            "source": "general",
            "stored": True,
            "validated_as": "off_script_question",
            "interview_complete": False,
            "pending_script_question": session.pending_question,
        }

    if session.pending_question:
        intent, validation_reply = validate_user_intent(user_text, session.pending_question)

        if intent == "wants_to_ask":
            invite = validation_reply or invite_customer_to_ask(user_text)
            session.awaiting_customer_question = True
            save_off_script_qa(
                session_id=session.session_id,
                user_question=user_text,
                assistant_response=invite,
            )
            return {
                "reply": invite,
                "source": "general",
                "stored": True,
                "validated_as": "wants_to_ask",
                "interview_complete": False,
                "pending_script_question": session.pending_question,
            }

        if intent == "off_script_question":
            llm_reply = answer_off_script(user_text, req.messages[:-1])
            save_off_script_qa(
                session_id=session.session_id,
                user_question=user_text,
                assistant_response=llm_reply,
            )
            reply = resume_with_pending_question(llm_reply, session.pending_question)
            return {
                "reply": reply,
                "source": "general",
                "stored": True,
                "validated_as": "off_script_question",
                "interview_complete": False,
                "pending_script_question": session.pending_question,
            }

        save_script_qa(
            session_id=session.session_id,
            script_question=session.pending_question,
            user_response=user_text,
        )
        session.clear_pending()
        validated_as = "answer"
    else:
        validated_as = None

    reply, complete = build_next_reply(session)
    return {
        "reply": reply,
        "source": "script",
        "script_question": session.pending_question,
        "stored": validated_as == "answer",
        "validated_as": validated_as,
        "interview_complete": complete,
    }


@app.post("/api/tts")
async def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is required")

    try:
        token = get_cached_speech_token()
    except Exception as e:
        raise HTTPException(502, f"Failed to get speech token: {e}")

    safe_text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
    ssml = f"""<speak version="1.0" xml:lang="hi-IN">
  <voice name="{TTS_VOICE}">
    <prosody rate="5%" pitch="0%">{safe_text}</prosody>
  </voice>
</speak>"""

    try:
        audio_resp = requests.post(
            f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
            },
            data=ssml.encode("utf-8"),
            timeout=30,
        )
        audio_resp.raise_for_status()
    except Exception as e:
        raise HTTPException(502, f"TTS synthesis failed: {e}")

    return StreamingResponse(
        io.BytesIO(audio_resp.content),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=tts.mp3"},
    )


@app.get("/api/stt-token")
async def stt_token():
    try:
        return {"token": get_cached_speech_token(), "region": AZURE_SPEECH_REGION}
    except Exception as e:
        raise HTTPException(502, f"Could not issue speech token: {e}")
