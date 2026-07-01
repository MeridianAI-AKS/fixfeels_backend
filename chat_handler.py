"""Chat logic + streaming for script calls with context memory."""

from __future__ import annotations

import json
import re
from typing import Generator, Iterator

import requests
from fastapi import HTTPException

from async_tasks import defer
from guardrails import (
    GUARDRAIL_RULES,
    agent_transfer_reply,
    ensure_fship_link,
    is_clearly_unrelated,
    unknown_knowledge_reply,
)
from humanize import fallback_transition
from intent_fast import (
    STATIC_INVITE,
    extract_answer_part,
    extract_question_part,
    has_embedded_question,
    is_combined_answer_and_question,
    is_repeat_request,
    is_script_reanswer,
    is_valid_script_answer,
    quick_intent,
    reask_pending_message,
    repeat_agent_message,
)
from knowledge_base import knowledge_for_prompt
from qa_store import save_off_script_qa, save_script_qa
from session_manager import SessionState

_openai_http = requests.Session()

GENERAL_SYSTEM_PROMPT = (
    "You are Preeti from Fship, a polite Indian voice assistant on a live sales call. "
    "Use a formal, soft and polite tone like a professional Fship call-centre agent. "
    "Address the customer respectfully with ji, Sir, or Ma'am where natural. "
    "Sound warm, calm and helpful — never robotic or abrupt. "
    "Never mention script, interview, or going back to the script. "
    "Use the conversation history for context — remember what was already discussed. "
    "If the customer gave a script answer AND asked a Fship question in the same message, "
    "answer their Fship question from the knowledge base first (1-2 sentences). "
    "Do not skip their question. Do not only move to the next script line. "
    "Answer customer questions ONLY from the FSHIP KNOWLEDGE BASE — never from general knowledge. "
    "If the knowledge base has relevant facts, answer them directly — never reply with a generic "
    "'main sirf Fship services mein madad kar sakti hoon' refusal for shipping or Fship questions. "
    "If the exact detail is not in the knowledge base, say that briefly and share fship.in or support. "
    "Answer ONLY what the customer asked — nothing extra. "
    "Be concise: 1 short sentence, maximum 2 if truly needed (about 25 words total). "
    "Do not invite them to ask more questions. "
    "Do not ask the next script question — the system appends it after you. "
    "Answer in Hindi/Hinglish. Do not use bullet points. "
    "Never use markdown, backslashes, emojis, or stage directions like (smile). "
    "Write plain natural spoken Hinglish — normal spelling (main, ji, nahi, rahi) is fine.\n\n"
    f"{GUARDRAIL_RULES}"
)

INTERRUPT_SYSTEM_PROMPT = (
    "You are Preeti from Fship on a live call. The customer interrupted you mid-sentence. "
    "Answer their new question ONLY from the FSHIP KNOWLEDGE BASE in context. "
    "If not in the knowledge base, say you do not have that information. "
    "Be concise (1-2 short sentences), polite, Hindi/Hinglish. "
    "Do not mention interruption or the script. "
    "No markdown, backslashes, emojis, or (smile). Natural Hinglish spelling.\n\n"
    f"{GUARDRAIL_RULES}"
)


def resume_with_pending_question(answer: str, pending_question: str) -> str:
    """Resume script after off-script answer — no 'ji' after website link."""
    answer = answer.rstrip(".!? ").strip()
    pending = pending_question.strip()
    if not pending:
        return answer
    return f"{answer} {pending}"


def _last_assistant_line(session: SessionState) -> str | None:
    for turn in reversed(session.conversation):
        if turn.get("role") == "assistant" and turn.get("content", "").strip():
            return turn["content"].strip()
    return None


def _stream_repeat_reply(session: SessionState, user_text: str) -> Generator[str, None, None]:
    reply = repeat_agent_message(
        pending_question=session.pending_question,
        last_assistant_line=_last_assistant_line(session),
    )
    session.append_turn("user", user_text)
    session.append_turn("assistant", reply)
    yield from _stream_instant(reply, {
        "source": "script",
        "validated_as": "wants_repeat",
        "script_question": session.pending_question,
        "interview_complete": False,
        "stored": False,
    })


def build_opening_reply(session: SessionState) -> str:
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


def _conversation_history(
    session: SessionState,
    client_messages: list,
    user_text: str,
) -> list[dict[str, str]]:
    """Merge session + client history; drop duplicate current user turn."""
    client: list[dict[str, str]] = []
    for m in client_messages or []:
        role = m.role if hasattr(m, "role") else m.get("role")
        content = m.content if hasattr(m, "content") else m.get("content")
        if role and content:
            client.append({"role": role, "content": str(content).strip()})

    if client and user_text and client[-1]["role"] == "user":
        if client[-1]["content"].strip() == user_text.strip():
            client = client[:-1]

    server = list(session.conversation)
    merged = client if len(client) >= len(server) else server
    return merged[-16:]


def _context_messages(
    session: SessionState,
    client_messages: list,
    user_text: str,
    *,
    interrupt: bool = False,
) -> list[dict]:
    system = INTERRUPT_SYSTEM_PROMPT if interrupt else GENERAL_SYSTEM_PROMPT
    kb = knowledge_for_prompt()
    if kb:
        system = f"{system}\n\n{kb}"
    messages: list[dict] = [{"role": "system", "content": system}]

    history = _conversation_history(session, client_messages, user_text)
    for turn in history:
        messages.append(turn)

    if session.script_answers:
        collected = "\n".join(
            f"- Script Q: {item['question']} | Customer already answered: {item['answer']}"
            for item in session.script_answers[-10:]
        )
        messages.append({
            "role": "system",
            "content": (
                "Already collected from this customer (NEVER re-ask these; treat as facts):\n"
                f"{collected}"
            ),
        })

    if session.pending_question:
        messages.append({
            "role": "system",
            "content": (
                f"Pending script question (customer must answer before the call moves on): "
                f"{session.pending_question} "
                "If the customer only asked a Fship question or did not answer this, "
                "answer their Fship question only — do not skip ahead."
            ),
        })

    if is_script_reanswer(user_text, session.pending_question):
        messages.append({
            "role": "system",
            "content": (
                "The customer is saying they ALREADY answered the pending script question. "
                "Apologize briefly for repeating, confirm you noted their answer, "
                "and do NOT ask the same question again."
            ),
        })

    if is_combined_answer_and_question(user_text):
        q_part = extract_question_part(user_text) or user_text
        messages.append({
            "role": "system",
            "content": (
                "The customer answered the script question AND asked a Fship question "
                f"in the same message. Their question to answer now: \"{q_part}\". "
                "Answer ONLY this question from the knowledge base using the facts available. "
                "Do not refuse with a generic scope message — answer helpfully or say exact "
                "detail is not in KB and share fship.in."
            ),
        })

    if session.pending_question and quick_intent(user_text) in (
        "off_script_question",
        "answer_and_question",
    ):
        messages.append({
            "role": "system",
            "content": (
                "Side question during the sales call — answer from the knowledge base if possible. "
                "Do NOT say you only help with Fship services; give a direct helpful answer."
            ),
        })

    messages.append({"role": "user", "content": user_text})
    return messages


def stream_openai(
    messages: list[dict],
    *,
    endpoint: str,
    api_key: str,
    max_tokens: int = 100,
    temperature: float = 0.35,
) -> Iterator[str]:
    try:
        resp = _openai_http.post(
            endpoint,
            headers={"Content-Type": "application/json", "api-key": api_key},
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
            timeout=10,
            stream=True,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise HTTPException(502, "OpenAI request timed out")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(502, f"OpenAI error: {e.response.status_code}")

    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data: "):
            continue
        payload = raw[6:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content") or ""
            if text:
                yield text
        except (json.JSONDecodeError, KeyError, IndexError):
            continue


def _emit_meta(meta: dict) -> str:
    return json.dumps({"type": "meta", **meta}, ensure_ascii=False) + "\n"


def _emit_delta(text: str) -> str:
    return json.dumps({"type": "delta", "text": text}, ensure_ascii=False) + "\n"


def _emit_done(reply: str) -> str:
    return json.dumps({"type": "done", "reply": reply}, ensure_ascii=False) + "\n"


def _stream_instant(reply: str, meta: dict) -> Generator[str, None, None]:
    yield _emit_meta({**meta, "streaming": False})
    yield _emit_delta(reply)
    yield _emit_done(reply)


def _instant_general_reply(
    session: SessionState,
    user_text: str,
    reply: str,
    validated_as: str,
    *,
    interview_complete: bool = False,
) -> Generator[str, None, None]:
    defer(
        save_off_script_qa,
        session_id=session.session_id,
        user_question=user_text,
        assistant_response=reply,
    )
    session.append_turn("user", user_text)
    session.append_turn("assistant", reply)
    yield from _stream_instant(reply, {
        "source": "general",
        "validated_as": validated_as,
        "interview_complete": interview_complete,
        "pending_script_question": session.pending_question,
        "stored": True,
    })


def _capture_script_answer_if_needed(
    session: SessionState,
    user_text: str,
) -> str | None:
    """Store script answer only when the customer clearly answered the pending question."""
    if not session.pending_question:
        return None
    pending = session.pending_question

    if is_script_reanswer(user_text, pending):
        session.record_script_answer(pending, user_text)
        defer(
            save_script_qa,
            session_id=session.session_id,
            script_question=pending,
            user_response=user_text,
        )
        return pending

    if is_combined_answer_and_question(user_text):
        answer_part = extract_answer_part(user_text)
        if answer_part and is_valid_script_answer(answer_part, pending):
            session.record_script_answer(pending, user_text)
            defer(
                save_script_qa,
                session_id=session.session_id,
                script_question=pending,
                user_response=user_text,
            )
            return pending
        return None

    return None


def _append_next_script_after_kb(
    session: SessionState,
    kb_reply: str,
    *,
    script_advanced: bool,
) -> tuple[str, bool]:
    """After KB answer, resume script — next question if answer was captured, else re-ask pending."""
    if script_advanced:
        if session.is_script_exhausted():
            session.interview_complete = True
            return kb_reply, True
        next_line, complete = build_next_reply(session)
        if complete:
            session.interview_complete = True
        if next_line:
            ack_index = sum(1 for t in session.conversation if t["role"] == "assistant")
            transition = fallback_transition(next_line, ack_index=ack_index)
            return f"{kb_reply.rstrip('. ')}. {transition}", complete
        return kb_reply, complete

    if session.pending_question:
        return resume_with_pending_question(kb_reply, session.pending_question), False

    return kb_reply, False


_POST_CALL_GOODBYE = re.compile(
    r"^(thanks?|thank you|dhanyawad|dhanyavaad|shukriya|ok|okay|bye|goodbye|theek|haan ji)\b",
    re.I,
)


def _is_post_call_goodbye(user_text: str) -> bool:
    return bool(_POST_CALL_GOODBYE.match(user_text.strip())) and quick_intent(user_text) == "answer"


def _stream_post_call_reply(
    session: SessionState,
    user_text: str,
    client_messages: list,
    *,
    endpoint: str,
    api_key: str,
) -> Generator[str, None, None]:
    """After script closing, answer follow-up questions from KB (not just Dhanyawad)."""
    if _is_post_call_goodbye(user_text):
        reply = "Dhanyawad ji, hamari team jald hi aap se contact karegi."
        session.append_turn("user", user_text)
        session.append_turn("assistant", reply)
        yield from _stream_instant(reply, {
            "source": "general",
            "validated_as": "post_call_goodbye",
            "interview_complete": True,
            "pending_script_question": None,
            "stored": False,
        })
        return

    if quick_intent(user_text) == "wants_human_agent":
        yield from _instant_general_reply(
            session,
            user_text,
            agent_transfer_reply(user_text),
            "wants_human_agent",
            interview_complete=True,
        )
        return

    yield from _stream_kb_answer(
        session,
        user_text,
        client_messages,
        interrupt=False,
        endpoint=endpoint,
        api_key=api_key,
        meta={
            "source": "general",
            "validated_as": "post_call_question",
            "interview_complete": True,
            "pending_script_question": None,
            "stored": True,
            "interrupt": False,
        },
    )


def _stream_kb_answer(
    session: SessionState,
    user_text: str,
    client_messages: list,
    *,
    interrupt: bool,
    endpoint: str,
    api_key: str,
    meta: dict,
) -> Generator[str, None, None]:
    """Answer user questions from knowledge base only; decline unknown topics."""
    script_advanced = bool(_capture_script_answer_if_needed(session, user_text))

    if is_clearly_unrelated(user_text):
        reply, complete = _append_next_script_after_kb(
            session,
            unknown_knowledge_reply(user_text),
            script_advanced=script_advanced,
        )
        defer(
            save_off_script_qa,
            session_id=session.session_id,
            user_question=user_text,
            assistant_response=reply,
        )
        session.append_turn("user", user_text)
        session.append_turn("assistant", reply)
        yield from _stream_instant(reply, {
            **meta,
            "streaming": False,
            "validated_as": "unknown_topic",
            "interview_complete": complete,
            "script_advanced": script_advanced,
        })
        return

    messages = _context_messages(
        session, client_messages, user_text, interrupt=interrupt
    )
    parts: list[str] = []
    yield _emit_meta({**meta, "streaming": True, "script_advanced": script_advanced})
    for token in stream_openai(messages, endpoint=endpoint, api_key=api_key):
        parts.append(token)
        yield _emit_delta(token)

    kb_reply = ensure_fship_link("".join(parts).strip(), user_text)

    reply, complete = _append_next_script_after_kb(
        session, kb_reply, script_advanced=script_advanced
    )
    defer(
        save_off_script_qa,
        session_id=session.session_id,
        user_question=user_text,
        assistant_response=reply,
    )
    if reply != kb_reply:
        suffix = reply[len(kb_reply) :].lstrip()
        if suffix:
            yield _emit_delta(suffix if suffix.startswith(" ") else f" {suffix}")

    session.append_turn("user", user_text)
    session.append_turn("assistant", reply)
    yield _emit_done(reply)


def process_chat_stream(
    session: SessionState,
    user_text: str,
    client_messages: list,
    *,
    interrupt: bool,
    endpoint: str,
    api_key: str,
) -> Generator[str, None, None]:
    if session.awaiting_customer_question or interrupt:
        if quick_intent(user_text, interrupt=interrupt) == "wants_human_agent":
            yield from _instant_general_reply(
                session,
                user_text,
                agent_transfer_reply(user_text),
                "wants_human_agent",
            )
            return

        if is_repeat_request(user_text):
            yield from _stream_repeat_reply(session, user_text)
            return

        was_awaiting = session.awaiting_customer_question
        if was_awaiting:
            session.awaiting_customer_question = False

        yield from _stream_kb_answer(
            session,
            user_text,
            client_messages,
            interrupt=interrupt or was_awaiting,
            endpoint=endpoint,
            api_key=api_key,
            meta={
                "source": "general",
                "validated_as": "off_script_question",
                "interview_complete": False,
                "pending_script_question": session.pending_question,
                "stored": True,
                "interrupt": interrupt,
            },
        )
        return

    if is_repeat_request(user_text):
        yield from _stream_repeat_reply(session, user_text)
        return

    if session.is_script_exhausted() and not session.pending_question:
        yield from _stream_post_call_reply(
            session,
            user_text,
            client_messages,
            endpoint=endpoint,
            api_key=api_key,
        )
        return

    if session.pending_question:
        intent = quick_intent(user_text, interrupt=interrupt)
        if intent == "answer" and has_embedded_question(user_text):
            intent = "answer_and_question"

        if intent == "wants_to_ask":
            session.awaiting_customer_question = True
            defer(
                save_off_script_qa,
                session_id=session.session_id,
                user_question=user_text,
                assistant_response=STATIC_INVITE,
            )
            session.append_turn("user", user_text)
            session.append_turn("assistant", STATIC_INVITE)
            yield from _stream_instant(STATIC_INVITE, {
                "source": "general",
                "validated_as": "wants_to_ask",
                "interview_complete": False,
                "pending_script_question": session.pending_question,
                "stored": True,
            })
            return

        if intent == "wants_human_agent":
            reply = agent_transfer_reply(user_text)
            if session.pending_question:
                reply = resume_with_pending_question(reply, session.pending_question)
            yield from _instant_general_reply(
                session,
                user_text,
                reply,
                "wants_human_agent",
            )
            return

        if intent in ("off_script_question", "answer_and_question"):
            yield from _stream_kb_answer(
                session,
                user_text,
                client_messages,
                interrupt=False,
                endpoint=endpoint,
                api_key=api_key,
                meta={
                    "source": "general",
                    "validated_as": intent,
                    "interview_complete": False,
                    "pending_script_question": session.pending_question,
                    "stored": True,
                    "interrupt": False,
                },
            )
            return

        if is_script_reanswer(user_text, session.pending_question):
            answered_question = session.pending_question or ""
            session.record_script_answer(answered_question, user_text)
            validated_as = "answer"
        elif is_valid_script_answer(user_text, session.pending_question):
            answered_question = session.pending_question or ""
            session.record_script_answer(answered_question, user_text)
            validated_as = "answer"
        else:
            pending = session.pending_question or ""
            reply = reask_pending_message(pending)
            session.append_turn("user", user_text)
            session.append_turn("assistant", reply)
            yield from _stream_instant(reply, {
                "source": "script",
                "validated_as": "no_answer_reask",
                "script_question": pending,
                "interview_complete": False,
                "stored": False,
            })
            return
    else:
        validated_as = None
        answered_question = ""

    reply, complete = build_next_reply(session)
    if complete:
        session.interview_complete = True

    if validated_as == "answer" and answered_question:
        ack_index = sum(1 for t in session.conversation if t["role"] == "assistant")
        reply = fallback_transition(reply, ack_index=ack_index)
        defer(
            save_script_qa,
            session_id=session.session_id,
            script_question=answered_question,
            user_response=user_text,
        )
        session.append_turn("user", user_text)
        session.append_turn("assistant", reply)
        yield from _stream_instant(reply, {
            "source": "script",
            "validated_as": validated_as,
            "script_question": session.pending_question,
            "interview_complete": complete,
            "stored": True,
        })
        return

    yield _emit_meta({
        "source": "script",
        "validated_as": validated_as,
        "script_question": session.pending_question,
        "interview_complete": complete,
        "stored": validated_as == "answer",
    })
    yield _emit_delta(reply)

    session.append_turn("user", user_text)
    session.append_turn("assistant", reply)
    yield _emit_done(reply)
