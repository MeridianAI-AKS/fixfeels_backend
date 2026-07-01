"""Heuristic intent detection — no LLM latency."""

from __future__ import annotations

import re

from guardrails import is_human_agent_request

STATIC_INVITE = "Ji bilkul, aap apna sawal puchhiye."

_QUESTION_START = re.compile(
    r"^(kya|kaise|kyun|kab|kahan|kaun|what|how|why|when|where|who|"
    r"tell me|explain|batao|bataiye|bataye|bata dena|can you|do you|"
    r"is|are|does|do|can|could)\b",
    re.I,
)

_ANSWER_HINT = re.compile(
    r"^(haan|han|ha|ji|nahi|na\b|theek|okay|ok\b|main\b|mera\b|mere\b|"
    r"hum\b|hamare\b|delhi|mumbai|bangalore|bengaluru|pune|hyderabad|"
    r"chennai|kolkata|ahmedabad|jaipur|lucknow|noida|gurgaon|gurugram|\d+)",
    re.I,
)

_CLAUSE_SPLIT = re.compile(
    r"[,;]| aur | and | lekin | par | but | बट | और | लेकिन ",
    re.I,
)

_REPEAT_ANSWER = re.compile(
    r"(bataya|bola|kah|kaha|already|pehle|phir se|dobara|wahi|same|again|repeat|"
    r"बताया|बोला|फिर|दोबारा|वही)",
    re.I,
)

_REPEAT_QUESTION = re.compile(
    r"(sawal|question|pooch|puch|kyon|kyun|why|पूछ)",
    re.I,
)

_REPEAT_AGENT = re.compile(
    r"(repeat|dubara|dohrao?|phir se|fir se|wapas|vapis|"
    r"sunai nahi|samajh nahi aaya|kya bola|kya kaha|kya boli|kya kah[ai]|"
    r"what did you say|say again|bolo phir|fir se bol)",
    re.I,
)

_CUSTOMER_ALREADY_ANSWERED = re.compile(
    r"\b(maine|main to|mera jawab|humne|hamne|pehle hi)\b.*\b(bataya|bola|kah|kaha)\b",
    re.I,
)


def is_pure_question_only(text: str) -> bool:
    """True when the utterance is only a question (no script answer bundled)."""
    t = text.strip()
    if not t:
        return False
    if _QUESTION_START.match(t):
        return True
    has_question = t.endswith("?") or bool(
        re.search(r"\b(kya|kaise|kyun|kab|kahan)\b.*\b(hai|hain|ho|hoti|hota|milega)\b", t, re.I)
    )
    if not has_question:
        return False
    if _CLAUSE_SPLIT.search(t):
        return False
    if _ANSWER_HINT.search(t):
        return False
    return True


def is_repeat_request(user_text: str) -> bool:
    """Customer wants the agent to repeat what was just said."""
    t = user_text.strip()
    if not t:
        return False
    lower = t.lower()
    if _CUSTOMER_ALREADY_ANSWERED.search(t):
        return False
    if re.search(r"\b(aapne|aap ne|tumne|aap kya|tum kya)\b", lower):
        return True
    if _REPEAT_AGENT.search(t):
        return True
    if re.match(r"^(repeat|dubara|phir se|fir se|wapas|vapis)\b", lower):
        return True
    if re.search(r"\b(kya bola|kya kaha|kya boli)\b", lower):
        return True
    return False


def is_script_reanswer(user_text: str, pending_question: str | None) -> bool:
    """Customer is re-stating a script answer (often upset about repeat questions)."""
    if not pending_question:
        return False
    t = user_text.strip()
    if not t:
        return False
    if is_repeat_request(t):
        return False
    if _CUSTOMER_ALREADY_ANSWERED.search(t):
        return True
    if _REPEAT_ANSWER.search(t) and _REPEAT_QUESTION.search(t):
        if re.search(r"\b(aapne|aap ne|tumne|kya bola|kya kaha)\b", t, re.I):
            return False
        if re.search(r"\b(maine|mera|humne|hamne)\b", t, re.I):
            return True
    if re.search(r"\d+", t) and re.search(
        r"(order|shipment|monthly|kitne|volume|quantity)",
        pending_question,
        re.I,
    ):
        return not is_pure_question_only(t) and not is_repeat_request(t)
    return False


def has_embedded_question(text: str) -> bool:
    """True when the message contains a question, not only a script answer."""
    t = text.strip()
    if not t:
        return False
    if is_pure_question_only(t):
        return False
    if t.endswith("?"):
        return True
    if re.search(
        r"\b(kya|kaise|kyun|kab|kahan|kaun)\b.*\b(hai|hain|ho|hota|hoti|sakte|skte|milega|dena|"
        r"provide|bataye|batao|bata\s*sakte|kar\s*sakte|de\s*sakte|differ)\b",
        t,
        re.I,
    ):
        return True
    if re.search(
        r"(bata\s*sakte|provide\s*kar|de\s*sakte|diff(er)?|services?\s*provide|"
        r"प्रोवाइड|बता\s*सकते|कर\s*सकते|दे\s*सकते)",
        t,
        re.I,
    ):
        return True
    return bool(re.search(r"\b(kya|kaise|kyun|kab|kaun)\b", t, re.I))


def extract_question_part(text: str) -> str | None:
    """Best-effort split: script answer first, customer question in later clause."""
    t = text.strip()
    if not t:
        return None
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(t) if p.strip()]
    if len(parts) < 2:
        return t if has_embedded_question(t) and not is_pure_question_only(t) else None
    for part in reversed(parts):
        if has_embedded_question(part) or part.endswith("?"):
            return part
    return parts[-1]


def extract_answer_part(text: str) -> str | None:
    """Answer clause when user bundles answer + question in one utterance."""
    t = text.strip()
    if not t:
        return None
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(t) if p.strip()]
    if len(parts) < 2:
        return t if not has_embedded_question(t) else None
    for part in parts:
        if not has_embedded_question(part) and not part.endswith("?"):
            if len(part) >= 2:
                return part
    return parts[0] if not has_embedded_question(parts[0]) else None


_FILLER_ONLY = re.compile(
    r"^(hmm+|umm+|uh+|ok+|okay+|theek hai|theek|achha|right|suniye|ruko|wait)\.?$",
    re.I,
)

_ANSWER_SHORT = re.compile(r"^(haan|han|ha|ji|nahi|na|yes|no)\b", re.I)


def is_valid_script_answer(user_text: str, pending_question: str | None = None) -> bool:
    """True only when the customer actually answered the pending script question."""
    t = user_text.strip()
    if not t:
        return False

    if is_repeat_request(t):
        return False

    if is_script_reanswer(t, pending_question):
        return True

    if is_pure_question_only(t):
        return False

    intent = quick_intent(t)
    if intent in ("off_script_question", "wants_to_ask", "wants_human_agent", "wants_repeat"):
        return False

    if _FILLER_ONLY.match(t):
        return False

    if is_combined_answer_and_question(t):
        answer_part = extract_answer_part(t)
        return bool(
            answer_part
            and len(answer_part.strip()) >= 2
            and not is_pure_question_only(answer_part)
        )

    if has_embedded_question(t):
        return False

    if re.search(r"\d+", t):
        return True
    if _ANSWER_SHORT.match(t):
        return True
    return len(t) >= 3


def reask_pending_message(pending_question: str) -> str:
    """Re-ask the same script question without advancing."""
    pending = pending_question.strip()
    if not pending:
        return "Ji, kripya is sawal ka jawab dijiye."
    return f"Ji, pehle yeh batayiye — {pending}"


def repeat_agent_message(
    *,
    pending_question: str | None,
    last_assistant_line: str | None,
) -> str:
    """Repeat what the agent last said — do not advance the script."""
    pending = (pending_question or "").strip()
    if pending:
        return f"Ji, maine pucha tha — {pending}"
    last = (last_assistant_line or "").strip()
    if last:
        return f"Ji, maine kaha tha — {last}"
    return "Ji, main dubara bolti hoon — ek moment."


def is_combined_answer_and_question(text: str) -> bool:
    """User gave a script answer and asked something in the same utterance."""
    t = text.strip()
    if not t:
        return False
    if is_pure_question_only(t):
        return False
    return has_embedded_question(t)


def quick_intent(user_text: str, *, interrupt: bool = False) -> str:
    """
    Returns: answer | wants_to_ask | wants_human_agent | off_script_question
    Default is answer (keeps script call moving fast).
    """
    text = user_text.strip()
    lower = text.lower()

    if interrupt:
        if is_human_agent_request(text):
            return "wants_human_agent"
        return "off_script_question"

    if is_human_agent_request(text):
        return "wants_human_agent"

    if is_repeat_request(text):
        return "wants_repeat"

    if re.search(
        r"(sawal|question|baat).*(karna tha|puchna tha|poochhna tha|puchni thi|puchna hai|"
        r"karna hai|puch sakta|puch sakti|puchna chahta|puchna chahti)",
        lower,
    ):
        return "wants_to_ask"
    if re.search(r"^(mujhe|main).*(sawal|question|baat).*(karna|puch)", lower):
        return "wants_to_ask"
    if re.search(r"^(mujhe|main)\b.*\b(puchna|poochhna|sawal)\b", lower):
        return "wants_to_ask"
    if re.search(r"\b(puchna hai|poochhna hai|sawal puch)\b", lower):
        return "wants_to_ask"
    if re.search(r"^(kuch|ek).*(puchna|poochhna|sawal)", lower) and not text.endswith("?"):
        return "wants_to_ask"

    if is_combined_answer_and_question(text):
        return "answer_and_question"

    if is_pure_question_only(text):
        return "off_script_question"
    if text.endswith("?"):
        return "off_script_question"
    if re.match(
        r"^(kya|kaise|kyun|kab|kahan|kaun|what|how|why|when|where|who|tell me|explain|"
        r"batao|bataiye|bataye|bata dena|can you|do you)\b",
        text,
        re.I,
    ):
        return "off_script_question"

    return "answer"
