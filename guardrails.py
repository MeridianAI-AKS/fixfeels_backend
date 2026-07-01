"""Off-script guardrails — KB-only answers, agent transfer, safe replies."""

from __future__ import annotations

import re

FSHIP_WEBSITE = "https://fship.in/"

GUARDRAIL_RULES = (
    "Guardrails (always follow):\n"
    "- Answer ONLY from the FSHIP KNOWLEDGE BASE in this prompt — no guessing or outside facts.\n"
    "- FIRST check if the knowledge base has facts for the customer's question (services, COD, "
    "tracking, pricing plans, couriers, integrations, contact, etc.) and answer directly in "
    "1-2 short Hinglish sentences.\n"
    "- Treat these as in-scope and answer from KB when possible: Fship, shipping, courier, delivery, "
    "logistics, ecommerce/D2C selling, orders, RTO, NDR, pickup, pincodes, labels, API, "
    "who you are / what Fship does.\n"
    "- If the question is about Fship but the exact detail is NOT in the knowledge base, say briefly "
    f"that exact detail is not with you right now and share {FSHIP_WEBSITE} or support@fship.in.\n"
    "- ONLY for clearly off-topic questions (weather, cricket, movies, recipes, politics, etc.) "
    "say politely that you do not have that information — do NOT use a long refusal lecture.\n"
    "- NEVER say 'main sirf Fship ki services mein madad kar sakti hoon' or similar boilerplate "
    "when the customer asked a shipping/Fship/ecommerce question — answer from KB or give the "
    "missing-detail + website reply instead.\n"
    "- Never invent rates, courier names, policies, or features not in the knowledge base.\n"
    "- Never discuss competitors negatively. Never mention script or interview.\n"
    "- Do NOT repeat 'samajh gayi' every turn — vary wording.\n"
    "- Add the website link ONLY when KB lacks the requested detail or user needs pricing/signup/docs.\n"
    "- Support (if asked, from KB): support@fship.in, +91 9999795111, 10 AM–7 PM Mon–Sat."
)

STATIC_AGENT_TRANSFER = (
    "Jee bilkul main aapki call hamare senior agent ko transfer kar sakti hoon "
    "kripya apna registered mobile number bol dijiye jaise hi number confirm ho jayega "
    "main aapko agent se connect karwa dungi"
)

STATIC_AGENT_TRANSFER_WITH_NUMBER = (
    "Jee aapka number note kar liya hai main abhi aapki call hamare senior agent ko "
    "transfer kar rahi hoon kripya line par bane rahiye"
)

_UNKNOWN_FSHIP = (
    "Ji, iski exact detail mere paas abhi nahi hai "
    f"aap {FSHIP_WEBSITE} par dekh sakte hain ya support@fship.in par likh sakte hain"
)

_UNKNOWN_UNRELATED = (
    "Ji, is topic ki jaankari mere paas nahi hai "
    "shipping ya Fship ke baare mein kuch puchna ho to main help kar sakti hoon"
)

_WEBSITE_TOPICS = re.compile(
    r"\b(pric(e|ing)|rate\s*card|charges?|tariff|cost|fees?|signup|sign\s*up|"
    r"register|registration|api\s*doc|terms|career|refund|cancellation)\b",
    re.I,
)

_FSHIP_SCOPE = re.compile(
    r"\b(fship|preeti|ship(ping)?|courier|deliver(y|ies)|logistics|cod|cash\s*on\s*delivery|"
    r"tracking|pickup|pincode|pin\s*code|label|bulkit|ndd|sdd|aggregator|"
    r"warehouse|manifest|rto|reverse|return\s*ship|ndr|remittance|ecommerce|e-commerce|"
    r"d2c|seller|orders?|shipment|freight|air\s*ship|surface|shopify|woocommerce|"
    r"integration|api|pricing|rate|charges?|signup|register|bulky|b2b|awb|"
    r"kaun ho|kya karti|kya karte|services?|features?)\b",
    re.I,
)

_UNRELATED_SCOPE = re.compile(
    r"\b(weather|mausam|cricket|football|movie|film|recipe|politics|election|"
    r"stock\s*market|bitcoin|crypto|joke|poem|homework|math\s*problem)\b",
    re.I,
)


def unknown_knowledge_reply(user_text: str) -> str:
    """Fallback when the question is outside KB scope or clearly unrelated."""
    text = (user_text or "").strip()
    if is_fship_related(text) or _WEBSITE_TOPICS.search(text):
        return _UNKNOWN_FSHIP
    return _UNKNOWN_UNRELATED


def is_fship_related(text: str) -> bool:
    return bool(_FSHIP_SCOPE.search(text or ""))


def is_clearly_unrelated(text: str) -> bool:
    """True when the question is obviously not about Fship/shipping."""
    t = (text or "").strip()
    if not t:
        return False
    if is_fship_related(t):
        return False
    if _WEBSITE_TOPICS.search(t):
        return False
    return bool(_UNRELATED_SCOPE.search(t))


def needs_website_link(user_text: str, reply: str) -> bool:
    """Whether to append fship.in (pricing, missing KB detail, signup)."""
    if _WEBSITE_TOPICS.search(user_text or ""):
        return True
    lower = (reply or "").lower()
    missing_phrases = (
        "jaankari nahi",
        "jaankari mere paas",
        "detail nahi",
        "exact nahi",
        "nahi hai mere paas",
        "help nahi kar sakti",
        "pata nahi",
    )
    return any(p in lower for p in missing_phrases)


_REFUSAL_BOILERPLATE = re.compile(
    r"main\s+sirf\s+fship|sirf\s+fship\s+ki\s+services|"
    r"shipping[- ]related\s+queries\s+mein\s+hi\s+madad|"
    r"kisi\s+aur\s+topic\s+mein\s+assist\s+nahi",
    re.I,
)


def _strip_refusal_boilerplate(reply: str, user_text: str) -> str:
    """Replace generic LLM refusals with a helpful KB-miss or scope reply."""
    if not _REFUSAL_BOILERPLATE.search(reply or ""):
        return reply
    if is_fship_related(user_text) or _WEBSITE_TOPICS.search(user_text or ""):
        return _UNKNOWN_FSHIP
    if is_clearly_unrelated(user_text):
        return _UNKNOWN_UNRELATED
    return _UNKNOWN_FSHIP


def ensure_fship_link(text: str, user_text: str = "") -> str:
    """Append website only when KB lacks detail or user asked pricing/signup/docs."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip()).strip()
    if not cleaned:
        return unknown_knowledge_reply(user_text)

    cleaned = _strip_refusal_boilerplate(cleaned, user_text)

    if re.search(r"fship\.in", cleaned, re.I):
        return cleaned

    if needs_website_link(user_text, cleaned):
        return f"{cleaned.rstrip('. ')} aap {FSHIP_WEBSITE} par details dekh sakte hain"

    return cleaned


def has_phone_number(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    return len(digits) == 10 and digits[0] in "6789"


def agent_transfer_reply(user_text: str) -> str:
    if has_phone_number(user_text):
        return STATIC_AGENT_TRANSFER_WITH_NUMBER
    return STATIC_AGENT_TRANSFER


def is_human_agent_request(text: str) -> bool:
    lower = text.strip().lower()
    if re.search(
        r"\b(agent|representative|human|manager|supervisor|executive|advisor)\b",
        lower,
    ):
        return True
    if re.search(
        r"(insaan|aadmi|kisi se|agent se|manager se|senior se).*(baat|connect|milna|transfer)",
        lower,
    ):
        return True
    if re.search(
        r"(transfer|callback|call back|call kara|connect kara|line pe)",
        lower,
    ) and re.search(r"(agent|human|manager|senior|aadmi|insaan)", lower):
        return True
    if re.search(r"(senior|manager).*(baat|connect|transfer|chahiye)", lower):
        return True
    if re.search(r"agent.*(chahiye|chahie|dedo|de do|se baat)", lower):
        return True
    return False
