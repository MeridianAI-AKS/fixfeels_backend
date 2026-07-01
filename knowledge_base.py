"""Load Fship website knowledge for off-script LLM answers."""

from __future__ import annotations

import json
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "fship_knowledge.json"
_prompt_cache: str | None = None


def load_knowledge() -> dict:
    if not _DATA_PATH.exists():
        return {}
    with _DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def reload_knowledge() -> None:
    global _prompt_cache
    _prompt_cache = None


def _append_section(lines: list[str], title: str, items: list[str]) -> None:
    if items:
        lines.append(f"- {title}: {', '.join(items)}")


def knowledge_for_prompt() -> str:
    global _prompt_cache
    if _prompt_cache is not None:
        return _prompt_cache

    data = load_knowledge()
    if not data:
        _prompt_cache = ""
        return _prompt_cache

    company = data.get("company", {})
    contact = data.get("contact", {})
    lines = [
        "FSHIP KNOWLEDGE BASE — your ONLY source of truth for customer questions:",
        "Use ONLY the facts below to answer. If a question matches these topics, answer directly — "
        "do not refuse or say you only help with Fship without trying the facts first. "
        "If a Fship detail is missing here, say you do not have that exact detail and share "
        "https://fship.in/ or support@fship.in. "
        "Only decline clearly off-topic questions (weather, sports, etc.).",
        f"- Company: {company.get('name', 'Fship')} — {company.get('summary', '')}",
        f"- Tagline: {company.get('tagline', '')}",
        f"- Website: {company.get('website', 'https://fship.in/')}",
    ]

    if company.get("about"):
        lines.append(f"- About: {company['about'][:600]}")

    _append_section(lines, "Services", data.get("services") or [])
    _append_section(lines, "Products", data.get("products") or [])
    _append_section(lines, "Platform features", data.get("platform_features") or [])

    stats = data.get("stats") or {}
    if stats:
        stat_bits = [f"{k.replace('_', ' ')}: {v}" for k, v in stats.items()]
        lines.append(f"- Coverage stats: {'; '.join(stat_bits)}")

    for item in data.get("why_choose_us") or []:
        lines.append(f"- {item.get('title', '')}: {item.get('detail', '')}")

    for partner in (data.get("courier_partners_list") or [])[:12]:
        name = partner.get("name", "")
        detail = partner.get("detail", "")
        if name:
            lines.append(f"- Courier partner {name}: {detail[:200]}")

    integrations = data.get("integrations") or {}
    if integrations.get("ecommerce_platforms"):
        lines.append(
            f"- E-commerce integrations: {', '.join(integrations['ecommerce_platforms'])}"
        )

    for plan in data.get("pricing_plans") or []:
        lines.append(
            f"- Pricing plan {plan.get('plan', '')}: starts at {plan.get('starting_rate', '')}"
        )

    service_pages = data.get("service_pages") or {}
    for key, page in service_pages.items():
        if page.get("summary"):
            lines.append(f"- {page.get('title', key)}: {page['summary'][:180]}")

    if contact:
        lines.append(
            f"- Contact: {contact.get('address', '')} | {contact.get('email', '')} | "
            f"{contact.get('phone', '')} | {contact.get('support_hours', '')}"
        )
        if contact.get("alternate_address"):
            lines.append(f"- Alternate office: {contact['alternate_address']}")

    if data.get("courier_partners"):
        lines.append(f"- {data['courier_partners']}")

    lines.append(f"- Signup: {data.get('signup', 'https://fship.in/')}")
    lines.append(f"- {data.get('pricing_note', '')}")

    policies = data.get("policies") or {}
    if policies:
        lines.append(
            f"- Policies: privacy {policies.get('privacy_policy_url', '')}, "
            f"terms {policies.get('terms_url', '')}, "
            f"refund {policies.get('refund_policy_url', '')}"
        )

    _prompt_cache = "\n".join(lines)
    return _prompt_cache
