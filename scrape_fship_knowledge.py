"""Crawl https://fship.in/ and merge into backend/data/fship_knowledge.json (append, never overwrite)."""

from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE = "https://fship.in"
OUT_PATH = Path(__file__).parent / "data" / "fship_knowledge.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FshipVoiceBot/2.0)"}

# All known internal pages (discovered from site navigation + crawl).
SEED_URLS = [
    f"{BASE}/",
    f"{BASE}/about-us",
    f"{BASE}/features",
    f"{BASE}/pricing.html",
    f"{BASE}/partners.html",
    f"{BASE}/ordertracking.html",
    f"{BASE}/ratecalculator.html",
    f"{BASE}/contact.html",
    f"{BASE}/customvalidation.html",
    f"{BASE}/brandedTrackingPage.html",
    f"{BASE}/bulkordershipping.html",
    f"{BASE}/fshipbulkit.html",
    f"{BASE}/sameday-nextday-delivery.html",
    f"{BASE}/life@fship.html",
    f"{BASE}/nextday",
    f"{BASE}/postship",
    f"{BASE}/privacy-policy",
    f"{BASE}/refund-policy",
    f"{BASE}/terms-and-conditions.html",
    f"{BASE}/documents/fship_sop.pdf",
]

SKIP_SUFFIXES = (".pdf", ".jpg", ".png", ".gif", ".zip")

MOJIBAKE = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€¦": "...",
    "â€”": "—",
    "â€“": "–",
    "Â©": "©",
    "â¹": "₹",
    "Ã—": "×",
}


def clean(text: str | None) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", str(text)).strip()
    for bad, good in MOJIBAKE.items():
        t = t.replace(bad, good)
    return t


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"https://fship.in{path}"


def discover_urls(html: str) -> set[str]:
    soup = BeautifulSoup(html, "lxml")
    found: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        full = urljoin(BASE, href)
        parsed = urlparse(full)
        if parsed.netloc.replace("www.", "") != "fship.in":
            continue
        if any(parsed.path.lower().endswith(s) for s in SKIP_SUFFIXES):
            continue
        found.add(normalize_url(full))
    return found


def fetch(session: requests.Session, url: str) -> tuple[str | None, str | None]:
    try:
        resp = session.get(url, timeout=30, headers=HEADERS, allow_redirects=True)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text, None
    except Exception as exc:
        return None, str(exc)


def extract_page(url: str, html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = clean(soup.title.string if soup.title else "")
    meta_desc = ""
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        meta_desc = clean(md["content"])

    headings: dict[str, list[str]] = {}
    for level in ("h1", "h2", "h3", "h4", "h5"):
        seen: set[str] = set()
        items: list[str] = []
        for h in soup.find_all(level):
            text = clean(h.get_text())
            if text and text not in seen:
                seen.add(text)
                items.append(text)
        headings[level] = items

    paras: list[str] = []
    seen_p: set[str] = set()
    for p in soup.find_all("p"):
        text = clean(p.get_text())
        if len(text) < 25 or text in seen_p:
            continue
        if "Copyright" in text and "Fixfeels" in text:
            continue
        seen_p.add(text)
        paras.append(text)

    summary = meta_desc or (paras[0] if paras else "")
    return {
        "url": url,
        "title": title,
        "meta_description": meta_desc,
        "summary": summary,
        "headings": headings,
        "paragraphs": paras[:40],
    }


def _dedupe_list(items: list[str]) -> list[str]:
    return list(dict.fromkeys(clean(x) for x in items if clean(x)))


def _merge_list(existing: list[Any], new: list[Any]) -> list[Any]:
    out = list(existing or [])
    seen = {json.dumps(x, sort_keys=True) if isinstance(x, dict) else str(x) for x in out}
    for item in new or []:
        key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def build_structured(pages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    home = pages.get(f"{BASE}/") or {}
    about = pages.get(f"{BASE}/about-us") or {}
    features = pages.get(f"{BASE}/features") or {}
    partners = pages.get(f"{BASE}/partners.html") or {}
    pricing = pages.get(f"{BASE}/pricing.html") or {}

    services = _dedupe_list([
        "Custom Validation",
        "Branded Tracking Page",
        "Bulk Order Shipping",
        "Fship Bulkit",
        "SDD or NDD Deliveries",
    ])
    products = _dedupe_list([
        "Domestic Shipping",
        "Next Day / Same Day Delivery",
        "Postship",
        "Tracking",
        "B2B & Bulk Shipping",
        "Fship Fulfillment",
    ])
    platform_features = _dedupe_list([
        "Cash on Delivery",
        "Serviceable Pincodes",
        "Multiple Pickup Locations",
        "Print Shipping labels",
        "Email / SMS Notification",
        "API Integration",
        "NDR Management",
        "AI-Based Courier Allocation",
        "Branded Tracking Page",
        "Custom Order Validation",
        "Dedicated Account Manager",
        "Easy Channel Integration",
    ])

    courier_names = [
        "Amazon Shipping",
        "Blue Dart Air",
        "Blue Dart Express",
        "DTDC",
        "Delhivery",
        "ECOM",
        "EKART",
        "Shadowfax",
        "XpressBees",
    ]
    courier_partners_list: list[dict[str, str]] = []
    for name in courier_names:
        for para in partners.get("paragraphs") or []:
            if para.startswith(name):
                courier_partners_list.append({"name": name, "detail": para})
                break
        else:
            courier_partners_list.append({"name": name, "detail": ""})

    integrations = {
        "ecommerce_platforms": ["Shopify", "WooCommerce", "Unicommerce", "Clickpost"],
        "sales_channels_mentioned": ["Shopify", "WooCommerce", "EasyEcom", "Amazon"],
    }

    pricing_plans = [
        {"plan": "Lite", "starting_rate": "₹30/0.5Kg", "url": f"{BASE}/pricing.html"},
        {"plan": "Advanced", "starting_rate": "₹28.5/0.5Kg", "url": f"{BASE}/pricing.html"},
        {"plan": "Professional", "starting_rate": "₹27.5/0.5Kg", "url": f"{BASE}/pricing.html"},
        {"plan": "Enterprise", "starting_rate": "₹27.5/0.5Kg", "url": f"{BASE}/pricing.html"},
        {"plan": "Enterprise+", "starting_rate": "₹26.5/0.5Kg", "url": f"{BASE}/pricing.html"},
    ]

    testimonials = []
    for para in home.get("paragraphs") or []:
        if any(k in para for k in ("founder of", "Founder of", "Co-Founder", "Co-founder", "Ops Manager")):
            testimonials.append({"quote": para})
        elif "recommend Fship" in para or "recommend fship" in para.lower():
            testimonials.append({"quote": para})

    for page in pages.values():
        for para in page.get("paragraphs") or []:
            if ("recommend Fship" in para or "recommend fship" in para.lower()) and para not in [
                t["quote"] for t in testimonials
            ]:
                testimonials.append({"quote": para})

    service_pages = {
        "custom_validation": {
            "url": f"{BASE}/customvalidation.html",
            "title": "Custom Validation Rules for Smarter Shipping",
            "summary": (pages.get(f"{BASE}/customvalidation.html") or {}).get("summary", ""),
        },
        "branded_tracking": {
            "url": f"{BASE}/brandedTrackingPage.html",
            "title": "Branded Tracking Page",
            "summary": (pages.get(f"{BASE}/brandedTrackingPage.html") or {}).get("summary", ""),
        },
        "bulk_order_shipping": {
            "url": f"{BASE}/bulkordershipping.html",
            "title": "Effortless Bulk Order Shipping",
            "summary": (pages.get(f"{BASE}/bulkordershipping.html") or {}).get("summary", ""),
        },
        "fship_bulkit": {
            "url": f"{BASE}/fshipbulkit.html",
            "title": "Power Your B2B Shipments with Bulkit",
            "summary": (pages.get(f"{BASE}/fshipbulkit.html") or {}).get("summary", ""),
        },
        "sdd_ndd": {
            "url": f"{BASE}/sameday-nextday-delivery.html",
            "title": "Same Day or Next Day Deliveries",
            "summary": (pages.get(f"{BASE}/sameday-nextday-delivery.html") or {}).get("summary", ""),
        },
        "order_tracking": {
            "url": f"{BASE}/ordertracking.html",
            "title": "Track Your Order",
            "summary": (pages.get(f"{BASE}/ordertracking.html") or {}).get("summary", ""),
        },
        "rate_calculator": {
            "url": f"{BASE}/ratecalculator.html",
            "title": "Calculate your Rates",
            "summary": (pages.get(f"{BASE}/ratecalculator.html") or {}).get("summary", ""),
        },
    }

    stats = {
        "serviceable_pincodes": "29000+",
        "bulkit_pincodes": "24000+",
        "sdd_ndd_cities": "30+",
        "bulkit_courier_partners": "7+",
        "sdd_ndd_courier_partners": "5+",
    }

    navigation = {
        "about": ["Contact Us", "Life @ Fship", "About Us"],
        "services": services,
        "products": products,
        "resources": [
            "API Integration",
            "Blogs",
            "Rate calculator",
            "Track Orders",
            "Pricing",
            "Partners",
        ],
        "company_links": [
            f"{BASE}/about-us",
            f"{BASE}/contact.html",
            f"{BASE}/life@fship.html",
            f"{BASE}/terms-and-conditions.html",
            f"{BASE}/privacy-policy",
            f"{BASE}/refund-policy",
            f"{BASE}/documents/fship_sop.pdf",
        ],
    }

    policies = {
        "privacy_policy_url": f"{BASE}/privacy-policy",
        "terms_url": f"{BASE}/terms-and-conditions.html",
        "refund_policy_url": f"{BASE}/refund-policy",
        "sop_pdf_url": f"{BASE}/documents/fship_sop.pdf",
        "terms_registered_office": "Hw 08 sector 70, Noida, Uttar Pradesh, 201301 India",
        "privacy_contact": "support@fship.in",
    }

    about_text = about.get("paragraphs") or []
    meta_home = home.get("meta_description") or ""

    return {
        "source": f"{BASE}/",
        "scraped_at": date.today().isoformat(),
        "pages_crawled": sorted(pages.keys()),
        "company": {
            "name": "Fship",
            "legal_name": "Fixfeels Technologies Private Limited",
            "tagline": "Your Shipment Our Commitment",
            "summary": meta_home or (
                "AI-enabled e-commerce shipping aggregator for B2C and D2C sellers — "
                "reliable domestic shipping, COD, tracking, and API integration."
            ),
            "website": f"{BASE}/",
            "about": " ".join(about_text[:3]) if about_text else "",
            "meta_description": meta_home,
        },
        "services": services,
        "products": products,
        "platform_features": platform_features,
        "stats": stats,
        "pricing_plans": pricing_plans,
        "pricing_note": (
            "Pricing plans on https://fship.in/pricing.html — Lite from ₹30/0.5Kg, "
            "Advanced ₹28.5/0.5Kg, Professional & Enterprise ₹27.5/0.5Kg, Enterprise+ ₹26.5/0.5Kg. "
            "Use rate calculator at https://fship.in/ratecalculator.html for exact quotes."
        ),
        "courier_partners_list": courier_partners_list,
        "integrations": integrations,
        "service_pages": service_pages,
        "testimonials": testimonials[:12],
        "navigation": navigation,
        "policies": policies,
        "pages": [
            {
                "url": p["url"],
                "title": p.get("title", ""),
                "summary": p.get("summary", ""),
                "headings": p.get("headings", {}),
                "paragraphs": p.get("paragraphs", [])[:15],
            }
            for p in pages.values()
        ],
        "contact": {
            "address": "D 247/3, D Block, Sector 63, Noida, Uttar Pradesh 201301",
            "alternate_address": (
                "1018 10th floor, Galaxy blue Sapphire Plaza, Greater Noida west, "
                "Uttar Pradesh, 201009"
            ),
            "email": "support@fship.in",
            "phone": "+91 9999795111",
            "support_hours": "10 AM to 7 PM, Monday to Saturday",
        },
        "signup": f"Register or Try for Free on {BASE}/",
        "topics_for_website": _dedupe_list([
            "pricing",
            "rate card",
            "rate calculator",
            "sign up",
            "register",
            "API docs",
            "API integration",
            "terms",
            "privacy policy",
            "refund policy",
            "careers",
            "life at fship",
            "track order",
            "bulk shipping",
            "B2B shipping",
            "same day delivery",
            "next day delivery",
            "COD remittance",
            "NDR management",
            "branded tracking",
            "custom validation",
        ]),
        "features_highlights": (features.get("paragraphs") or [])[:10],
        "leadership_mentions": [
            {"name": "Raju Kumar Sinha", "role": "CBO Fship"},
        ],
    }


def merge_knowledge(existing: dict[str, Any], scraped: dict[str, Any]) -> dict[str, Any]:
    """Preserve existing keys/values; append new data and enrich empty fields."""
    merged = dict(existing)

    merged["last_appended_at"] = scraped.get("scraped_at")
    if scraped.get("pages_crawled"):
        merged["pages_crawled"] = _merge_list(
            merged.get("pages_crawled") or [], scraped["pages_crawled"]
        )

    for key in ("services", "products", "platform_features", "topics_for_website"):
        merged[key] = _merge_list(merged.get(key) or [], scraped.get(key) or [])

    # why_choose_us — never overwrite existing entries
    if "why_choose_us" not in merged:
        merged["why_choose_us"] = scraped.get("why_choose_us") or []
    elif scraped.get("why_choose_us"):
        merged["why_choose_us"] = _merge_list(merged["why_choose_us"], scraped["why_choose_us"])

    # company — keep existing strings, fill missing sub-keys
    company = dict(merged.get("company") or {})
    for k, v in (scraped.get("company") or {}).items():
        if k not in company or not company[k]:
            company[k] = v
    merged["company"] = company

    # contact — keep existing, add alternate_address if missing
    contact = dict(merged.get("contact") or {})
    for k, v in (scraped.get("contact") or {}).items():
        if k not in contact or not contact[k]:
            contact[k] = v
    merged["contact"] = contact

    # Append-only sections (merge lists, add dict keys if new)
    append_keys = (
        "pages",
        "testimonials",
        "courier_partners_list",
        "pricing_plans",
        "features_highlights",
        "leadership_mentions",
    )
    for key in append_keys:
        if scraped.get(key):
            merged[key] = _merge_list(merged.get(key) or [], scraped[key])

    # Dict sections — shallow merge, existing wins on conflict
    for key in ("stats", "integrations", "service_pages", "navigation", "policies"):
        if scraped.get(key):
            base = dict(merged.get(key) or {})
            for k, v in scraped[key].items():
                if k not in base:
                    base[k] = v
            merged[key] = base

    # Enrich pricing_note / signup / courier_partners string if empty
    for key in ("pricing_note", "signup", "courier_partners"):
        if not merged.get(key) and scraped.get(key):
            merged[key] = scraped[key]

    if merged.get("courier_partners") and scraped.get("courier_partners_list"):
        names = ", ".join(c["name"] for c in merged["courier_partners_list"] if c.get("name"))
        merged["courier_partners"] = (
            f"Trusted courier partners via Fship: {names}."
            if names
            else merged["courier_partners"]
        )

    return merged


def crawl_fship() -> dict[str, dict[str, Any]]:
    session = requests.Session()
    to_visit = {normalize_url(u) for u in SEED_URLS if not u.lower().endswith(".pdf")}
    visited: set[str] = set()
    pages: dict[str, dict[str, Any]] = {}

    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)

        html, err = fetch(session, url)
        if err or not html:
            print(f"  skip {url}: {err}")
            continue

        pages[url] = extract_page(url, html)
        print(f"  ok  {url} ({len(pages[url]['paragraphs'])} paragraphs)")

        for link in discover_urls(html):
            if link not in visited:
                to_visit.add(link)
        time.sleep(0.25)

    return pages


def load_existing() -> dict[str, Any]:
    if not OUT_PATH.exists():
        return {}
    with OUT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_knowledge(data: dict[str, Any]) -> Path:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return OUT_PATH


def scrape_and_merge() -> dict[str, Any]:
    print("Crawling fship.in …")
    pages = crawl_fship()
    scraped = build_structured(pages)
    existing = load_existing()
    merged = merge_knowledge(existing, scraped) if existing else scraped
    return merged


def main() -> None:
    data = scrape_and_merge()
    path = save_knowledge(data)
    print(f"\nMerged knowledge base -> {path}")
    print(f"  Pages in DB: {len(data.get('pages') or [])}")
    print(f"  Services: {len(data.get('services') or [])}")
    print(f"  Courier partners: {len(data.get('courier_partners_list') or [])}")


if __name__ == "__main__":
    main()
