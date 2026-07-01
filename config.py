"""Load all secrets and service settings from backend/.env only."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_FILE)


def _require(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            f"Add it to {_ENV_FILE}"
        )
    return value


AZURE_SPEECH_KEY: str = _require("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION: str = _require("AZURE_SPEECH_REGION")
AZURE_OPENAI_KEY: str = _require("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT: str = _require("AZURE_OPENAI_ENDPOINT")
TTS_VOICE: str = (os.getenv("TTS_VOICE") or "en-IN-Aarti:DragonHDLatestNeural").strip()

# Comma-separated origins for static frontend, e.g. https://myapp.azurestaticapps.net
# Leave empty to allow all origins (fine for dev).
CORS_ORIGINS: str = (os.getenv("CORS_ORIGINS") or "").strip()
