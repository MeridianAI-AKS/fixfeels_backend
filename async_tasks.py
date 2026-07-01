"""Defer slow disk writes so voice responses are not blocked."""

from __future__ import annotations

import threading
from typing import Any, Callable


def defer(fn: Callable[..., Any], **kwargs: Any) -> None:
    threading.Thread(target=lambda: fn(**kwargs), daemon=True).start()
