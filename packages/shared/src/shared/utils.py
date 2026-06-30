"""Shared cross-package utilities."""

from __future__ import annotations

from datetime import datetime


def now_iso() -> str:
    """Return current local time (UTC+8) as ISO 8601 string."""
    return datetime.now().astimezone().isoformat()
