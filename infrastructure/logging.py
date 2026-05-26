from __future__ import annotations

import logging
import sys

from infrastructure.settings import settings

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        level = getattr(logging, settings.log_level.upper(), logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root = logging.getLogger("reeftwin")
        root.setLevel(level)
        root.addHandler(handler)
        root.propagate = False
        _configured = True

    return logging.getLogger(f"reeftwin.{name}")
