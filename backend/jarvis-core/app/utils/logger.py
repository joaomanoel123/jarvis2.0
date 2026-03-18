"""
app/utils/logger.py
Structured, color-coded logger shared by every JARVIS module.
"""

import logging
import os
import sys
from datetime import datetime, timezone

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_FMT = "%(asctime)s | %(levelname)-8s | %(name)-26s | %(message)s"
_DATE = "%H:%M:%S"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=_FMT,
    datefmt=_DATE,
    stream=sys.stdout,
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
