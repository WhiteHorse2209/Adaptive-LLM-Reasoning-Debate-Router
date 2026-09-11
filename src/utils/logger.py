"""Structured logging utility with request context support."""

import json
import logging
import sys
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Formats log records as structured, readable text with contextual metadata."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt or "%Y-%m-%d %H:%M:%S")
        request_id = getattr(record, "request_id", None)
        req_prefix = f"[{request_id}] " if request_id else ""
        
        extra_data = {}
        for key, val in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "request_id",
            }:
                extra_data[key] = val

        extra_str = f" | {json.dumps(extra_data)}" if extra_data else ""
        return f"{timestamp} [{record.levelname:<5}] [{record.name}] {req_prefix}{record.getMessage()}{extra_str}"


def get_logger(name: str = "router", level: int = logging.INFO) -> logging.Logger:
    """Retrieve or configure a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
