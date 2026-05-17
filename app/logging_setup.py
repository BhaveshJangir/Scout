"""
Structured JSON logging with structlog.

Why JSON:
    * Log aggregators (Datadog, Loki, CloudWatch, ELK) parse JSON natively.
    * You can filter by run_id / request_id / event without regex.
    * Easy to ship to S3/BigQuery for offline analysis.

Every log line carries `request_id` and (when inside the agent loop) `run_id`,
which is the single most useful thing for debugging in production.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Idempotent — safe to call from app startup and from tests."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # stdlib root logger emits to stdout; structlog formats to JSON.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # pulls request_id, run_id
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
