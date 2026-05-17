"""
Entry point.

    python main.py            # serve HTTP API
    uvicorn app.api:app       # equivalent, if you prefer the uvicorn CLI

Loads .env, starts uvicorn with the FastAPI app.
"""

from __future__ import annotations

import uvicorn
from dotenv import load_dotenv

from app.config import get_settings


def main() -> None:
    load_dotenv()
    settings = get_settings()
    uvicorn.run(
        "app.api:app",
        host=settings.host,
        port=settings.port,
        log_config=None,  # we use structlog
        access_log=False,
    )


if __name__ == "__main__":
    main()
