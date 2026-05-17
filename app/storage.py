"""
SQLite-backed persistence of every run.

Why persist runs:
    * post-hoc debugging: someone reports a bad answer, you can replay
      every prompt and tool call.
    * eval datasets: build regression tests from real production traffic.
    * cost auditing: sum cost.usd over time / per user.

aiosqlite gives us the same async style as the rest of the app. Swap to
asyncpg + Postgres later by changing only this file.
"""

from __future__ import annotations

import json
from typing import Any

import aiosqlite

from .logging_setup import get_logger

log = get_logger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    question      TEXT NOT NULL,
    final_answer  TEXT,
    iterations    INTEGER NOT NULL,
    cost_usd      REAL NOT NULL,
    duration_ms   INTEGER NOT NULL,
    stopped_reason TEXT NOT NULL,
    error         TEXT,
    payload_json  TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
"""


class RunStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        log.info("runstore_ready", db_path=self._db_path)

    async def aclose(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def save(self, payload: dict[str, Any]) -> None:
        assert self._db is not None, "RunStore.connect() not called"
        await self._db.execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, question, final_answer, iterations, cost_usd, duration_ms,
             stopped_reason, error, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["run_id"],
                payload["question"],
                payload.get("final_answer", ""),
                payload["iterations"],
                payload["cost"]["usd"],
                payload["duration_ms"],
                payload["stopped_reason"],
                payload.get("error"),
                json.dumps(payload, default=str),
            ),
        )
        await self._db.commit()

    async def get(self, run_id: str) -> dict[str, Any] | None:
        assert self._db is not None, "RunStore.connect() not called"
        cur = await self._db.execute(
            "SELECT payload_json FROM runs WHERE run_id = ?", (run_id,)
        )
        row = await cur.fetchone()
        await cur.close()
        if row is None:
            return None
        return json.loads(row[0])
